import ctypes
import time
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm


# =========================================================
# Paths
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

DLL_PATH = PROJECT_ROOT / "pgn_halfkp_training.dll"

# Your PGN is inside src/nnue
PGN_PATH = SCRIPT_DIR / "lichess_db_standard_rated_2017-01_mario.pgn"

OUTPUT_DIR = PROJECT_ROOT / "nnue_training_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# =========================================================
# HalfKP NNUE configuration
# =========================================================

INPUTS = 40960
MAX_ACTIVE = 32

L1 = 256
L2 = 32
L3 = 32

BATCH_SIZE = 2048
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0

MAX_STEPS = 1000

# 4 checkpoints + 1 final model = 5 model files total
SNAPSHOT_COUNT = 4
CHECKPOINT_STEPS = {
    int(MAX_STEPS * i / (SNAPSHOT_COUNT + 1))
    for i in range(1, SNAPSHOT_COUNT + 1)
}

GRAD_CLIP_NORM = 10.0

RESUME_FROM = None
# Example:
# RESUME_FROM = OUTPUT_DIR / "halfkp_nnue_step_10000.pt"


# =========================================================
# C++ batch structure
# Must match SimpleHalfKPBatch from pgn_training_bridge.hpp
# =========================================================

class SimpleHalfKPBatch(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("max_active", ctypes.c_int),

        ("white_indices", ctypes.POINTER(ctypes.c_int)),
        ("black_indices", ctypes.POINTER(ctypes.c_int)),

        ("white_counts", ctypes.POINTER(ctypes.c_int)),
        ("black_counts", ctypes.POINTER(ctypes.c_int)),

        ("stm", ctypes.POINTER(ctypes.c_float)),
        ("target", ctypes.POINTER(ctypes.c_float)),
    ]


# =========================================================
# Load C++ DLL
# =========================================================

if not DLL_PATH.exists():
    raise FileNotFoundError(f"Could not find DLL: {DLL_PATH}")

lib = ctypes.CDLL(str(DLL_PATH))

lib.pgn_bridge_version.restype = ctypes.c_char_p
print("DLL version:", lib.pgn_bridge_version().decode("utf-8"))

lib.create_pgn_training_reader.restype = ctypes.c_void_p
lib.create_pgn_training_reader.argtypes = [ctypes.c_char_p]

lib.get_next_pgn_training_batch.restype = ctypes.POINTER(SimpleHalfKPBatch)
lib.get_next_pgn_training_batch.argtypes = [ctypes.c_void_p, ctypes.c_int]

lib.destroy_simple_halfkp_batch.restype = None
lib.destroy_simple_halfkp_batch.argtypes = [ctypes.POINTER(SimpleHalfKPBatch)]

lib.destroy_pgn_training_reader.restype = None
lib.destroy_pgn_training_reader.argtypes = [ctypes.c_void_p]


# =========================================================
# HalfKP NNUE model
# =========================================================

class HalfKPNNUE(nn.Module):
    def __init__(self):
        super().__init__()

        # Sparse HalfKP feature transformer.
        # Builds accumulator by summing selected rows from a 40960 x 256 table.
        self.feature_transformer = nn.EmbeddingBag(
            INPUTS,
            L1,
            mode="sum",
            sparse=True,
        )

        # Shared bias added to both white and black accumulators.
        self.feature_transformer_bias = nn.Parameter(torch.zeros(L1))

        # NNUE-style dense layers.
        self.hidden1 = nn.Linear(L1 * 2, L2)
        self.hidden2 = nn.Linear(L2, L3)
        self.output = nn.Linear(L3, 1)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.feature_transformer.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.feature_transformer_bias)

        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.zeros_(self.hidden1.bias)

        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.zeros_(self.hidden2.bias)

        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def clipped_relu(x):
        return torch.clamp(x, 0.0, 1.0)

    def forward(self, white_features, white_offsets, black_features, black_offsets, stm):
        # Build white and black accumulators.
        white_acc = self.feature_transformer(white_features, white_offsets)
        black_acc = self.feature_transformer(black_features, black_offsets)

        white_acc = white_acc + self.feature_transformer_bias
        black_acc = black_acc + self.feature_transformer_bias

        # Side-to-move perspective:
        # white to move -> [white_acc, black_acc]
        # black to move -> [black_acc, white_acc]
        white_to_move_input = torch.cat([white_acc, black_acc], dim=1)
        black_to_move_input = torch.cat([black_acc, white_acc], dim=1)

        stm = stm.view(-1, 1)

        x = stm * white_to_move_input + (1.0 - stm) * black_to_move_input

        x = self.clipped_relu(x)
        x = self.clipped_relu(self.hidden1(x))
        x = self.clipped_relu(self.hidden2(x))

        return self.output(x).squeeze(1)


# =========================================================
# Convert fixed C++ feature arrays to PyTorch EmbeddingBag format
# =========================================================

def flatten_indices(indices, counts, device):
    flat = []
    offsets = []
    current_offset = 0

    for row, count in zip(indices, counts):
        offsets.append(current_offset)

        count = int(count.item())

        for j in range(count):
            value = int(row[j].item())

            if value >= 0:
                flat.append(value)
                current_offset += 1

    # Safety fallback. Normally this should not happen.
    if len(flat) == 0:
        flat.append(0)

    return (
        torch.tensor(flat, dtype=torch.long, device=device),
        torch.tensor(offsets, dtype=torch.long, device=device),
    )


# =========================================================
# Convert C++ batch pointer to PyTorch tensors
# =========================================================

def convert_batch(batch_ptr):
    batch = batch_ptr.contents

    size = batch.size
    max_active = batch.max_active

    white = torch.empty((size, max_active), dtype=torch.long)
    black = torch.empty((size, max_active), dtype=torch.long)

    white_counts = torch.empty(size, dtype=torch.long)
    black_counts = torch.empty(size, dtype=torch.long)

    stm = torch.empty(size, dtype=torch.float32)
    target = torch.empty(size, dtype=torch.float32)

    for i in range(size):
        white_counts[i] = batch.white_counts[i]
        black_counts[i] = batch.black_counts[i]

        stm[i] = batch.stm[i]
        target[i] = batch.target[i]

        for j in range(max_active):
            white[i, j] = batch.white_indices[i * max_active + j]
            black[i, j] = batch.black_indices[i * max_active + j]

    return white, black, white_counts, black_counts, stm, target


# =========================================================
# Checkpoint helpers
# =========================================================

def checkpoint_payload(model, sparse_optimizer, dense_optimizer, step, total_positions):
    return {
        "step": step,
        "total_positions": total_positions,

        "model_state_dict": model.state_dict(),
        "sparse_optimizer_state_dict": sparse_optimizer.state_dict(),
        "dense_optimizer_state_dict": dense_optimizer.state_dict(),

        "feature_type": "HalfKP",
        "input_size": INPUTS,
        "max_active": MAX_ACTIVE,

        "l1": L1,
        "l2": L2,
        "l3": L3,

        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_steps": MAX_STEPS,
    }


def save_checkpoint(model, sparse_optimizer, dense_optimizer, step, total_positions):
    checkpoint_path = OUTPUT_DIR / f"halfkp_nnue_step_{step}.pt"

    torch.save(
        checkpoint_payload(
            model,
            sparse_optimizer,
            dense_optimizer,
            step,
            total_positions,
        ),
        checkpoint_path,
    )

    tqdm.write(f"Saved checkpoint: {checkpoint_path}")


def save_final_model(model, sparse_optimizer, dense_optimizer, step, total_positions):
    final_path = OUTPUT_DIR / "halfkp_nnue_final.pt"

    torch.save(
        checkpoint_payload(
            model,
            sparse_optimizer,
            dense_optimizer,
            step,
            total_positions,
        ),
        final_path,
    )

    tqdm.write(f"Saved final model: {final_path}")


def load_checkpoint_if_needed(model, sparse_optimizer, dense_optimizer, device):
    if RESUME_FROM is None:
        return 0, 0

    checkpoint_path = Path(RESUME_FROM)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    sparse_optimizer.load_state_dict(checkpoint["sparse_optimizer_state_dict"])
    dense_optimizer.load_state_dict(checkpoint["dense_optimizer_state_dict"])

    step = int(checkpoint.get("step", 0))
    total_positions = int(checkpoint.get("total_positions", 0))

    print(f"Resumed from checkpoint: {checkpoint_path}")
    print(f"Resume step: {step}")
    print(f"Resume total positions: {total_positions}")

    return step, total_positions


# =========================================================
# Main training loop
# =========================================================

def train(pgn_path):
    pgn_path = Path(pgn_path).resolve()

    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=================================================")
    print("HalfKP NNUE training")
    print("=================================================")
    print(f"Using device: {device}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"PGN file: {pgn_path}")
    print(f"DLL: {DLL_PATH}")
    print("Feature type: HalfKP")
    print(f"Input size: {INPUTS}")
    print(f"Network: {INPUTS} sparse -> {L1} accumulators -> {L1 * 2} -> {L2} -> {L3} -> 1")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Checkpoint steps: {sorted(CHECKPOINT_STEPS)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=================================================")

    reader = lib.create_pgn_training_reader(str(pgn_path).encode("utf-8"))

    if not reader:
        raise RuntimeError("Could not open PGN reader from C++ DLL.")

    model = HalfKPNNUE().to(device)

    sparse_optimizer = torch.optim.SparseAdam(
        model.feature_transformer.parameters(),
        lr=LEARNING_RATE,
    )

    dense_optimizer = torch.optim.Adam(
        [model.feature_transformer_bias]
        + list(model.hidden1.parameters())
        + list(model.hidden2.parameters())
        + list(model.output.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    step, total_positions = load_checkpoint_if_needed(
        model,
        sparse_optimizer,
        dense_optimizer,
        device,
    )

    loss_function = nn.MSELoss()

    start_time = time.time()
    last_loss = None

    progress = tqdm(
        total=MAX_STEPS,
        initial=step,
        desc="Training HalfKP NNUE",
        unit="batch",
        dynamic_ncols=True,
    )

    try:
        while step < MAX_STEPS:
            batch_ptr = lib.get_next_pgn_training_batch(reader, BATCH_SIZE)

            if not batch_ptr:
                tqdm.write("No more batches available from PGN reader.")
                break

            white, black, white_counts, black_counts, stm, target = convert_batch(batch_ptr)

            lib.destroy_simple_halfkp_batch(batch_ptr)

            batch_size = target.shape[0]
            total_positions += batch_size

            white_features, white_offsets = flatten_indices(
                white,
                white_counts,
                device,
            )

            black_features, black_offsets = flatten_indices(
                black,
                black_counts,
                device,
            )

            stm = stm.to(device)
            target = target.to(device)

            prediction = model(
                white_features,
                white_offsets,
                black_features,
                black_offsets,
                stm,
            )

            loss = loss_function(prediction, target)
            last_loss = loss.item()

            sparse_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [model.feature_transformer_bias]
                + list(model.hidden1.parameters())
                + list(model.hidden2.parameters())
                + list(model.output.parameters()),
                GRAD_CLIP_NORM,
            )

            sparse_optimizer.step()
            dense_optimizer.step()

            step += 1
            progress.update(1)

            elapsed = time.time() - start_time
            positions_per_second = total_positions / elapsed if elapsed > 0 else 0.0

            progress.set_postfix(
                {
                    "loss": f"{last_loss:.6f}",
                    "positions": total_positions,
                    "pos/s": f"{positions_per_second:.1f}",
                }
            )

            if step in CHECKPOINT_STEPS:
                save_checkpoint(
                    model,
                    sparse_optimizer,
                    dense_optimizer,
                    step,
                    total_positions,
                )

    finally:
        progress.close()
        lib.destroy_pgn_training_reader(reader)

    save_final_model(
        model,
        sparse_optimizer,
        dense_optimizer,
        step,
        total_positions,
    )

    print("=================================================")
    print("Training finished")
    print(f"Total training steps: {step}")
    print(f"Total positions used: {total_positions}")
    if last_loss is not None:
        print(f"Final loss: {last_loss:.6f}")
    print("=================================================")


if __name__ == "__main__":
    train(PGN_PATH)
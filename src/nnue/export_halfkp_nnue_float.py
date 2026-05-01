import struct
from pathlib import Path

import torch
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

INPUT_PT = PROJECT_ROOT / "nnue_training_output" / "halfkp_nnue_final.pt"
OUTPUT_BIN = PROJECT_ROOT / "halfkp_nnue_float.bin"

INPUTS = 40960
L1 = 256
L2 = 32
L3 = 32


def write_int32(file, value):
    file.write(struct.pack("<i", int(value)))


def write_float_array(file, array):
    array = np.asarray(array, dtype=np.float32)
    file.write(array.tobytes(order="C"))


def main():
    if not INPUT_PT.exists():
        raise FileNotFoundError(f"Model not found: {INPUT_PT}")

    checkpoint = torch.load(INPUT_PT, map_location="cpu")

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    else:
        state = checkpoint

    ft_weight = state["feature_transformer.weight"].cpu().numpy().astype(np.float32)
    ft_bias = state["feature_transformer_bias"].cpu().numpy().astype(np.float32)

    h1_weight = state["hidden1.weight"].cpu().numpy().astype(np.float32)
    h1_bias = state["hidden1.bias"].cpu().numpy().astype(np.float32)

    h2_weight = state["hidden2.weight"].cpu().numpy().astype(np.float32)
    h2_bias = state["hidden2.bias"].cpu().numpy().astype(np.float32)

    out_weight = state["output.weight"].cpu().numpy().astype(np.float32)
    out_bias = state["output.bias"].cpu().numpy().astype(np.float32)

    print("Checking shapes...")
    print("feature_transformer.weight:", ft_weight.shape)
    print("feature_transformer_bias:", ft_bias.shape)
    print("hidden1.weight:", h1_weight.shape)
    print("hidden1.bias:", h1_bias.shape)
    print("hidden2.weight:", h2_weight.shape)
    print("hidden2.bias:", h2_bias.shape)
    print("output.weight:", out_weight.shape)
    print("output.bias:", out_bias.shape)

    if ft_weight.shape != (INPUTS, L1):
        raise RuntimeError(f"Wrong feature transformer shape: {ft_weight.shape}")

    if ft_bias.shape != (L1,):
        raise RuntimeError(f"Wrong feature transformer bias shape: {ft_bias.shape}")

    if h1_weight.shape != (L2, L1 * 2):
        raise RuntimeError(f"Wrong hidden1 weight shape: {h1_weight.shape}")

    if h1_bias.shape != (L2,):
        raise RuntimeError(f"Wrong hidden1 bias shape: {h1_bias.shape}")

    if h2_weight.shape != (L3, L2):
        raise RuntimeError(f"Wrong hidden2 weight shape: {h2_weight.shape}")

    if h2_bias.shape != (L3,):
        raise RuntimeError(f"Wrong hidden2 bias shape: {h2_bias.shape}")

    if out_weight.shape != (1, L3):
        raise RuntimeError(f"Wrong output weight shape: {out_weight.shape}")

    if out_bias.shape != (1,):
        raise RuntimeError(f"Wrong output bias shape: {out_bias.shape}")

    with open(OUTPUT_BIN, "wb") as file:
        file.write(b"HKF1")

        write_int32(file, INPUTS)
        write_int32(file, L1)
        write_int32(file, L2)
        write_int32(file, L3)

        write_float_array(file, ft_weight.reshape(-1))
        write_float_array(file, ft_bias)

        write_float_array(file, h1_weight.reshape(-1))
        write_float_array(file, h1_bias)

        write_float_array(file, h2_weight.reshape(-1))
        write_float_array(file, h2_bias)

        write_float_array(file, out_weight.reshape(-1))
        write_float_array(file, out_bias)

    print(f"Exported NNUE binary: {OUTPUT_BIN}")


if __name__ == "__main__":
    main()
import struct
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

INPUT_PT = PROJECT_ROOT / "nnue_training_output" / "halfkp_nnue_final.pt"
OUTPUT_BIN = PROJECT_ROOT / "halfkp_nnue_quantized.bin"

INPUTS = 40960
L1 = 256
L2 = 32
L3 = 32

QUANT_SCALE = 1024


def write_int32(file, value):
    file.write(struct.pack("<i", int(value)))


def write_int16_array(file, array):
    array = np.asarray(array, dtype=np.int16)
    file.write(array.tobytes(order="C"))


def write_int32_array(file, array):
    array = np.asarray(array, dtype=np.int32)
    file.write(array.tobytes(order="C"))


def quantize_weight_to_int16(array, scale):
    array = np.asarray(array, dtype=np.float32)
    quantized = np.round(array * scale)

    quantized = np.clip(
        quantized,
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    )

    return quantized.astype(np.int16)


def quantize_bias_to_int32(array, scale):
    array = np.asarray(array, dtype=np.float32)
    quantized = np.round(array * scale)

    quantized = np.clip(
        quantized,
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
    )

    return quantized.astype(np.int32)


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

    print("Quantizing weights...")
    print(f"Quantization scale: {QUANT_SCALE}")

    ft_weight_q = quantize_weight_to_int16(ft_weight, QUANT_SCALE)
    ft_bias_q = quantize_bias_to_int32(ft_bias, QUANT_SCALE)

    h1_weight_q = quantize_weight_to_int16(h1_weight, QUANT_SCALE)
    h1_bias_q = quantize_bias_to_int32(h1_bias, QUANT_SCALE)

    h2_weight_q = quantize_weight_to_int16(h2_weight, QUANT_SCALE)
    h2_bias_q = quantize_bias_to_int32(h2_bias, QUANT_SCALE)

    out_weight_q = quantize_weight_to_int16(out_weight, QUANT_SCALE)
    out_bias_q = quantize_bias_to_int32(out_bias, QUANT_SCALE)

    with open(OUTPUT_BIN, "wb") as file:
        file.write(b"HKQ1")

        write_int32(file, INPUTS)
        write_int32(file, L1)
        write_int32(file, L2)
        write_int32(file, L3)
        write_int32(file, QUANT_SCALE)

        write_int16_array(file, ft_weight_q.reshape(-1))
        write_int32_array(file, ft_bias_q)

        write_int16_array(file, h1_weight_q.reshape(-1))
        write_int32_array(file, h1_bias_q)

        write_int16_array(file, h2_weight_q.reshape(-1))
        write_int32_array(file, h2_bias_q)

        write_int16_array(file, out_weight_q.reshape(-1))
        write_int32_array(file, out_bias_q)

    print(f"Exported quantized NNUE binary: {OUTPUT_BIN}")


if __name__ == "__main__":
    main()
"""
export_weights.py
-----------------
Exports ConvNetRELU parameters for HLS / C++ inference.

Outputs
-------
By default the script writes:

  weights.h
      BN-folded Conv weights + biases, and FC weights + biases,
      emitted as C/C++ arrays for direct inclusion in the HLS project.

If --export_mem is specified, the script also writes one .mem file per tensor:

  mem/conv1_w.mem, mem/conv1_b.mem, ...
  mem/fc_w.mem,    mem/fc_b.mem

Each .mem file contains one fixed-point integer per line. 

BatchNorm folding
-----------------
At inference, BatchNorm is a per-channel linear transform:

    y = gamma * (x - mean) / sqrt(var + eps) + beta

When folded into the preceding convolution y = Wx + b_conv, for each
output channel c:

    scale_c = gamma_c / sqrt(var_c + eps)
    w_fold  = w_conv * scale_c
    b_fold  = (b_conv - mean_c) * scale_c + beta_c

This is exact. After folding, the hardware only needs to implement:

    Conv -> ReLU -> Pool

with no separate BatchNorm logic, reducing hardware complexity and latency.

Quantisation modes
------------------
Use --quant to select the export format:

  --quant float
      Export full-precision float values.

  --quant q<I>.<F>
      Export values quantised to ap_fixed<I+F, I>.
      Example:
          q6.10 -> ap_fixed<16, 6>

For fixed-point export:
- values are rounded to the selected fixed-point grid
- values outside the representable range are clipped
- a warning is printed if clipping occurs

Examples:
  q6.10  -> ap_fixed<16, 6>
  q4.12  -> ap_fixed<16, 4>

.mem export
-----------
When --export_mem is enabled, each quantised tensor is also written as a
flattened memory file.

Format:
- one value per line
- fixed-width two's-complement hex word
- row-major flattening order
- same tensor traversal order as the C arrays

For example, in q6.10 (16 bits total):
    0.5   -> 0200
   -0.5   -> FE00

For example, a tensor declared as:

    W_CONV1[OC][IC][KH][KW]

is flattened in this order:

    for oc
      for ic
        for kh
          for kw

Notes
-----
- .mem export requires a fixed-point mode such as q6.10.
- .mem export is not supported with --quant float.
- weights.h is still generated even when .mem export is enabled.

Usage
-----
    python export_weights.py
        Export float weights to ./hls_src/weights.h

    python export_weights.py --quant q6.10
        Export q6.10 weights to ./hls_src/weights.h

    python export_weights.py --quant q6.10 --export_mem
        Export q6.10 weights to weights.h and per-tensor .mem files

    python export_weights.py --quant q4.12 --checkpoint path/to/model.pth
        Export q4.12 weights from a specific checkpoint

    python export_weights.py --checkpoint path/to/model.pth --out_dir ./hls_src
        Export to a custom output directory
"""

import argparse
import os
import re
import numpy as np
import torch
import torch.nn as nn


# ── Model definition ──────────────────────────────────────────────────────────

class ConvNetReluV2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),   # 32 -> 16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2, 2),   # 16 -> 8

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ── Quantisation helpers ──────────────────────────────────────────────────────

def parse_quant(quant_str: str):
    """
    Parse a quantisation specifier and return (dtype_str, int_bits, frac_bits).

    'float'   → ('float', None, None)
    'q6.10'   → ('ap_fixed<16, 6>', 6, 10)
    """
    if quant_str.lower() == "float":
        return "float", None, None

    m = re.fullmatch(r"q(\d+)\.(\d+)", quant_str.lower())
    if not m:
        raise ValueError(
            f"Unrecognised --quant '{quant_str}'. "
            "Use 'float' or 'q<I>.<F>' e.g. 'q6.10'."
        )
    int_bits  = int(m.group(1))
    frac_bits = int(m.group(2))
    total     = int_bits + frac_bits
    dtype_str = f"ap_fixed<{total}, {int_bits}>"
    return dtype_str, int_bits, frac_bits


def quantise(arr: np.ndarray, int_bits: int, frac_bits: int) -> np.ndarray:
    """
    Quantise float array to ap_fixed<int_bits+frac_bits, int_bits>.
    Returns float64 array with values snapped to the fixed-point grid.
    """
    total  = int_bits + frac_bits
    scale  = 2 ** frac_bits
    max_v  =  (2 ** (total - 1) - 1) / scale
    min_v  = -(2 ** (total - 1))     / scale
    clipped = np.clip(arr, min_v, max_v)
    n_clip  = int(np.sum(np.abs(arr) > max_v))
    if n_clip:
        print(f"  ⚠  {n_clip} values clipped to [{min_v:.4f}, {max_v:.4f}]"
              f" — consider larger I bits")
    return np.round(clipped * scale) / scale

def fixed_to_int(arr: np.ndarray, int_bits: int, frac_bits: int) -> np.ndarray:
    """
    Convert float/fixed-grid values into signed integer representation for
    ap_fixed<int_bits+frac_bits, int_bits> before hex/two's-complement export.
    """
    
    total = int_bits + frac_bits
    scale = 2 ** frac_bits

    max_int = 2 ** (total - 1) - 1
    min_int = -(2 ** (total - 1))

    q = np.round(arr * scale).astype(np.int64)
    q = np.clip(q, min_int, max_int)
    return q


def write_mem_file(path: str, arr: np.ndarray, int_bits: int, frac_bits: int) -> None:
    """
    Write one tensor to a .mem file, one fixed-width two's-complement hex word
    per line.

    Example for q6.10 -> ap_fixed<16, 6>:
        0.5   -> 0200
       -0.5   -> FE00
        -1.0  -> FC00
    """
    total_bits = int_bits + frac_bits
    hex_digits = (total_bits + 3) // 4
    mask = (1 << total_bits) - 1

    arr_int = fixed_to_int(arr, int_bits, frac_bits)

    with open(path, "w") as f:
        for v in arr_int.flatten():
            raw = int(v) & mask
            f.write(f"{raw:0{hex_digits}X}\n")

# ── BN folding ────────────────────────────────────────────────────────────────

def fold_bn(w_conv: np.ndarray, b_conv: np.ndarray,
            bn_gamma: np.ndarray, bn_beta: np.ndarray,
            bn_mean: np.ndarray,  bn_var: np.ndarray,
            eps: float = 1e-5):
    """
    Fold BatchNorm into conv weight and bias.
      w_conv : [OC, IC, KH, KW]
      b_conv : [OC]
    Returns w_folded [OC, IC, KH, KW], b_folded [OC].
    """
    scale  = bn_gamma / np.sqrt(bn_var + eps)
    w_fold = w_conv * scale[:, None, None, None]
    b_fold = (b_conv - bn_mean) * scale + bn_beta
    return w_fold, b_fold


def extract_conv_bn(features, conv_idx: int, bn_idx: int):
    conv = features[conv_idx]
    bn   = features[bn_idx]

    w = conv.weight.detach().cpu().numpy()
    b = (conv.bias.detach().cpu().numpy()
         if conv.bias is not None
         else np.zeros(w.shape[0], dtype=np.float32))

    return fold_bn(
        w, b,
        bn.weight.detach().cpu().numpy(),
        bn.bias.detach().cpu().numpy(),
        bn.running_mean.detach().cpu().numpy(),
        bn.running_var.detach().cpu().numpy(),
    )


# ── C array formatting ────────────────────────────────────────────────────────

def array_to_c(name: str, arr: np.ndarray, dtype: str) -> str:
    """
    Render numpy array as a C static const declaration.
    Shape is expressed as multi-dimensional C brackets; storage is row-major,
    matching numpy's default and C's multi-dimensional array layout.
    """
    shape_str = "".join(f"[{d}]" for d in arr.shape)
    vals      = ", ".join(f"{v:.8f}f" for v in arr.flatten())
    return f"static const {dtype} {name}{shape_str} = {{{vals}}};\n"


# ── Main export ───────────────────────────────────────────────────────────────

def export(checkpoint: str, out_dir: str,
           quant_str: str = "float", num_classes: int = 10,
           export_mem: bool = False) -> None:

    os.makedirs(out_dir, exist_ok=True)
    mem_dir = os.path.join(out_dir, "mem")
    if export_mem:
        os.makedirs(mem_dir, exist_ok=True)

    dtype_str, int_bits, frac_bits = parse_quant(quant_str)
    is_float = (int_bits is None)

    def maybe_quantise(arr):
        if is_float:
            return arr
        return quantise(arr, int_bits, frac_bits)

    # ── Load model ────────────────────────────────────────────────────────────
    model = ConvNetReluV2(num_classes=num_classes)
    state = torch.load(checkpoint, map_location="cpu")
    # support plain state-dict or wrapped checkpoints
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint : {checkpoint}")
    print(f"Quantisation      : {quant_str}  →  dtype = {dtype_str}\n")

    # ── features block: Conv+BN pairs ─────────────────────────────────────────
    layers = [
        ("CONV1",  0,  1),
        ("CONV2",  3,  4),
        ("CONV3",  7,  8),
        ("CONV4", 10, 11),
        ("CONV5", 14, 15),
        ("CONV6", 17, 18),
    ]

    lines = [
        "#ifndef WEIGHTS_H\n",
        "#define WEIGHTS_H\n",
        "\n",
        "// Auto-generated by export_weights.py — do not edit by hand.\n",
        f"// Checkpoint : {checkpoint}\n",
        f"// Quantisation: {quant_str}  ({dtype_str})\n",
        "// BatchNorm has been folded into each Conv layer.\n",
        "// Activation: ReLU (no LUT needed).\n",
        "\n",
    ]

    if not is_float:
        lines += [
            "#include <ap_fixed.h>\n",
            f"typedef {dtype_str} weight_t;\n",
            "\n",
        ]
        dtype_decl = "weight_t"
    else:
        dtype_decl = "float"

    print("Folding BatchNorm into conv layers:")

    for name, ci, bi in layers:
        w, b = extract_conv_bn(model.features, ci, bi)
        w_q = maybe_quantise(w)
        b_q = maybe_quantise(b)

        print(f"  {name}: weight {w.shape}  bias {b.shape}"
            f"   |w|_max={np.abs(w).max():.4f}  |b|_max={np.abs(b).max():.4f}")

        lines.append(f"// {name}: weight {list(w.shape)}, BN folded\n")
        lines.append(array_to_c(f"W_{name}", w_q, dtype_decl))
        lines.append(array_to_c(f"B_{name}", b_q, dtype_decl))
        lines.append("\n")

        if export_mem:
            if is_float:
                raise ValueError("Cannot export .mem files with --quant float. Use q<I>.<F> such as q6.10.")
            write_mem_file(os.path.join(mem_dir, f"{name.lower()}_w.mem"), w_q, int_bits, frac_bits)
            write_mem_file(os.path.join(mem_dir, f"{name.lower()}_b.mem"), b_q, int_bits, frac_bits)

    # ── FC layer ──────────────────────────────────────────────────────────────
    fc   = model.classifier[1]
    fc_w = fc.weight.detach().cpu().numpy()
    fc_b = fc.bias.detach().cpu().numpy()

    fc_w_q = maybe_quantise(fc_w)
    fc_b_q = maybe_quantise(fc_b)

    print(f"\n  FC:    weight {fc_w.shape}  bias {fc_b.shape}"
        f"   |w|_max={np.abs(fc_w).max():.4f}  |b|_max={np.abs(fc_b).max():.4f}")

    lines.append("// FC layer\n")
    lines.append(array_to_c("W_FC", fc_w_q, dtype_decl))
    lines.append(array_to_c("B_FC", fc_b_q, dtype_decl))
    lines.append("\n")

    if export_mem:
        if is_float:
            raise ValueError("Cannot export .mem files with --quant float. Use q<I>.<F> such as q6.10.")
        write_mem_file(os.path.join(mem_dir, "fc_w.mem"), fc_w_q, int_bits, frac_bits)
        write_mem_file(os.path.join(mem_dir, "fc_b.mem"), fc_b_q, int_bits, frac_bits)
    # ── Write output ──────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "weights.h")
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"\n  wrote {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_params = sum(
        np.prod(model.features[ci].weight.shape) +
        model.features[ci].weight.shape[0]          # bias
        for _, ci, _ in layers
    ) + fc_w.size + fc_b.size
    print(f"\nTotal parameters exported : {total_params:,}")
    print(f"Include in your HLS project: weights.h")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export ConvNetRELU weights for C++/HLS"
    )
    parser.add_argument(
        "--checkpoint",  default="model/best_model_relu.pth",
        help="Path to .pth checkpoint"
    )
    parser.add_argument(
        "--out_dir",     default="./hls_src",
        help="Output directory"
    )
    parser.add_argument(
        "--quant",       default="float",
        help=(
            "Quantisation mode: 'float' for full precision, "
            "or 'q<I>.<F>' for ap_fixed<I+F,I>  (e.g. q6.10, q4.12)"
        )
    )
    parser.add_argument(
        "--num_classes", default=10, type=int,
        help="Number of output classes"
    )
    parser.add_argument(
        "--export_mem", action="store_true",
        help="Also export each tensor as a .mem file with one two's-complement hex word per line"
    )
    args = parser.parse_args()
    export(args.checkpoint, args.out_dir, args.quant, args.num_classes, args.export_mem)

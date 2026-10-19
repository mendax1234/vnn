# CNN architecture and parameter export

## Network architecture

The CNN takes a CIFAR-10 image with dimensions 3 × 32 × 32 and produces ten
class scores. Output dimensions below use channel-height-width order (C × H × W),
with the batch dimension omitted.

| Layer type | Output dimensions | MACs |
| --- | --- | ---: |
| Conv2D + ReLU | 32 × 32 × 32 | 884,736 |
| Conv2D + ReLU | 32 × 32 × 32 | 9,437,184 |
| MaxPool | 32 × 16 × 16 | – |
| Conv2D + ReLU | 64 × 16 × 16 | 4,718,592 |
| Conv2D + ReLU | 64 × 16 × 16 | 9,437,184 |
| MaxPool | 64 × 8 × 8 | – |
| Conv2D + ReLU | 128 × 8 × 8 | 4,718,592 |
| Conv2D + ReLU | 128 × 8 × 8 | 9,437,184 |
| Global AvgPool | 128 | – |
| Fully Connected | 10 | 1,280 |
| **Total** | | **38,634,752** |

Each convolution uses a 3 × 3 kernel, stride 1, and padding 1. Max pooling uses
a 2 × 2 window with stride 2. BatchNorm is folded into the convolution parameters
for inference, and each convolution includes ReLU activation. The MAC count covers
convolutions and the fully connected layer; one MAC counts as two operations when
reporting throughput in GOPS.

## Files and parameters

| File | Purpose |
| --- | --- |
| `model/export_weights.py` | PyTorch definition, BatchNorm folding, and parameter export |
| `model/checkpoints/cnn.pth` | Trained checkpoint |
| `model/test_batch.bin` | CIFAR-10 binary test set |
| `model/weights_q6_10.bin` | 288,298 little-endian signed int16 parameter words |
| `model/cpu/` | Q6.10 CPU inference and benchmarking |
| `model/cuda/` | FP16 and Q6.10 CUDA inference |

Training code is not included. Build and execution instructions are separated by
implementation:

- [CPU inference](cpu/README.md)
- [CUDA inference](cuda/README.md)

## Parameter export

Run from the repository root with NumPy and PyTorch installed:

```sh
python -m pip install -r model/requirements.txt
python model/export_weights.py --checkpoint model/checkpoints/cnn.pth --out_dir model/build/export --quant q6.10 --export_mem
```

Always supply `--checkpoint` and `--out_dir` for this layout. Change the checkpoint
argument to evaluate another trained model. `--quant q6.10` exports 16-bit values;
`--export_mem` writes 14 tensor files under `model/build/export/mem/` and a header
at `model/build/export/weights.h`. Use `--quant float` without `--export_mem` for
a floating-point header.

The executable parameter blob is `model/weights_q6_10.bin`; exporting tensor files
does not automatically replace that blob. If you update model parameters, keep
the blob and tensor files in the same order: Conv1-Conv6 weights, FC weights,
Conv1-Conv6 biases, then FC biases. Update the CUDA FP16 header separately when
using a different checkpoint.

## Evaluation data

### CIFAR-10 test images

`model/test_batch.bin` contains 10,000 test images in 30,730,000 bytes. Each
3,073-byte record consists of one label byte followed by 3,072 image bytes in
channel-height-width order (`3 x 32 x 32`).

SHA-256: `8e2eb146ae340b09e24670f29cabc6326dba54da8789dab6768acf480273f65b`

### Model parameters

| File | Format | Use |
| --- | --- | --- |
| `model/weights_q6_10.bin` | 288,298 little-endian signed int16 words; 576,596 bytes | CPU, CUDA Q6.10, and PYNQ inference |
| `model/checkpoints/cnn.pth` | PyTorch checkpoint | Parameter export |
| `model/cuda/weights.h` | Float arrays with Q6.10-rounded values | CUDA FP16 build |

Parameter stream order is Conv1-Conv6 weights, FC weights, Conv1-Conv6 biases,
and FC biases. Interpret a Q6.10 word as its signed integer value divided by 1024.

### Choose input paths

Run CPU and CUDA commands with explicit dataset and parameter arguments, as shown
in their component guides.

For PYNQ inference, copy the dataset and parameter blob to the configured absolute
paths or edit `CIFAR_BIN` and `PARAM_BIN` in the corresponding runner. See
[board configuration](../pynq/README.md) for overlay and IP requirements.

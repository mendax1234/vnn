# CNN model and parameter export

The model has six 3x3 convolutions with channels 32, 32, 64, 64, 128, and 128.
Max pooling follows Conv2 and Conv4, then global average pooling and a 10-class
fully connected layer. Parameters are exported with BatchNorm folded into the
convolutions.

## Files and parameters

| File | Purpose |
| --- | --- |
| `model/export_weights.py` | PyTorch definition, BatchNorm folding, and parameter export |
| `model/checkpoints/cnn.pth` | Trained checkpoint |
| `model/test_batch.bin` | CIFAR-10 binary test set |
| `model/weights_q6_10.bin` | 288,298 little-endian signed int16 parameter words |
| `model/cpu/` | Q6.10 CPU inference and benchmarking |
| `model/cuda/` | FP16 and Q6.10 CUDA inference |
| `tb/data/` | The same parameters as 14 tensor hex files |

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

### CNN-V2 parameters

| File | Format | Use |
| --- | --- | --- |
| `model/weights_q6_10.bin` | 288,298 little-endian signed int16 words; 576,596 bytes | CPU, CUDA Q6.10, and PYNQ inference |
| `tb/data/conv1_w.mem` through `conv6_w.mem`, plus `fc_w.mem` | One 16-bit hexadecimal word per line | HLS and RTL simulation weights |
| `tb/data/conv1_b.mem` through `conv6_b.mem`, plus `fc_b.mem` | One 16-bit hexadecimal word per line | HLS and RTL simulation biases |
| `model/checkpoints/cnn.pth` | PyTorch checkpoint | Parameter export |
| `model/cuda/weights.h` | Float arrays with Q6.10-rounded values | CUDA FP16 build |

Parameter stream order is Conv1-Conv6 weights, FC weights, Conv1-Conv6 biases,
and FC biases. Interpret a Q6.10 word as its signed integer value divided by 1024.

### Choose input paths

Run CPU and CUDA commands with explicit dataset and parameter arguments, as shown
in their component guides. The HLS flow uses `tb/data/` and `model/test_batch.bin`;
change `mem_dir` and `cifar_bin` in `hls/run_hls.tcl` for other locations.

For PYNQ HLS inference, retain the repository layout on the board. For the RTL
runner, copy the dataset and parameter blob to its configured absolute paths or
edit `CIFAR_BIN` and `PARAM_BIN` in `pynq/infer_rtl.py`. See
[board configuration](../pynq/README.md) for overlay and IP requirements.

Additional RTL simulation vectors are not included. Their filenames and required
locations are listed in [the testbench guide](../tb/README.md#required-data).

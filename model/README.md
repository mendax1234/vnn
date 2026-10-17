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

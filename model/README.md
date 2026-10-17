# CNN model and CPU inference

The model has six 3x3 convolutions with channels 32, 32, 64, 64, 128, and 128.
Max pooling follows Conv2 and Conv4, then global average pooling and a 10-class
fully connected layer. The CPU implementation uses Q6.10 arithmetic.

## Files and parameters

| File | Purpose |
| --- | --- |
| `model/export_weights.py` | PyTorch definition, BatchNorm folding, and parameter export |
| `model/checkpoints/cnn.pth` | Trained checkpoint |
| `model/weights_q6_10.bin` | 288,298 little-endian signed int16 parameter words |
| `model/cpu/cnn_fixed.cpp`, `cnn_fixed.h` | Q6.10 inference executable and arithmetic definitions |
| `model/cpu/benchmark.py` | CPU accuracy and timing wrapper |
| `model/cuda/` | FP16 and Q6.10 CUDA inference |
| `tb/data/` | The same parameters as 14 tensor hex files |

Training code is not included. CUDA Q6.10 also includes the CPU arithmetic header.
See [the CUDA guide](cuda/README.md) for GPU build and execution instructions.

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

## CPU build and inference

Run from the repository root. Create `model/cpu/build/`, then build with a C++17
compiler:

```sh
g++ -std=c++17 -O2 model/cpu/cnn_fixed.cpp -o model/cpu/build/cnn_fixed
model/cpu/build/cnn_fixed 100 data/test_batch.bin 1 1 model/weights_q6_10.bin
```

On Windows, append `.exe` to the output and executable names. Build on the target
CPU for performance comparisons. Avoid `-ffast-math` when matching input
preprocessing with hardware.

The executable accepts:

```text
cnn_fixed [images] [cifar.bin] [warmup] [repeats] [params.bin] [dump_directory]
```

Pass dataset and parameter paths explicitly. The optional dump directory must
already exist. To report accuracy and timing through the Python wrapper:

```sh
python model/cpu/benchmark.py 100 --mode fixed --binary model/cpu/build/cnn_fixed --params model/weights_q6_10.bin --cifar data/test_batch.bin --warmup 5 --repeats 3
```

Change `--binary` for your compiled executable, including `.exe` on Windows.
Keep `--mode fixed`: the FP32 CPU executable is not provided. CPU execution does
not require PYNQ, PyTorch, or NumPy; the benchmark uses Python's standard library.

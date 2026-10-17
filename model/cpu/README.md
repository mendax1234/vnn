# Q6.10 CPU inference

The single-threaded C++ implementation evaluates the six-convolution CNN using
Q6.10 inputs, weights, and activations, with wider accumulators. It uses the
included CIFAR-10 test batch and parameter blob.

## Files and requirements

| File | Purpose |
| --- | --- |
| `model/cpu/cnn_fixed.cpp` | Inference executable and timing |
| `model/cpu/cnn_fixed.h` | Fixed-point arithmetic and network execution; also used by CUDA Q6.10 |
| `model/cpu/benchmark.py` | Accuracy and repeated timing reports |
| `model/weights_q6_10.bin` | 288,298 little-endian signed int16 parameter words |
| `model/test_batch.bin` | 10,000 CIFAR-10 test images |

Requires a C++17 compiler. Python 3 is needed only for the benchmark wrapper.
PYNQ, PyTorch, NumPy, and Vitis are not required for CPU inference.

## Build and inference

Run from the repository root. Create `model/cpu/build/`, then build with a C++17
compiler:

```sh
g++ -std=c++17 -O2 model/cpu/cnn_fixed.cpp -o model/cpu/build/cnn_fixed
model/cpu/build/cnn_fixed 100 model/test_batch.bin 1 1 model/weights_q6_10.bin
```

On Windows, append `.exe` to the output and executable names. Build on the target
CPU for performance comparisons. Avoid `-ffast-math` when matching input
preprocessing with hardware.

The executable accepts:

```text
cnn_fixed [images] [cifar.bin] [warmup] [repeats] [params.bin] [dump_directory]
```

Pass dataset and parameter paths explicitly. The optional dump directory must
already exist.

## Benchmarking

To report accuracy and timing through the Python wrapper:

```sh
python model/cpu/benchmark.py 100 --mode fixed --binary model/cpu/build/cnn_fixed --params model/weights_q6_10.bin --cifar model/test_batch.bin --warmup 5 --repeats 3
```

Change `--binary` for your compiled executable, including `.exe` on Windows.
Keep `--mode fixed`: the FP32 CPU executable is not provided. The benchmark
uses Python's standard library.

For model definition and parameter export, see [the model guide](../README.md).
For GPU build and inference, see [the CUDA guide](../cuda/README.md).

# CUDA CNN inference

`cnn_fp16.cu` uses FP16 activations/weights and FP32 accumulation through cuBLAS.
`cnn_q6_10.cu` uses integer CUDA kernels and includes `../cpu/cnn_fixed.h`.

## Build

Install the CUDA Toolkit and a C++ toolchain. FP16 requires cuBLAS and a GPU with
compute capability 7.0 or newer. On Windows, install Visual Studio's Desktop
development with C++ workload. From the repository root in PowerShell:

```powershell
cd model/cuda
./build.ps1 -Precision FP16
./build.ps1 -Precision Q6_10
```

Executables go under `model/cuda/build/`. The build helper discovers Visual Studio;
use `-VisualStudioPath PATH` if needed. `-Architecture native` targets the installed
GPU; use an explicit target such as `-Architecture sm_89` for another deployment.
`./build_q6_10.ps1` is a shortcut for the Q6.10 build.

The FP16 build uses `model/cuda/weights.h`, whose float values are rounded to
Q6.10. Use `-WeightsHeader PATH` to select another float header. Q6.10 loads
`model/weights_q6_10.bin` at runtime.

## Run

Run from `model/cuda/` and pass the included data explicitly:

```powershell
./build/cnn_fp16.exe 100 ../../data/test_batch.bin 5 3
./build/cnn_q6_10.exe 100 ../../data/test_batch.bin 5 3 ../weights_q6_10.bin
```

The last two numbers are warm-up images and timed passes. Full positional options:

```text
cnn_fp16 [images] [cifar.bin] [warmup] [repeats] [reference_images] [logits.bin]
cnn_q6_10 [images] [cifar.bin] [warmup] [repeats] [params.bin] [reference_images] [dump_directory]
```

Use `reference_images` for the built-in CPU comparison. Create the dump directory
before requesting Q6.10 dumps. FP16's optional final argument writes FP32 logits.
Images are processed sequentially with batch size one; reported inference timing
excludes dataset loading, normalization, allocations, and parameter/input transfers.

## FP16 weights from a checkpoint

Install NumPy/PyTorch from `model/requirements.txt`. From `model/cuda/`:

```powershell
python ./export_fp32_weights.py --checkpoint ../checkpoints/cnn.pth --out-dir ../build/fp32
./build.ps1 -Precision FP16 -WeightsHeader ../build/fp32/weights.h
```

Change `--checkpoint` to select a different model. The export folds BatchNorm and
keeps floating-point parameters without Q6.10 rounding. Rebuild FP16 whenever its
header changes. Updating the Q6.10 parameter blob requires no GPU rebuild.

## Linux build

From `model/cuda/`, create `build/` and run:

```sh
nvcc -O3 -std=c++17 -arch=native '-DCNN_WEIGHTS_HEADER="weights.h"' cnn_fp16.cu -lcublas -o build/cnn_fp16
nvcc -O3 -std=c++17 -arch=native cnn_q6_10.cu -o build/cnn_q6_10
```

Use the same inference arguments as above and omit `.exe` from executable names.

# CNN-V2 HLS

The inference graph is:

`3x32x32 -> Conv32 -> Conv32 -> Pool -> Conv64 -> Conv64 -> Pool -> Conv128 -> Conv128 -> GAP -> FC10`

Each convolution is followed by ReLU. Export folds BatchNorm into the convolution
parameters. All paths below are relative to the repository root.

## Design and testbench

| File | Use |
| --- | --- |
| `hls/cnn_accelerator.cpp` | Accelerator composition and simulation-vector dumps |
| `hls/cnn_accelerator.h` | Layer templates and fixed-point types |
| `tb/hls/cnn_accelerator_tb.cpp` | CIFAR-10 C simulation testbench |
| `hls/run_hls.tcl` | C simulation, synthesis, and IP export flow |
| `hls/Makefile` | Command-line flow entry point |

In Vitis HLS, add both accelerator files as design sources and the C testbench
as a testbench source. Add `hls/` to the testbench include search path. Select
`cnn_accelerator` as the top function.

## Configuration

Edit `hls/run_hls.tcl` for your FPGA and timing target:

| Setting | Default | Where to change it |
| --- | --- | --- |
| FPGA part | `xck26-sfvc784-2LV-c` | `part`, or environment variable `HLS_PART` |
| Clock period | 5 ns | `create_clock -period 5` |
| C simulation images | 1 | `num_images`, or `HLS_NUM_IMAGES` |
| Skip C simulation | 0 | `skip_csim`, or `HLS_SKIP_CSIM` |
| Tensor directory | `tb/data/` | `mem_dir` |
| CIFAR-10 input | `data/test_batch.bin` | `cifar_bin` |

C simulation arguments, when configured manually in the GUI, are:

```text
<absolute-path-to-vnn/tb/data> <absolute-path-to-vnn/data/test_batch.bin> <number-of-images>
```

For a configured Vitis installation, run from the repository root:

```sh
vitis-run --mode hls --tcl hls/run_hls.tcl
```

Alternatively, run `make` inside `hls/`. The flow uses a reset project under
`hls/build/`; use that directory only for generated files. Exported IP is under
`hls/build/solution1/impl/ip/`.

## Stream format and vectors

Each accelerator call consumes 288,298 parameter words followed by normalized
CHW image pixels, and returns one class index per image. The AXI stream carries
raw 16-bit words. Parameter order is Conv1-Conv6 weights, FC weights,
Conv1-Conv6 biases, and FC biases. The 14 tensor files are included in `tb/data/`.

Inputs and parameters use `ap_fixed<16,6>`. Accumulator and logit types are declared
in `hls/cnn_accelerator.h`; compare their arithmetic explicitly when validating
against the fixed-point CPU or RTL implementation.

C simulation writes `test_input_convN_relu.mem` and
`test_result_expected_convN_relu.mem` for the first image, where N is 1-6, into
the simulation working directory. Copy them to the locations described in
[the testbench guide](../tb/README.md). This does not generate the full ten-image
RTL system fixtures. Additional post-layer dump paths are defined in
`hls/cnn_accelerator.cpp`; change those output paths if you need those dumps.

C++ syntax checks passed with Vitis headers. Vitis C simulation, synthesis, and
IP export have not been validated for the current repository configuration.

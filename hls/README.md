# HLS implementation

## Design and testbench

| File | Use |
| --- | --- |
| `hls/cnn_accelerator.cpp` | Accelerator composition and simulation-vector dumps |
| `hls/cnn_accelerator.h` | Layer templates and fixed-point types |
| `tb/hls/cnn_accelerator_tb.cpp` | CIFAR-10 C simulation testbench |

## Configuration

Create an HLS component in the Vitis software and select the Vivado IP flow.
Add the accelerator files as design sources and the C testbench as a testbench
source. Configure the component using the following settings:

| Setting | Value | Where to configure it in Vitis |
| --- | --- | --- |
| Top function | `cnn_accelerator` | Design sources |
| FPGA part | `xck26-sfvc784-2LV-c` | Target device / General settings |
| Clock period | 5 ns (200 MHz) | General settings |
| Testbench include path | Absolute path to `hls/` | Testbench compiler flags: `-I"<absolute-path-to-vnn/hls>"` |
| C simulation arguments | Tensor directory, CIFAR-10 file, image count | C Simulation settings |

Set the C simulation arguments to:

```text
"<absolute-path-to-vnn/tb/data>" "<absolute-path-to-vnn/model/test_batch.bin>" 1
```

The final argument is the number of images (1-10,000). Use 1 to generate the
first-image layer test vectors. Run **C Simulation** to check the design, then
run **C Synthesis** and export/package the result as Vivado IP for use in Vivado.

> [!NOTE]
> See the [EE4218 HLS Flow guide](https://nus-ee4218.github.io/Lab_4/2_HLSFLow/)
> for a walkthrough of the Vitis GUI, C simulation, synthesis, and IP packaging.

## Dump inputs and outputs for RTL testbenches

The `CNN_V2_DUMP_CONV1_RELU_VECTORS` through
`CNN_V2_DUMP_CONV6_RELU_VECTORS` definitions at the top of
`hls/cnn_accelerator.cpp` enable the dumps and are already enabled. Run
**C Simulation** with the arguments above. For the first image, it writes the
following files into the simulation working directory:

| Dump | File (replace N with 1-6) |
| --- | --- |
| Layer input, weights, and biases | `test_input_convN_relu.mem` |
| Expected layer output | `test_result_expected_convN_relu.mem` |

Copy each pair to `tb/data/conv_relu/convN/`, creating the directory if needed.
These files are used by the corresponding Conv-ReLU RTL layer testbench. See
[the testbench guide](../tb/README.md) for running the tests.

`CNN_V2_DUMP_POST_LAYER_VECTORS` also enables first-image MaxPool, GAP, and final
class-index dumps. Their output paths are defined in `hls/cnn_accelerator.cpp`;
edit those paths if needed. The layer dumps do not generate the full ten-image
input/output files required by the full CNN RTL testbench.

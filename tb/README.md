# Hardware simulation

All paths below are relative to the repository root.

| Directory | Purpose |
| --- | --- |
| `tb/rtl/unit/` | Six individual fused Conv-ReLU layer tests |
| `tb/rtl/system/` | Full CNN inference and parameter-loader tests |
| `tb/rtl/wrappers/` | AXI-stream interfaces and local memories used to simulate each layer |
| `tb/hls/` | HLS C simulation testbench |
| `tb/data/` | Shared parameter tensors and simulation vectors |

The wrappers instantiate the compute cores under `rtl/` and expose standalone
stream interfaces for the layer tests. Add them as **Simulation Sources**.
The production design uses the sources under `rtl/`.

## Vivado/XSim setup

1. Create a Vivado RTL project for the target FPGA.
2. Add all `.v` files under `rtl/` as Design Sources. Set `cnn_dataflow` as the
   design top.
3. Add `.v` files under `tb/rtl/unit/`, `tb/rtl/system/`, and
   `tb/rtl/wrappers/` as Simulation Sources.
4. Enable `XPM_MEMORY` in the project for `xpm_memory_tdpram` UltraRAM modules.
5. Select one simulation top from the table below.
6. In the XSim Tcl console, use `cd {<absolute-path-to-vnn>}`, then `restart`
   and `run all`. This reloads the memory files using repository-relative paths.

| Test | Simulation top |
| --- | --- |
| Convolution layer N, where N is 1-6 | `tb_myip_convN_relu_unrolled` |
| Full ten-image CNN | `tb_cnn_dataflow` |
| Parameter loading | `tb_cnn_dataflow_weight_loader_v2` |

When selecting a layer test, replace N in the top name with its layer number.
The HLS testbench is configured separately in [the HLS guide](../hls/README.md).

## Required data

All 14 parameter tensors are included directly in `tb/data/`, totaling 288,298
16-bit words. Stream order is Conv1-Conv6 weights, FC weights, Conv1-Conv6 biases,
and FC biases. The parameter-loader test uses these files.

The following image/expected-output vectors are **not included**. Provide them
before running the corresponding tests.

| Test | Required files |
| --- | --- |
| Conv-ReLU layer N | `tb/data/conv_relu/convN/test_input_convN_relu.mem` and `test_result_expected_convN_relu.mem` in the same directory |
| Full CNN | `tb/data/system/test_input_v2_10.mem` and `tb/data/system/test_result_expected_v2_10.mem` |

Replace N with 1-6. HLS C simulation writes the first image's Conv-ReLU vectors
into its working directory. Copy each pair into the corresponding `convN/`
directory, creating it if needed. Confirm the expected outputs match the RTL's
fixed-point arithmetic before using them for numerical validation.

The full CNN input file contains the parameter stream followed by ten normalized
CHW images. Its expected-output file contains ten class indices. HLS first-image
layer dumps do not supply these full-system fixtures.

Vivado simulation, synthesis, timing, and FPGA measurements have not been
validated for the current repository configuration.

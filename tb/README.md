# Hardware simulation

All paths below are relative to the repository root.

| Directory | Purpose |
| --- | --- |
| `tb/rtl/layers/` | Six individual fused Conv-ReLU layer tests |
| `tb/rtl/accelerator/` | Full CNN inference testbench |
| `tb/rtl/wrappers/` | AXI-stream interfaces and local memories used to simulate each layer |
| `tb/hls/` | HLS C simulation testbench |
| `tb/data/` | Shared parameter tensors and simulation vectors |

The wrappers instantiate the compute cores under `rtl/` and expose standalone
stream interfaces for the layer tests. Add them as **Simulation Sources**.
The production design uses the sources under `rtl/`.

## Vivado Simulator setup

1. Create a Vivado RTL project for the target FPGA.
2. Add all `.v` files under `rtl/` as Design Sources. Set `cnn_dataflow` as the
   design top.
3. Add `.v` files under `tb/rtl/layers/`, `tb/rtl/accelerator/`, and
   `tb/rtl/wrappers/` as Simulation Sources.
4. Add the input and expected-output `.mem` files for the selected test as
   Simulation Sources. In the testbench, set the `$readmemh` paths to the absolute
   locations of these files (use forward slashes on Windows). The checked-in
   paths are relative to the repository root; absolute paths avoid depending on
   the simulator's working directory.
5. In the Sources pane, right-click the testbench module from the table below
   and select **Set as Top** under Simulation Sources.
6. In **Settings > Simulation**, select **Vivado Simulator** as the target
   simulator. Choose **Run Simulation > Run Behavioral Simulation**, then use
   **Run All** in the simulation toolbar. Check the simulator output for the
   testbench's pass/fail result.

| Test | Simulation top |
| --- | --- |
| Convolution layer N, where N is 1-6 | `tb_myip_convN_relu_unrolled` |
| Full ten-image CNN | `tb_cnn_dataflow` |

When selecting a layer test, replace N in the top name with its layer number.
The HLS testbench is configured separately in [the HLS guide](../hls/README.md).

## Required data

| Location | Contents | Preparation |
| --- | --- | --- |
| `tb/data/` | 14 weight and bias tensor files for HLS C simulation | Included |
| `tb/data/system/` | Full ten-image CNN input and expected-output vectors | Included |
| `tb/data/conv_relu/convN/` | Input and expected-output vectors for Conv-ReLU layer N (1-6) | Generate using HLS C simulation |

### Full CNN test

Add these two files from `tb/data/system/` to the Vivado project for
`tb_cnn_dataflow`:

| File | Contents |
| --- | --- |
| `test_input_10.mem` | 288,298 parameter words followed by ten normalized CHW images, 3,072 words each |
| `test_result_expected_10.mem` | Ten expected class indices, one per image |

The testbench streams the parameters first, then the images. The input file
already includes all weights and biases; the individual tensor files in
`tb/data/` are only needed by the HLS C testbench. Configure the two `$readmemh`
paths as described in the Vivado Simulator setup above.

### Conv-ReLU layer tests

Run [HLS C Simulation](../hls/README.md#dump-inputs-and-outputs-for-rtl-testbenches)
to generate the first image's layer vectors in the simulation working directory.
For each layer, replace N with 1-6 and copy the following pair into
`tb/data/conv_relu/convN/`, creating the directory if needed:

| File | Contents |
| --- | --- |
| `test_input_convN_relu.mem` | Layer activations, weights, and biases |
| `test_result_expected_convN_relu.mem` | Expected output after convolution and ReLU |

Add the pair for the selected layer to the Vivado project and configure its
testbench's `$readmemh` paths. Confirm the HLS and RTL fixed-point arithmetic
match when using these vectors for numerical validation.

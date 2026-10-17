# PYNQ inference

Run on a supported FPGA board with NumPy, PYNQ, and an overlay that matches the
application's DMA and control interfaces. FPGA board block designs, constraints,
and deployable overlays are not included.

## Files and deployment paths

| Runner | Setting | Current value | What to configure |
| --- | --- | --- | --- |
| `pynq/infer_rtl.py` | `OVERLAY_PATH` | `/home/ubuntu/hdl/cnn_dataflow.xsa` | Your board's RTL overlay path |
| `pynq/infer_rtl.py` | `PARAM_BIN` | `/home/ubuntu/weights_q6_10.bin` | Board copy of `model/weights_q6_10.bin` |
| `pynq/infer_rtl.py` | `CIFAR_BIN` | `/home/ubuntu/test_batch.bin` | Board copy of `data/test_batch.bin` |
| `pynq/infer_rtl.py` | `NUM_IMAGES` | 1 | Requested image count, consistent with the packaged design |
| `pynq/infer_hls.py` | `Overlay(...)` | `/home/ubuntu/hls/cnn_accelerator.xsa` | Your board's HLS overlay path |
| `pynq/infer_hls.py` | `PARAM_BIN`, `CIFAR_BIN` | Repository-relative model/data paths | Keep the repository layout, or edit these paths |

Copy the repository to the board for HLS inference so `REPO_ROOT` can resolve
`model/weights_q6_10.bin` and `data/test_batch.bin`. The RTL application uses the
absolute deployment paths in its constants; copy those two inputs there or edit
the constants to point to your repository checkout.

## HLS inference

The expected block-design instance names are `axi_dma_0` and
`cnn_accelerator_0`. Edit the `dma` and `ip` assignments in `infer_hls.py` if your
instances have different names. The IP must expose the `num_images` register
and `CTRL.AP_START`, with 16-bit AXI-stream input words and output class indices.

After building and deploying the HLS overlay, run from the repository root:

```sh
python3 pynq/infer_hls.py 100
```

The image count must be 1-10,000. Each call sends all 288,298 CNN-V2 parameter
words followed by normalized CHW image words, then receives one class index per
image. DMA and output buffers must match the 16-bit stream interface.

## RTL interface configuration required

`infer_rtl.py` currently looks up `cnn_v2_accelerator_0`, writes `num_images`, and
starts `CTRL.AP_START`. The `rtl/cnn_dataflow.v` top has **no AXI-Lite control
interface**. Renaming the overlay to `cnn_dataflow.xsa` does not make these
interfaces compatible.

Before using this runner with the provided RTL, adapt its control and DMA setup
to the packaged board design: remove the unsupported AXI-Lite operations, use
the compile-time `NUM_IMAGES` from `rtl/cnn_dataflow.v`, and match the DMA stream
widths and output packing to the board's width converters. Parameters load once
after RTL reset; repeated calls must follow that weight-loading protocol.
If using an overlay with the runner's current control interface, match its IP
name and registers instead. This configuration has not been validated on a board.

Both runners normalize CIFAR-10 inputs and transfer signed Q6.10 parameter/image
words. Board execution has not been validated for the current repository setup.

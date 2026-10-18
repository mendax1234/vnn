# PYNQ inference

Run on a supported FPGA board with NumPy, PYNQ, and an overlay that matches the
application's DMA and control interfaces. FPGA board block designs, constraints,
and deployable overlays are not included.

## Files and deployment paths

| Runner | Setting | Current value | What to configure |
| --- | --- | --- | --- |
| `pynq/infer_rtl.py` | `OVERLAY_PATH` | `/home/ubuntu/hdl/cnn_dataflow.xsa` | Your board's RTL overlay path |
| `pynq/infer_rtl.py` | `PARAM_BIN` | `/home/ubuntu/weights_q6_10.bin` | Board copy of `model/weights_q6_10.bin` |
| `pynq/infer_rtl.py` | `CIFAR_BIN` | `/home/ubuntu/test_batch.bin` | Board copy of `model/test_batch.bin` |
| `pynq/infer_rtl.py` | `NUM_IMAGES` | 1 | Set to the image count supported by the deployed HDL overlay |
| `pynq/infer_hls.py` | `Overlay(...)` | `/home/ubuntu/hls/cnn_accelerator.xsa` | Your board's HLS overlay path |
| `pynq/infer_hls.py` | `PARAM_BIN` | `/home/ubuntu/weights_biases_int16.bin` | Board copy of `model/weights_q6_10.bin`, renamed to `weights_biases_int16.bin` |
| `pynq/infer_hls.py` | `CIFAR_BIN` | `/home/ubuntu/test_batch.bin` | Board copy of `model/test_batch.bin` |

Both runners use absolute deployment paths. Copy `model/test_batch.bin` to
`/home/ubuntu/test_batch.bin`. Copy `model/weights_q6_10.bin` to
`/home/ubuntu/weights_biases_int16.bin` for HLS inference or
`/home/ubuntu/weights_q6_10.bin` for RTL inference, or edit the constants to point
to your deployed input files.

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

## RTL inference

After building and deploying a compatible HDL overlay, configure `OVERLAY_PATH`,
`PARAM_BIN`, `CIFAR_BIN`, and `NUM_IMAGES` in `pynq/infer_rtl.py` using the table
above. Run from the repository root on the board:

```sh
python3 pynq/infer_rtl.py
```

The image count is set by `NUM_IMAGES` in the script. And it should match the settings in the Vivado block design for the RTL. Then the runner loads the parameters and CIFAR-10 images, transfers them through DMA, and prints the predictions, accuracy, and elapsed inference time.

Both runners normalize CIFAR-10 inputs and transfer signed Q6.10 parameter/image
words.

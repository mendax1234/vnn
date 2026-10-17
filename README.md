<h1 align="center">VNN: A High-Performance RTL Accelerator<br>for Edge CNN Inference on FPGAs</h1>

<p align="center">
  <strong>Zhu Wenbo, Tan Chou Rong Roel, Ng Joo An Sean, and Rajesh C. Panicker</strong><br>
  Department of Electrical and Computer Engineering<br>
  National University of Singapore
</p>

<!-- Replace ARXIV_ID with the paper identifier and uncomment this block after publication.
<p align="center">
  <a href="https://arxiv.org/abs/ARXIV_ID">
    <img src="https://img.shields.io/badge/arXiv-Paper-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv Paper">
  </a>
  <a href="https://arxiv.org/pdf/ARXIV_ID">
    <img src="https://img.shields.io/badge/arXiv-PDF-red?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv PDF">
  </a>
</p>
-->

<p align="center">
  <a href="assets/vnn-architecture.png">
    <img src="assets/vnn-architecture.png" alt="VNN architecture: system-level pipeline, packed weight organization, and convolutional layer microarchitecture" width="100%">
  </a>
</p>

---

VNN (Verilog Neural Network) is an RTL accelerator for multi-bit fixed-point CNN
inference on edge FPGAs. It coordinates channel parallelism, on-chip memory
organization, and register-level timing optimization within a layer-wise
pipelined architecture. The evaluated implementation is a 16-bit Q6.10 CIFAR-10
classifier on the AMD Kria KV260 Vision AI Starter Kit.

## Architecture

VNN combines four architectural techniques:

- **Layer-wise pipelining:** dedicated processing stages exchange feature maps
  through ping-pong buffers, allowing adjacent layers to process different images
  concurrently.
- **Channel parallelism:** input- and output-channel unrolling expose parallel
  arithmetic, with folding mapping channel groups onto the available lanes.
- **Packed on-chip memory:** word packing and channel folding organize weights
  into wider, deeper BRAM, URAM, and LUTRAM banks for parallel access and improved
  memory utilization.
- **Timing optimization:** localized pipeline-register insertion, control/data
  alignment, and register replication reduce critical-path and routing delays.

The accelerator targets networks whose parameters and intermediate activations
fit in on-chip memory. Parameters are loaded before an image batch; intermediate
feature maps remain on chip.

### Reference network

The custom CNN uses six 3 x 3 convolutions with output channels
32, 32, 64, 64, 128, and 128. Each convolution is followed by ReLU. Max pooling
after Conv2 and Conv4 reduces the spatial dimensions from 32 x 32 to 8 x 8,
followed by global average pooling and a ten-class fully connected layer.
Batch normalization is folded into the convolution parameters during export.

The network requires **38,634,752 MACs per image** and **288,298 parameter words**.
The RTL top module is `cnn_dataflow`; the HLS baseline top is `cnn_accelerator`.

## Results reported in the paper

The following results compare VNN with the same-network Vitis HLS baseline on
the Kria KV260, using 16-bit weights and activations and 10,000 images per run.
These are the paper's implementation results, rather than measurements generated
by the README examples.

| Metric | HLS baseline | VNN RTL |
| --- | ---: | ---: |
| Clock frequency (MHz) | 200 | 333 |
| CIFAR-10 top-1 accuracy (%) | 89.31 | 89.31 |
| Throughput (FPS) | 333.1 | **6,771.3** |
| Throughput (GOPS) | 25.74 | **523.2** |
| LUTs | 51,052 | 23,371 |
| BRAM36 equivalents | 120.5 | 141 |
| URAMs | 36 | 56 |
| DSPs | 357 | 781 |
| Estimated on-chip power (W) | 4.98 | 3.909 |
| Energy efficiency (GOPS/W) | 5.17 | 133.8 |

VNN provides **20.3x higher throughput** with **54.2% fewer LUTs** than this HLS
baseline, with increased DSP and on-chip memory utilization. Against the board's
single-threaded Q6.10 ARM Cortex-A53 implementation, the reported throughput
improvement is **1,546x**.

GOPS counts one MAC as two operations. FPGA timings include parameter loading,
input/output transfers, and inference, and exclude overlay initialization, file
loading, and preprocessing. Power values are **Vivado post-implementation on-chip
estimates**; they are not measurements of total board power.

## Repository structure

```text
rtl/             VNN RTL implementation
  interfaces/    AXI-stream image input and result output
  util/          Shared counter and RTL utilities
hls/             Same-network HLS baseline and Vitis project flow
tb/rtl/          RTL layer and system testbenches
tb/hls/          HLS C simulation testbench
tb/data/         Parameter tensors and simulation-vector locations
model/           Model definition, checkpoint, dataset, exporter, and CPU/CUDA baselines
pynq/            FPGA board inference applications
```

Standalone AXI-stream wrappers for the layer testbenches are in
`tb/rtl/wrappers/`. Generated executables and hardware project outputs belong
under ignored `build/` directories.

## Getting started

The FPGA implementations and PYNQ applications were tested on the AMD Kria KV260
Vision AI Starter Kit.

Run commands from the repository root unless a component guide specifies another
working directory. The included evaluation inputs are `model/test_batch.bin`,
`model/weights_q6_10.bin`, and fourteen parameter tensors under `tb/data/`.
The trained checkpoint is `model/checkpoints/cnn.pth`.

| Workflow | Requirements | Guide |
| --- | --- | --- |
| Model and parameter export | Python 3, NumPy, PyTorch | [Model guide](model/README.md) |
| Q6.10 CPU inference | C++17 compiler; Python 3 for benchmarking | [CPU guide](model/cpu/README.md) |
| FP16 and Q6.10 GPU inference | NVIDIA GPU, CUDA Toolkit, C++ toolchain; cuBLAS for FP16 | [CUDA guide](model/cuda/README.md) |
| RTL simulation | AMD Vivado/XSim with XPM memory support | [Testbench guide](tb/README.md) |
| HLS simulation and synthesis | AMD Vitis HLS | [HLS guide](hls/README.md) |
| Board execution | Kria KV260/PYNQ environment and matching overlay | [PYNQ guide](pynq/README.md) |

Build, inference, and benchmark commands are provided in the component guides.
Dataset formats and the checksum are in the [data guide](model/README.md#evaluation-data).

## Reproducing the hardware configuration

The paper evaluates RTL at **333 MHz** and HLS at **200 MHz**. In
`hls/run_hls.tcl`, configure the FPGA part, `create_clock` period, and simulation
image count for the target platform. The provided HLS flow defaults to the K26
part, a 5 ns clock, and one C simulation image. Configure RTL timing and board
integration in Vivado for the intended operating frequency.

Select the simulation top and working directory as described in the testbench
guide. RTL memory paths are relative to the repository root. Before board
execution, match overlay paths, IP instance names, control registers, DMA stream
widths, and image counts to the packaged design using the PYNQ guide.

### Required artifacts

For reproduction, follow the component guides to configure file paths, FPGA
settings, overlay/IP names, and board interfaces for your environment. The
checkpoint and inference model are included. Training code, board-specific Vivado
project files/constraints, and deployable overlays are not included in the current
source tree. Additional simulation vectors needed by the RTL testbenches are
listed in the testbench guide.

## Citation

```bibtex
@misc{zhu2026vnn,
  title  = {VNN: A High-Performance RTL Accelerator for Edge CNN Inference on FPGAs},
  author = {Zhu, Wenbo and Tan, Chou Rong Roel and Ng, Joo An Sean and Panicker, Rajesh C.},
  year   = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE).

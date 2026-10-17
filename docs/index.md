# VNN

Welcome to the VNN Docs website!

## What is VNN?

VNN, short for <strong>V</strong>erilog <strong>N</strong>eural <strong>N</strong>etwork, is an adaptable, high-performance RTL design framework for CNN inference. It was developed by Wenbo, Roel, and Sean as part of the NUS EE4218 final project, under the guidance of Dr. Rajesh.

### Introduction

VNN provides a systematic way to build CNN inference accelerators in RTL. It maps the nested-loop structure of convolutional computation onto a reusable hardware microarchitecture, giving designers fine-grained control over datapath organization, control flow, pipelining, and memory access.

The framework improves inference throughput through both layer-level and inter-layer pipelining, while supporting frequency scaling with minimal redesign effort. As a validation example, VNN was used to implement a CIFAR-10 image classification accelerator on the AMD Kria KV260 Vision AI Starter Kit, achieving up to **13,542 FPS** at **0.639 W** dynamic power, or **21,192 FPS/W**.

### Motivation

Convolutional neural networks are widely used in image recognition, but their inference workloads are dominated by repeated multiply-and-accumulate operations. This makes efficient hardware implementation critical, especially for edge and embedded platforms where performance, power, memory bandwidth, and hardware resources are tightly constrained.

Although model-level optimizations such as quantization, binarization, and lightweight CNN architectures can reduce computation, efficient deployment still depends on how the model is mapped onto hardware. Direct RTL design provides low-level control over datapaths, control logic, pipelining, and memory access, but handcrafted accelerators are often model-specific and difficult to adapt when the network structure, parallelism, or target frequency changes.

VNN is motivated by the need for a reusable RTL framework that preserves fine-grained hardware control while making CNN accelerator design more systematic and adaptable.

## More VNN Resources

- [VNN Repository](https://github.com/mendax1234/vnn)
- Publication Paper

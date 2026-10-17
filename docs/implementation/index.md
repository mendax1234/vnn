# FPGA Implementation

This section documents the design process behind the VNN framework, from early HLS exploration to the final RTL implementation.

We first discuss the HLS-based approach and its limitations, which motivated a lower-level HDL design. We then introduce the RTL implementation of VNN, covering both microscopic and macroscopic pipelining strategies. Finally, we describe the optimization techniques used to improve throughput in the RTL design, including loop unrolling and clock frequency scaling.

The goal of this section is to explain not only how the FPGA accelerator is implemented, but also why each design decision was made.

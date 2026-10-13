# CUDA Vector-Add Experiment

This is a standalone CUDA C++ learning/benchmark program. It does not accelerate or replace any part of the GritSense volleyball pipeline.

## Requirements

- NVIDIA GPU supported by the installed CUDA toolkit and a compatible NVIDIA driver.
- CUDA Toolkit including `nvcc`.
- A C++17 host compiler supported by that CUDA Toolkit.
- GNU Make (the supplied Makefile uses GNU Make syntax).

## Build and Run

From this directory:

```sh
make cpu
./cpu_baseline 1048576
make cuda
./vector_add 1048576
```

The CUDA executable reports CPU host elapsed time, CUDA event time for the kernel only, and host end-to-end elapsed time including allocations, H2D/D2H copies, synchronization, and cleanup. It compares every CUDA output element against the CPU reference and returns a non-zero exit code on failure. The vector length is optional and defaults to 1,048,576.

On Windows, use a GNU Make environment such as MSYS2/MinGW with a host compiler supported by the installed CUDA Toolkit, or adapt the two compile commands for the Visual Studio developer shell. `nvcc` and its compiler/toolkit compatibility are NVIDIA requirements; no CUDA result can be produced without them.

## Measurement Notes

This example is intentionally a simple bandwidth-sensitive operation. One timing run is instructional, not a statistically robust performance claim. Kernel event time excludes host-device transfers; end-to-end time includes transfers, allocation, and cleanup. Do not infer a GritSense speedup from these numbers.
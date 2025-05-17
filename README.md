# CPU/GPU Memory & Near-Data Processing Homework

This repository contains assignments from a hardware systems class focusing on CPU/GPU memory architecture, cache design, DRAM simulation, GPU programming, and near-data processing (PIM). Each assignment applies industry-standard tools to analyze, simulate, and optimize real-world memory and processing behavior.

## 📁 Assignments Overview

### HW1: Roofline Model Analysis
We analyzed memory and compute bottlenecks using the Roofline model via Intel Advisor. The assignment profiled 10 matrix/vector implementations to observe performance trends. Key deliverables included:
- Roofline plots comparing different kernels
- INTOPS/sec and arithmetic intensity tables
- System configuration summary and ridge point classification

**Tools:** Intel Advisor, C++, Roofline plots

---

### HW2: Cache Design with CACTI
This assignment explored the trade-offs between cache size, associativity, port count, and technology node using CACTI. We swept various parameters and reported:
- Access time, area, energy, and data efficiency trends
- Optimal configurations for L1 and LLC caches

**Tools:** CACTI, Linux shell

---

### HW3: DRAM Simulation with DRAMsim3
We simulated memory access patterns (random, streaming, mixed) across different DRAM types: DDR4, LPDDR4, GDDR6, and HBM2. The analysis included:
- Bandwidth, energy, and latency comparison
- Command-level activity (`ACT`, `PRE`) trends
- DRAM selection for power vs performance needs

**Tools:** DRAMsim3, Python, JSON-to-CSV conversion

---

### HW4: GPU Programming with CUDA
This assignment focused on completing CUDA kernels for matrix addition, multiplication, and reduction. We:
- Verified CPU vs GPU output correctness
- Benchmarked GPU kernels with and without shared memory
- Profiled kernel execution using `nvprof` and CUDA events

**Tools:** CUDA Toolkit, `nvcc`, `nvprof`, OpenMP

---

### HW5: PIM Programming with PIMeval-PIMbench
We explored near-data processing using UVA’s PIMeval-PIMbench simulator. The assignment included:
- Running GEMV across various HBM PIM configs (1–32 banks)
- Implementing new benchmarks: RMS Norm and Layer Norm
- Comparing total time and energy with CPU baselines

**Tools:** PIMeval-PIMbench, C++, OpenMP, HBM DRAM configs

---

## 🧰 Technologies Used
- Intel Advisor (Roofline Modeling)
- CACTI (Cache Simulation)
- DRAMsim3 (DRAM Trace Simulation)
- NVIDIA CUDA Toolkit (GPU Kernel Programming)
- PIMeval-PIMbench (Near-Memory Processing Simulator)
- Python (Trace generation, data conversion, plotting)

## 📌 Notes
Each assignment folder contains:
- Source code or benchmark implementations
- Configuration files and scripts
- Analysis results (plots, CSVs, screenshots)
- A short README with assignment-specific goals

---

## 🔍 License
This repository is for academic and research use only. Please do not copy without permission.

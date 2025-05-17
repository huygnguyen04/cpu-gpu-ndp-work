# CPU/GPU Memory & Near-Data Processing Assignments

This repository contains my assignments from a hardware systems class focusing on CPU/GPU memory architecture, cache design, DRAM simulation, GPU programming, and near-data processing (PIM). Each assignment applies industry-standard tools to analyze, simulate, and optimize real-world memory and processing behavior.

## 📁 Assignments Overview

### HW1: Roofline Model Analysis
- **Assignment PDF:** [HW1 Assignment](./roofline-model-analysis/roofline_assignment.pdf)
- **Report:** [HW1 Report](./roofline-model-analysis/HuyNguyen_Roofline_Report.pdf)
- Analysis of memory and compute bottlenecks across multiple matrix/vector kernels using Intel Advisor's Roofline model. The assignment involved:
  - Profiling 10 distinct matrix/vector implementations with varying optimization levels
  - Generating Roofline plots to visualize performance bottlenecks
  - Measuring INTOPS/sec and arithmetic intensity across different implementations
  - Identifying the ridge point where code transitions from memory-bound to compute-bound

**Tools:** Intel Advisor, C++, Roofline visualization

---

### HW2: Cache Design with CACTI
- **Assignment PDF:** [HW2 Assignment](./cache-design-cacti/CACTI_assignment.pdf)
- **Report:** [HW2 Report](./cache-design-cacti/HuyNguyen_CACTI_Report.pdf)
- Systematic exploration of cache design tradeoffs using CACTI cache simulator. Key aspects:
  - Parameter sweeps across cache sizes (16KB to 8MB), associativity (1-way to 16-way)
  - Analysis of access time, area, energy consumption, and data efficiency
  - Examination of technology node impact (65nm vs. 32nm) on cache performance
  - Determination of optimal configurations for both L1 and LLC caches

**Tools:** CACTI 7.0, Bash scripting, data visualization

---

### HW3: DRAM Simulation with DRAMsim3
- **Assignment PDF:** [HW3 Assignment](./DRAM-simulation/DRAM_assignment.pdf)
- **Report:** [HW3 Report](./DRAM-simulation/HuyNguyen_DRAM_Report.pdf)
- Comprehensive simulation of various DRAM technologies under different memory access patterns:
  - Comparison of DDR4, LPDDR4, GDDR6, and HBM2 under random, streaming, and mixed patterns
  - Analysis of bandwidth scaling, energy consumption, and latency characteristics
  - Detailed examination of command-level activity distribution (ACT, PRE, RD/WR)
  - DRAM selection recommendations for power-constrained vs. performance-driven scenarios

**Tools:** DRAMsim3, Python for data processing, JSON-to-CSV conversion

---

### HW4: GPU Programming with CUDA
- **Assignment PDF:** [HW4 Assignment](./cuda-programming/cuda_assignment.pdf)
- **Report:** [HW4 Report](./cuda-programming/HuyNguyen_CUDA_Report.pdf)
- Implementation and optimization of parallel algorithms using NVIDIA CUDA:
  - Development of matrix addition, matrix multiplication, and parallel reduction kernels
  - Implementation of shared memory optimizations and thread cooperative strategies
  - Performance evaluation using CUDA events and nvprof profiling
  - Comparative analysis between optimized GPU implementations and CPU baselines

**Tools:** NVIDIA CUDA Toolkit, nvcc compiler, nvprof, CUDA events timing

---

### HW5: PIM Programming with PIMeval-PIMbench
- **Assignment PDF:** [HW5 Assignment](./pim-programming/assignment_PIMeval.pdf)
- **Report:** [HW5 Report](./pim-programming/HuyNguyen_PIM_Report.pdf)
- Exploration of near-data processing using UVA's PIMeval-PIMbench simulator:
  - Implementation of RMS Norm and Layer Norm algorithms for the PIM architecture
  - Performance analysis across varying HBM configurations (1-32 computing banks)
  - Energy efficiency analysis of PIM vs. traditional CPU implementations
  - Evaluation of parallelism scalability and resource utilization in PIM context

**Tools:** PIMeval-PIMbench, C++ for kernel implementation, OpenMP, HBM modeling

---

## 🧰 Technical Environment
- **Intel Advisor:** Roofline modeling and performance characterization
- **CACTI 7.0:** Cache architecture simulation and power/area analysis
- **DRAMsim3:** DRAM timing and energy simulation
- **NVIDIA CUDA Toolkit:** GPU kernel development and profiling
- **PIMeval-PIMbench:** Near-memory processing simulation framework
- **Supporting tools:** Python for data analysis, visualization libraries, shell scripting

## 📌 Repository Structure
Each assignment folder contains:
- Source code and implementations
- Configuration files and execution scripts
- Results and analysis visualizations
- Detailed technical reports

---

## 🔍 License
This repository is licensed under the [MIT License](LICENSE).

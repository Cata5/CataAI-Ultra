# Hardware Compatibility Matrix

**Platform:** LLM Maker v2.0  
**Evaluator:** Physical Runtime Diagnostics & PyTorch Backend Probes  

---

## 1. Supported Hardware Environments

| Execution Environment | Detection Mechanism | Telemetry Support | Precision Modes Supported | Profile Presets Supported | Compatibility Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA CUDA (Desktop / Laptop)** | `pynvml` (NVML) + `torch.cuda` | Full real-time metrics (VRAM, Util %, Temp, Power, Clocks) | FP32, FP16, BF16 (Compute Capability >= 8.0) | Conservative (65%), Balanced (80%), Performance (90%), Custom | **FULLY SUPPORTED & VERIFIED** |
| **Host CPU Only** | `platform`, `ctypes` (`GlobalMemoryStatusEx` on Windows, `/proc/meminfo` on Linux) | System RAM (Total / Available), CPU Utilization | FP32, BF16 (where supported by CPU architecture) | Conservative (50%), Balanced (70%), Performance (85%), Custom | **FULLY SUPPORTED & VERIFIED** |
| **Apple Silicon (MPS)** | `torch.backends.mps.is_available()` | Unified System Memory | FP32, FP16 | Balanced (75%), Performance (85%) | **SUPPORTED (Experimental)** |
| **AMD ROCm** | `torch.version.hip` | PyTorch allocator telemetry | FP32, FP16, BF16 (where ROCm driver active) | Conservative, Balanced, Performance | **SUPPORTED (Experimental)** |

---

## 2. Dynamic Memory Ceiling Calculations

Hardware memory limits are determined dynamically by `HardwareProfile.create()`:

$$\text{Allocation Ceiling} = \text{Total VRAM} \times \text{Usable Ratio}$$

$$\text{Warning Threshold} = \max(\text{Total VRAM} \times 0.10, 300\text{ MB})$$

$$\text{Critical Threshold} = \max(\text{Total VRAM} \times 0.06, 150\text{ MB})$$

$$\text{Emergency Threshold} = \max(\text{Total VRAM} \times 0.03, 80\text{ MB})$$

### Pre-calculated Empirical Profiles for Common VRAM Tiers

| Hardware Target | Total Memory | Conservative (65%) | Balanced (80%) | Performance (90%) | Recommended Micro-Batch |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Consumer 4GB (RTX 3050)** | 4,096 MB | 2,662 MB | 3,276 MB | 3,686 MB | 1 (Grad Checkpoint: ON) |
| **Consumer 6GB (RTX 3060 / 4050)** | 6,144 MB | 3,993 MB | 4,915 MB | 5,529 MB | 2–4 (Grad Checkpoint: OFF) |
| **Consumer 8GB (RTX 3070 / 4060)** | 8,192 MB | 5,324 MB | 6,553 MB | 7,372 MB | 4–8 |
| **Workstation 12GB (RTX 3060 12G / 4070)** | 12,288 MB | 7,987 MB | 9,830 MB | 11,059 MB | 8–16 |
| **Enthusiast 16GB (RTX 4080)** | 16,384 MB | 10,649 MB | 13,107 MB | 14,745 MB | 16 |
| **Flagship 24GB (RTX 3090 / 4090)** | 24,576 MB | 15,974 MB | 19,660 MB | 22,118 MB | 16–32 |
| **Host System CPU RAM (32GB–64GB)** | 32,768 MB | 21,299 MB | 26,214 MB | 29,491 MB | 4–8 (FP32) |

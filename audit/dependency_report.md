# Dependency Report

## Environment & Run-Time Stack
- **OS**: Windows 11
- **Host GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB GDDR6 VRAM)
- **CUDA Runtime**: 12.8
- **Python**: 3.14.6
- **Node.js**: v22.23.2
- **Package Manager**: npm 10.9.2

---

### Python Dependencies (`.venv`)

| Package | Version | Purpose | Risk Assessment |
| :--- | :--- | :--- | :--- |
| `torch` | 2.11.0+cu128 | Deep learning compute engine, CUDA kernels | Low (Stable cu128 build) |
| `torchvision` | 0.26.0+cu128 | Vision transform support (CNN templates) | Low |
| `aiohttp` | 3.14.5 | Async HTTP REST server and WebSockets | Low (Configured client_max_size=500MB) |
| `nvidia-ml-py` | 13.615.71 | Direct NVIDIA NVML GPU hardware telemetry | Low (Replaces deprecated pynvml) |
| `tokenizers` | 0.23.3 | Fast BPE tokenizer engine | Low |
| `safetensors` | 0.8.0 | Zero-copy safe model weights format | Low |
| `numpy` | 2.5.4 | Numerical matrix and memmap backing | Low |
| `pytest` | 9.2.0 | Test automation framework | Low |

---

### Frontend Dependencies (`frontend/package.json`)

| Package | Version | Purpose | Risk Assessment |
| :--- | :--- | :--- | :--- |
| `next` | 16.4.6 | React framework with App Router & Turbopack | Low |
| `react` / `react-dom` | 19.3.0 | Modern UI rendering library | Low |
| `@xyflow/react` | 12.10.1 | Node-based visual network architecture editor | Low (Wrapped with dynamic ssr:false) |
| `chart.js` / `react-chartjs-2` | 4.5.1 / 5.3.1 | Telemetry and learning rate schedule charts | Low |
| `@tanstack/react-query` | 5.75.1 | Server-state caching and synchronization | Low |
| `lucide-react` | 1.16.0 | Sci-Fi technical UI iconography | Low |
| `tailwindcss` | 4.1.18 | Utility CSS styling system with PostCSS | Low |
| `typescript` | 5.8.2 | Strict compile-time static type checker | Low |

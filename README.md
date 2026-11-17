# LLM Maker 🧠⚡
> **High-Performance Local Neural Architecture Studio & LLM Training Platform**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg?style=flat&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Next.js](https://img.shields.io/badge/Next.js-14-black.svg?style=flat&logo=next.js)](https://nextjs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-272%20Passed-brightgreen.svg)]()

**LLM Maker** is an end-to-end, local-first research platform designed to democratize pretraining, fine-tuning, and architectural experimentation for Large Language Models. Built for consumer hardware (from 6GB VRAM laptop GPUs up to enterprise clusters), it combines a **visual node-based neural architecture builder**, **WDDM-aware zero-paging VRAM protection**, **hybrid Muon + AdamW optimizers**, and a **multi-phase training engine** (Pretrain &rarr; SFT &rarr; Chat).

---

## 🚀 Key Features

### 1. Visual Neural Architecture Builder
- **Drag-and-Drop Canvas**: Visually compose custom transformer and hybrid architectures with strict tensor dimension contracts.
- **Modular Blocks**: MHA, GQA, MQA, RoPE (Rotary Position Embeddings), RMSNorm, LayerNorm, SwiGLU, GeLU, and MoE routing.
- **Instant Parameter & Memory Estimator**: Real-time analytical VRAM breakdown (weights, optimizer states, activations, KV-cache) before spending a single GPU second.
- **Hot-swappable Model Registry**: Export architectures to standardized JSON configs (`networks/registry/`) and load them dynamically in training or inference.

### 2. Zero-Paging VRAM Engine
- **Active Memory Ceiling**: Constrains peak allocation beneath the Windows WDDM paging cliff (e.g. strict < 5.0 GB ceiling on 6GB GPUs) to prevent memory spills into system RAM that collapse throughput by 80–90%.
- **Automatic Micro-Batching & Gradient Accumulation**: Solves the exact batch size and gradient accumulation steps needed to maximize token throughput without out-of-memory (OOM) faults.
- **Hardware Telemetry**: Real-time NVML monitoring of dedicated VRAM, GPU temperature, compute utilization, and PCIe throughput.

### 3. Multi-Phase Training Pipeline
- **Phase 1: Pre-training**: Autoregressive causal language modeling with dynamic context packing and loss masking.
- **Phase 2: Instruction Tuning (SFT)**: Supervised fine-tuning with prompt loss masking (`label: -100`) so the model only calculates gradients on responses.
- **Phase 3: Multi-turn Chat**: ChatML tokenization and structured dialog optimization.
- **Seamless Resume**: Atomic checkpointing with multi-phase epoch recalculation, optimizer momentum preservation, and automatic cleanup.

### 4. Advanced Optimizers
- **Hybrid Muon Optimizer**: Implements Moonshot-style Muon using quintic Newton-Schulz orthogonalization on 2D weight matrices, delivering faster loss convergence per token compared to standard AdamW.
- **AdamW Integration**: Automatically splits 1D parameters (RMSNorm, bias, embedding tables) to AdamW while routing attention and feedforward matrices to Muon.

### 5. Decoupled Web Workstation
- **Backend**: Async Python control plane (`aiohttp`, PyTorch multiprocessing, SQLite WAL journal).
- **Frontend**: Next.js 14 App Router, Tailwind CSS, Lucide icons, and real-time Server-Sent Events (SSE) streaming metrics and loss curves.
- **Interactive Evaluation Lab**: Built-in inference playground with adjustable temperature, top-k, top-p, repetition penalty, and real-time token throughput benchmarking.

---

## 💻 Hardware Requirements

| Hardware Tier | Recommended Model Size | Context Length | Batch Configuration |
| :--- | :--- | :--- | :--- |
| **Consumer (6 GB VRAM)**<br>*(e.g. RTX 3060 / 4060 Laptop)* | 15M – 48.5M parameters | 512 – 1024 tokens | Micro-batch: 8–16<br>Accumulation: 2–4 |
| **Mid-range (8 – 12 GB VRAM)**<br>*(e.g. RTX 3080 / 4070)* | 50M – 120M parameters | 1024 – 2048 tokens | Micro-batch: 16–32<br>Accumulation: 4 |
| **High-end (16 – 24 GB VRAM)**<br>*(e.g. RTX 3090 / 4090)* | 150M – 500M parameters | 2048 – 4096 tokens | Micro-batch: 32–64<br>Accumulation: 4 |
| **Datacenter (40 – 80 GB VRAM)**<br>*(e.g. A100 / H100)* | 1B – 7B parameters | 4096 – 8192 tokens | High-throughput distributed |

---

## ⚡ Quickstart

### Option A: One-Click Setup (Recommended)

#### Windows
```cmd
# Run setup script (creates .venv, installs PyTorch + CUDA, installs npm packages)
setup.bat

# Launch LLM Maker Workstation
run_studio_v2.bat
```

#### Linux / macOS / WSL2
```bash
# Make scripts executable and run setup
chmod +x setup.sh run_studio.sh
./setup.sh

# Launch LLM Maker Workstation
./run_studio.sh
```

---

### Option B: Manual Installation

#### 1. Clone the repository
```bash
git clone https://github.com/your-username/llm-maker.git
cd llm-maker
```

#### 2. Create Python virtual environment
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

#### 3. Install PyTorch with CUDA
Visit [pytorch.org](https://pytorch.org/get-started/locally/) to select your CUDA version:
```bash
# Example for CUDA 12.8 / 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install remaining backend dependencies
pip install -r requirements.txt
```

#### 4. Install Frontend dependencies
```bash
cd frontend
npm install
npm run build
cd ..
```

#### 5. Run the Platform
In two separate terminals (or using the launch scripts):
```bash
# Terminal 1: Python ML Backend (Port 8081)
python dashboard.py --port 8081

# Terminal 2: Next.js Studio UI (Port 8080)
cd frontend
npm run start -- -p 8080
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser.

---

## 📦 Bundled Sample Datasets

The repository includes curated mini-datasets ready for immediate testing:

| Dataset | Size | Tokens | Format | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `tech_pretrain_10M.txt` | ~25 MB | 10,000,419 | Plain Text / JSONL | Fast pre-training verification |
| `tech_chat_instruction_1M.txt`| ~3.2 MB | 1,000,432 | Instruction / Dialog | Supervised fine-tuning (SFT) |
| `tech_instruction_1M.txt` | ~3.2 MB | 1,000,000 | Alpaca format | Instruction alignment |
| `tech_chat_1M.txt` | ~3.3 MB | 1,000,000 | Multi-turn Chat | Conversational fine-tuning |

To train a model:
1. Open the **Datasets** tab in the UI.
2. Click **Import Dataset** and select `tech_pretrain_10M.txt` (Phase 1) or `tech_chat_instruction_1M.txt` (Phase 2).
3. Switch to **Training**, choose your network architecture, and start training with live loss curves!

---

## 🧪 Running the Test Suite

LLM Maker includes 270+ automated tests verifying neural layer arithmetic, tokenizer parity, dataset pipelines, VRAM estimators, checkpoint resumption, and API endpoints.

```bash
# Run the complete test suite
pytest

# Run specific unit tests
pytest tests/unit/test_architecture.py
pytest tests/unit/test_optimizer.py
pytest tests/e2e/test_product_generalization.py
```

---

## 🏛 Architecture & Project Layout

```
llm-maker/
├── dashboard.py               # Main control plane & server entry point
├── requirements.txt           # Python backend dependencies
├── pytest.ini                 # Pytest configuration
├── run_studio_v2.bat          # Windows launcher (Backend + Frontend)
├── run_studio.sh              # Linux / macOS / WSL launcher
├── setup.bat / setup.sh       # Automated installation assistants
│
├── src/                       # Core Python Backend
│   ├── checkpoints/           # Atomic checkpointing & resume engine
│   ├── dashboard/             # aiohttp async REST & SSE control plane
│   ├── datasets/              # Tokenization, streaming loaders & cache
│   ├── db/                    # SQLite WAL persistence for metrics & runs
│   ├── eval/                  # Real-time inference engine & benchmark lab
│   ├── model/                 # Baseline RoCanon architectures & layers
│   ├── networks/              # Visual Builder factory & MemoryEstimator
│   ├── optimizers/            # Hybrid Muon + AdamW implementations
│   ├── telemetry/             # NVML hardware monitoring
│   ├── training/              # Task adapters (Pretrain, SFT, Chat)
│   └── worker/                # Decoupled multiprocessing training loop
│
├── frontend/                  # Next.js 14 Web Studio
│   ├── src/
│   │   ├── components/        # Sci-Fi UI components & visual canvas
│   │   └── features/          # Builder, Training, Datasets, Runs, Inference
│   ├── package.json           # Node.js dependencies
│   └── tailwind.config.js     # Dark-mode sci-fi theme
│
├── networks/registry/         # Exported model architecture configs (JSON)
└── tests/                     # 270+ unit and end-to-end tests
```

---

## 🤝 Roadmap & Collaboration

We are actively exploring integrations and partnerships with:
- **Unsloth**: Integrating Unsloth's fast custom CUDA/Triton kernels and QLoRA engine directly into LLM Maker's visual modular network factory.
- **Hugging Face**: One-click Hub model export (`AutoModelForCausalLM` / `safetensors`) and dataset streaming.
- **Distributed Training**: FSDP and DDP support for multi-GPU scaling.
- **GGUF / llama.cpp**: Direct visual quantization to GGUF (Q4_K_M, Q8_0) for edge deployment.

---

## 📄 License
This project is open-source under the Apache License 2.0.

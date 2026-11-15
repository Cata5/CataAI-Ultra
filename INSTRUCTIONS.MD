# RoCanon-48.5M Training WebUI v2 — Complete User Manual & Settings Guide

Welcome to **RoCanon-48.5M Training WebUI v2**, a modular, GPU-safe local ML training and experimentation platform optimized specifically for the **NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM)**.

---

## Table of Contents
1. [Quick Start: How to Launch](#1-quick-start-how-to-launch)
2. [Cockpit Architecture & Safety Guarantees](#2-cockpit-architecture--safety-guarantees)
3. [Complete Guide to Every Tab & Setting](#3-complete-guide-to-every-tab--setting)
   - [Tab 1: 📊 Overview](#tab-1--overview)
   - [Tab 2: 📈 Training & Loss](#tab-2--training--loss)
   - [Tab 3: 📚 Datasets](#tab-3--datasets)
   - [Tab 4: 🧪 Evaluation Lab](#tab-4--evaluation-lab)
   - [Tab 5: 💾 Runs & Checkpoints](#tab-5--runs--checkpoints)
   - [Tab 6: ⚙️ Settings & Telemetry](#tab-6--settings--telemetry)
   - [Tab 7: 🧩 Network Studio (Architecture Builder)](#tab-7--network-studio-architecture-builder)
4. [Recommended Step-by-Step Training Workflow](#4-recommended-step-by-step-training-workflow)
5. [Network Studio: Custom Neural Network Guide](#5-network-studio-custom-neural-network-guide)
   - [Creating a New Architecture](#creating-a-new-architecture)
   - [Parameter Budget Assistant & RTX 3060 VRAM Estimator](#parameter-budget-assistant--rtx-3060-vram-estimator)
   - [Preflight Safety Check](#preflight-safety-check)
   - [Developing Custom PyTorch Modules](#developing-custom-pytorch-modules)
6. [Storage Layout & File Locations](#6-storage-layout--file-locations)
7. [Troubleshooting & FAQ](#7-troubleshooting--faq)

---

## 1. Quick Start: How to Launch

Open PowerShell in the project directory (`d:\proiecte\CataAI Ultra`) and run:

```powershell
# 1. Activate your virtual environment (if not already active)
.venv\Scripts\Activate.ps1

# 2. Start the WebUI Studio
python dashboard.py
```

- Open your browser at: **`http://localhost:8080`**
- To change the port or host:
  ```powershell
  python dashboard.py --port 8888 --host 127.0.0.1
  ```
- To start training immediately when the studio launches:
  ```powershell
  python dashboard.py --train
  ```

---

## 2. Cockpit Architecture & Safety Guarantees

- **Decoupled Training Process**: The training worker runs in its own isolated Python process (`multiprocessing` with the `spawn` context). Refreshing your browser or restarting the web server will not crash the training loop.
- **Strict 6GB VRAM Protection**:
  - The model and optimizer are budget-capped at $\le 5.0\text{ GB}$.
  - Real-time hardware monitoring via NVIDIA NVML.
  - Automatic emergency checkpointing if available VRAM drops below $400\text{ MB}$.
  - Checkpoint comparisons in the Evaluation Lab run **sequentially on a single model instance** — two models are never loaded simultaneously into GPU memory.
- **Zero Mock Data**: All displayed charts, throughput rates, temperatures, and generation samples reflect genuine hardware and model measurements.

---

## 3. Complete Guide to Every Tab & Setting

The WebUI has **6 primary navigation tabs**:

```
[ 📊 Overview ]  [ 📈 Training & Loss ]  [ 📚 Datasets ]  [ 🧪 Evaluation Lab ]  [ 💾 Runs & Checkpoints ]  [ ⚙️ Settings ]
```

---

### Tab 1: 📊 Overview

The primary mission control center.

#### Top Control Bar
- **State Pill**: Displays live status (`IDLE`, `STARTING`, `RUNNING`, `PAUSED`, `STOPPING`, `STOPPED`, `COMPLETED`, `FAILED`).
- **GPU Badge**: Displays compute hardware (`GPU: RTX 3060 (6GB)`).
- **VRAM Indicator**:
  - 🟢 **Normal**: $> 1000\text{ MB}$ free VRAM.
  - 🟡 **Warning**: $700\text{ MB} - 1000\text{ MB}$ free VRAM.
  - 🟠 **Critical**: $400\text{ MB} - 700\text{ MB}$ free VRAM.
  - 🔴 **Emergency**: $< 400\text{ MB}$ free VRAM (protective snapshot requested).
- **Control Buttons**:
  - **Start (▶)**: Initiates training session with your selected configuration.
  - **Pause (⏸)**: Suspends training between steps cleanly.
  - **Validate (🔍)**: Triggers an immediate validation loss pass on the held-out validation split.
  - **Snapshot (💾)**: Takes an atomic checkpoint at the next step boundary without stopping.
  - **Stop (🛑)**: Safely stops training, restores best validation weights, and saves a checkpoint.

#### Dynamic Learning Rate Multiplier Slider
- **Range**: `0.10x` to `3.00x` (default: `1.00x`).
- **Function**: Scales Muon and AdamW learning rates in real time while training without restarting the optimizers.
  - Decrease to `0.5x` if gradient norm spikes or loss oscillates.
  - Increase to `1.2x - 1.5x` if loss plateau is flat and gradient norm is low ($\le 0.4$).

#### Real-Time KPI Cards
1. **Train Loss**: Chunked Cross-Entropy loss computed on the active micro-batch.
2. **Validation Loss & Gap**: Held-out validation loss and Generalization Gap ($\text{Loss}_{\text{val}} - \text{Loss}_{\text{train}}$).
3. **Gradient Norm**: Global gradient norm across all model parameters (clipped at 1.0 threshold).
4. **Throughput**: Real-time processing speed in tokens/second (`tok/s`) and compute in `TFLOPS`.
5. **GPU Temp & Util**: Real-time core temperature in °C, GPU load %, and power draw in Watts via NVML.
6. **Free VRAM**: Genuine free VRAM remaining on the RTX 3060 and PyTorch allocated memory (`PT: X.XX GB`).

#### Session Configuration Card
- **Phase**:
  - `1. General Pretraining`: Causal language modeling on Romanian & English educational/technical datasets.
  - `2. Domain Adaptation`: Continued pretraining on focused technical shards.
  - `3. Instruction Tuning (SFT)`: Prompt-response fine-tuning with assistant loss masking.
  - `4. Creator Persona`: Style and identity refinement for the assistant.
- **Dataset Source**: Select from default presets (`tech_wiki_ro`, `alpaca_ro`, `chat_persona_ro`) or any uploaded dataset.
- **Resume Checkpoint**: Select `Start from Scratch` or choose an existing checkpoint (`best_model.pt`, `snapshot_step_X.pt`).
- **Target Steps & Presets**: Training duration is strictly step-driven (token budget cap has been removed). Choose from quick presets (`100`, `500`, `1k`, `5k`, `10k`) or enter any custom integer.
- **Validate Every (Steps)**: Frequency of validation passes on the held-out split (default: `10` steps).
- **Save Checkpoint Every (Steps)**: Frequency of periodic training snapshots (default: `50` steps).
- **Early Stopping**:
  - **Enable / Disable Toggle**: Easily enable or disable early stopping with a single click.
  - **Patience**: Number of validation evaluations without loss improvement before training automatically terminates (default: `5`).
  - **Min Delta**: Minimum decrease in validation loss to qualify as an improvement (default: `0.001`).
  - **Auto-Restore**: When early stopping triggers, the worker automatically restores the best weights, saves `early_stopped_best.pt`, and halts training safely.

---

### Tab 2: 📈 Training & Loss

Advanced metric visualizers and scheduler analysis.

#### Interactive Loss Chart
- **Train Loss (Raw)**: Step-by-step un-smoothed loss curve.
- **EMA Smoothed**: Bold cyan curve smoothed via Exponential Moving Average.
- **Validation Loss**: Green dot markers placed at evaluation steps.
- **EMA Smoothing Slider**: Range `0.01` to `0.30` (default: `0.08`). Adjusts responsiveness of the smoothed curve.
- **Scale Button (Linear / Logarithmic)**: Switch Y-axis between linear and logarithmic scale to inspect fine-grained convergence after initial loss drops.
- **Reset Zoom**: Restores standard view after zooming into specific step intervals.

#### Auxiliary Telemetry Charts
1. **Gradient Norm**: Monitors training stability. Normal values: $0.2 - 1.0$.
2. **Learning Rate**: Tracks Muon LR (cyan) and AdamW LR (amber).
3. **Throughput**: Tracks training speed in tokens/second over time.
4. **PyTorch Allocated VRAM**: Verifies tensor memory allocation stays within the 5.0 GB ceiling.

#### Learning Rate Scheduler Visualizer
- **Schedule Type**:
  - `Warmup-Stable-Decay (WSD)`: Ramps up linearly, stays flat, then decays (cosine or linear).
  - `Cosine Decay`: Standard cosine decay following warmup.
  - `Linear Decay`: Linear drop from peak LR to minimum LR.
  - `Constant`: Flat learning rate following warmup.
- **Parameters**:
  - `Peak LR`: Target peak learning rate (default: `0.003` for Muon, `0.0006` for AdamW).
  - `Warmup Steps`: Linear ramp step count (default: `200`).
  - `Stable Steps`: Stable plateau duration (default: `600`).
  - `Total Steps`: Step budget for the phase (default: `1000`).
- **Recalculate Curve**: Renders the exact mathematical curve that PyTorch executes.

---

### Tab 3: 📚 Datasets

Upload, validate, and inspect training data.

#### Drag & Drop Ingestion
- Supported formats: `.txt` (raw text), `.jsonl` (JSON lines), `.json` (JSON arrays).
- **Validation checks on upload**:
  - UTF-8 byte validation.
  - Empty line/record elimination.
  - Schema check for instruction datasets (`instruction`, `input`, `output` fields).
  - Exact duplicate detection using SHA-256 hashes.
- Each dataset is assigned an immutable SHA-256 revision hash for reproducibility.

#### Tokenizer Inspector
Test how Romanian and English text is split into tokens:
1. Type or paste text into the input box.
2. Click **Tokenize Input**.
3. **Metrics shown**:
   - **Total Tokens**: Exact number of BPE tokens.
   - **Tokens per Word**: Fertility ratio (typical range: $1.3 - 1.9$).
   - **Single-Digit Atomization**: Inspect numbers and math operators to verify they tokenize as separate tokens without fusion.
   - **Diacritic Unification**: Verifies `ș` and `ț` (comma below) decode identically.
   - **Token Sequence Breakdown**: Color-coded badges showing each token string and token ID.

---

### Tab 4: 🧪 Evaluation Lab

Inference, side-by-side model comparison, and benchmark suites.

#### Live Model Inference
- **Prompt**: Text input for the model.
- **Max Tokens**: Token generation ceiling (e.g. `64`, `128`, `256`).
- **Temperature**:
  - `0.2 - 0.4`: Factual, deterministic output.
  - `0.7`: Balanced generation.
  - `1.0+`: High variance, creative generation.
- **Top-P (Nucleus Sampling)**: Cumulative probability cutoff (default: `0.90`).
- **Top-K**: Top $K$ candidate tokens (default: `50`).
- **Repetition Penalty**: Penalizes repeated tokens to prevent degenerate loops (default: `1.10`).
- Click **Generate** to run inference and view latency and speed (`tok/s`).

#### Side-by-Side Checkpoint Comparison
- **GPU Safety Guarantee**: Checkpoints are evaluated **sequentially on a single model instance**, avoiding duplicate memory allocations on the 6GB GPU.
- **How to use**:
  1. Check 2 or more checkpoints in the list.
  2. Enter a test prompt.
  3. Click **Compare Selected**.
  4. View generated outputs side by side.

#### Automated Benchmark Suite
Click **Run Benchmark Suite** to evaluate the model across 5 standardized test categories:
1. **Creator Persona**: 5 factual questions about creator Manu Cătălin.
2. **Hardware Knowledge**: 5 questions covering computer hardware, bfloat16, and the RTX 3060 6GB Laptop GPU.
3. **Romanian Understanding**: Diacritics coverage and grammar reconstruction.
4. **Programming**: PyTorch syntax and parameter counting.
5. **Mathematics**: Single-digit atomized arithmetic reasoning.

---

### Tab 5: 💾 Runs & Checkpoints

Inspect historical records and manage disk storage.

#### Training Runs History
- Lists all previous training sessions stored in the SQLite database.
- Shows `Run ID`, `Status`, `Phase`, `Total Steps`, `Tokens Processed`, `Best Val Loss`, and timestamps.

#### Checkpoints on Disk Table
- Lists all `.pt` files in the `checkpoints/` directory.
- Shows `Filename`, `Step`, `Val Loss`, `File Size (MB)`, and `Modified Date`.
- **Delete Button**: Deletes selected checkpoint files with confirmation to free up disk space.

---

### Tab 6: ⚙️ Settings & Telemetry

System diagnostics and architecture specifications.

#### NVIDIA NVML Hardware Diagnostics
- **Device Name**: `NVIDIA GeForce RTX 3060 Laptop GPU`
- **Driver Version**: Active NVIDIA display driver.
- **Total VRAM**: `6144.0 MB`
- **Used / Free VRAM**: System-wide memory usage.
- **SM / Memory Clock**: Active GPU clock frequencies in MHz.
- **Power Draw / Limit**: Real-time wattage vs TDP cap (80W).
- **PyTorch Allocator**: Allocated vs Reserved GPU cache.

#### RoCanon-48.5M Architecture Specifications
- **Total Parameters**: Exactly `48,531,968`.
- **Layers**: 16 Transformer Blocks.
- **Model Dimension ($d_{\text{model}}$)**: 512.
- **Attention**: 8 Query heads, 2 Key/Value heads (4:1 Grouped-Query Attention).
- **Feed-Forward ($d_{\text{ff}}$)**: 1,280 with SwiGLU activation.
- **Convolutions**: 4 depthwise causal 1D convolutions (Canon-A, B, C, D) with kernel size $k=4$.
- **Embeddings**: 12,288 vocabulary with tied input/output weights.

---

### Tab 7: 🧩 Network Studio (Architecture Builder)

Network Studio transforms the platform into an open local neural network experimentation environment, allowing you to design, inspect, validate, budget, and train novel architectures beyond the baseline RoCanon model.

#### Sub-Components:
- **Architecture Catalog**:
  - Browse registered network architectures stored in `networks/registry/`.
  - Load pre-packaged industry templates:
    1. **RoCanon-48.5M Baseline** (Deep-Thin Transformer++ with 4 Canon convolutions).
    2. **Mini-GPT 15M** (Fast iteration causal language model).
    3. **BERT Encoder 25M** (Bidirectional text encoder for classification or embeddings).
    4. **Hybrid CNN-Transformer 20M** (Acoustic and local sequence feature modeling).
    5. **Deep MLP 5M** (Feed-forward dense tabular / feature predictor).
    6. **1D Causal CNN 4M** (Low-latency acoustic or character sequence model).
  - Actions: **Load**, **Clone**, **Delete** (RoCanon baseline is write-protected), and **Export JSON**.
- **Live Parameter Calculator & Budget Assistant**:
  - Instant calculation of total parameters, trainable parameters, non-embedding parameters, attention weights, FFN weights, and norm parameters.
  - One-click budget presets: `1M`, `5M`, `15M`, `25M`, `48.5M`, `100M`.
  - Automatically recommends optimal $d_{\text{model}}$, layer count, and head ratios for any target parameter budget.
- **RTX 3060 6GB VRAM Memory Estimator**:
  - Live allocation breakdown: Model weights, FP32 master gradients, optimizer states (Muon / AdamW), activation footprint, and CUDA context headroom.
  - Hardware safety status: `SAFE` ($< 4.8\text{ GB}$), `TIGHT` ($4.8 - 5.3\text{ GB}$), or `UNLIKELY_TO_FIT` ($> 5.3\text{ GB}$).
  - Actionable advice: Suggests micro-batch reductions or gradient checkpointing when approaching the 5.0 GB safety threshold.
- **Visual Architecture Flow Graph**:
  - Node-based diagram showing the entire tensor path from input projection through stacked blocks down to the final task head.
- **Advanced Configuration Editor (JSON)**:
  - Full bidirectional JSON schema editing.
  - Supports heterogeneous layer specifications (e.g. alternating GQA ratios, varying FFN dimensions, or selective convolutions).
- **Preflight Verification**:
  - Runs dry-run forward and backward passes with gradient propagation verification on synthetic tensors before training.

---

## 4. Recommended Step-by-Step Training Workflow

Follow this sequence to train RoCanon-48.5M:

```
[Phase 1: Pretraining] ──► [best_model.pt] ──► [Phase 2: Domain Adaptation] ──► [Phase 3: SFT Chat]
```

1. **Start the WebUI**: Run `python dashboard.py` and open `http://localhost:8080`.
2. **Launch Phase 1 (Pretraining)**:
   - Go to the **Overview** tab.
   - Set Phase to `1. General Pretraining`.
   - Set Dataset to `tech_wiki_ro`.
   - Set Resume Checkpoint to `Start from Scratch`.
   - Click **Start (▶)**.
3. **Monitor Progress**:
   - Check the **Training & Loss** tab.
   - Watch the raw loss decrease (typically from ~9.5 down to ~3.0).
   - Validation runs automatically every 10 steps. When validation loss improves, `best_model.pt` is saved automatically.
4. **Pause or Take a Snapshot**:
   - Click **Snapshot (💾)** to save a checkpoint at any time.
   - Click **Stop (🛑)** to end the run gracefully; state is preserved in SQLite so you can resume later.
5. **Evaluate Model Quality**:
   - Go to the **Evaluation Lab** tab.
   - Test prompts using **Live Model Inference**.
   - Click **Run Benchmark Suite** to check fact coverage and language coherence.

---

## 5. Network Studio: Custom Neural Network Guide

Network Studio enables full-stack neural network development directly from the browser without editing core training engine source code.

### Creating a New Architecture
1. Navigate to **Tab 7: 🧩 Network Studio**.
2. Under **Architecture Templates**, select a starting point (e.g. `Mini-GPT 15M` or `Deep MLP 5M`) and click **Load into Editor**.
3. Customize the hyperparameters in the **Architecture Parameters** panel:
   - **Dimensions**: Set $d_{\text{model}}$ (e.g. 256, 384, 512, 768) and layer count.
   - **Attention**: Choose Query heads ($n_q$) and KV heads ($n_{kv}$) for GQA, MQA, or standard MHA.
   - **Feed-Forward**: Choose SwiGLU, GeGLU, or standard MLP with custom expansion factor ($d_{\text{ff}}$).
   - **Convolutions & Normalization**: Toggle causal depthwise 1D convolutions, choose RMSNorm vs LayerNorm, and select RoPE vs learned embeddings.
4. Click **Save to Registry** to store your configuration in `networks/registry/<id>.json`.

### Parameter Budget Assistant & RTX 3060 VRAM Estimator
- Click on any **Budget Preset** (`1M`, `5M`, `15M`, `25M`, `48.5M`, `100M`) to instantly balance width, depth, and attention heads for a given target parameter count.
- The **RTX 3060 Memory Estimator** updates in real time:
  - Displays exact MB for weights, gradients, optimizer states, activations, and PyTorch workspace headroom.
  - Verifies whether your configuration will fit within the **5.0 GB safe headroom** on your RTX 3060 Laptop GPU.

### Preflight Safety Check
Before training a newly designed model:
1. Click **Run Preflight Verification (🔍)**.
2. The platform performs:
   - **Schema & Dimension Check**: Verifies tensor divisibility ($n_q \pmod{n_{kv}} = 0$) and layer dimension consistency.
   - **Model Instantiation**: Builds the PyTorch graph in memory.
   - **Forward Pass**: Passes dry-run synthetic input tensors.
   - **Backward Pass & Gradient Flow**: Propagates loss backward and checks that 100% of trainable parameters receive valid non-zero gradients without NaN or Inf.
3. If successful, a green verification badge confirms that the architecture is safe for execution.

### Training a Custom Architecture
1. Go to **Tab 1: 📊 Overview**.
2. Under **Model Architecture / Network**, select your custom registered network (or leave default `rocanon_48m`).
3. Select your dataset and click **Start (▶)**. The training engine dynamically instantiates your custom architecture via `ModelFactory.create()`.
4. Checkpoints for custom networks are cleanly saved to `networks/checkpoints/<network_id>/` and tracked in SQLite.

### Developing Custom PyTorch Modules
You can extend Network Studio with novel neural network components without modifying platform files:
1. Create a Python file in `networks/custom/` (e.g. `networks/custom/my_attention.py`).
2. Subclass `CustomModuleBase`:
   ```python
   from src.networks.custom_loader import CustomModuleBase
   import torch
   import torch.nn as nn

   class MyNovelLayer(CustomModuleBase):
       module_name = "my_novel_layer"
       description = "My custom attention or activation block"
       parameter_schema = {
           "dim": {"type": "integer", "default": 512, "description": "Channel dimension"}
       }
       input_shape_requirements = "[Batch, Sequence, Dim]"
       output_shape_rules = "[Batch, Sequence, Dim]"

       def __init__(self, dim: int = 512, **kwargs):
           super().__init__()
           self.proj = nn.Linear(dim, dim)

       def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
           return self.proj(x)
   ```
3. Network Studio discovers your module automatically upon reload, displaying it in the component catalog and making it available for architectural integration.

---

## 6. Storage Layout & File Locations

- **Database**: `data/rocanon_studio.db` (SQLite WAL mode: stores runs, metrics, alerts, pipelines, benchmarks).
- **Checkpoints (RoCanon)**: `checkpoints/` (contains `.pt` weight files; `best_model.pt` contains top validation weights).
- **Custom Networks Registry**: `networks/registry/` (versioned JSON architecture definitions).
- **Custom Network Checkpoints**: `networks/checkpoints/` (weights for custom trained architectures).
- **Custom Developer Modules**: `networks/custom/` (drop-in PyTorch modules inheriting from `CustomModuleBase`).
- **Uploaded Datasets**: `data/uploads/` (uploaded `.txt`, `.jsonl`, `.json` files).
- **Pre-Tokenized Cache**: `data/cache/` (tokenized `.pt` files).
- **Configs**: `configs/` (YAML training and hardware configurations).

---

## 7. Troubleshooting & FAQ

### What happens if background apps consume too much VRAM?
The built-in **VRAM Guard** monitors available GPU memory every second. If free memory drops below $400\text{ MB}$, an alert is logged to SQLite, an emergency checkpoint is saved, and training pauses cleanly to prevent an unrecoverable CUDA crash.

### How do I resume training from a checkpoint?
In the **Overview** tab, select your saved checkpoint from the **Resume Checkpoint** dropdown (e.g. `best_model.pt` or `snapshot_step_500.pt`) and click **Start (▶)**.

### Can I run inference while training is active?
Inference in the Evaluation Lab is designed for paused runs or completed checkpoints. On a 6GB GPU, running heavy inference concurrently during active training can cause memory pressure. Pause training first with **Pause (⏸)** before running benchmark suites.


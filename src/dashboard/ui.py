"""
src/dashboard/ui.py
Professional Dark Technical UI for RoCanon-48.5M Studio v2.
Featuring 7 primary navigation tabs including the custom Network Studio Architecture Builder,
real-time WebSocket telemetry, interactive Chart.js charts, EMA smoothing, drag-and-drop dataset upload,
tokenizer inspector, live inference, side-by-side checkpoint comparison, and benchmark suites.
"""

INDEX_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RoCanon-48.5M | ML Training & Network Studio</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    if (typeof tailwind !== 'undefined') {
      tailwind.config = {
        darkMode: 'class',
        theme: {
          extend: {
            colors: {
              brand: { 400: '#00f0ff', 500: '#38bdf8', 600: '#0284c7' },
              darkbg: '#050811',
              darkcard: '#0a101f',
              darkcard2: '#0f172a',
              darkborder: 'rgba(56, 189, 248, 0.16)',
              accent: '#00f0ff',
            }
          }
        }
      };
    }

    // Early global switchTab definition so button onclick never throws ReferenceError
    function switchTab(tabId) {
      try {
        const tabs = ['overview', 'network_studio', 'training', 'datasets', 'evaluation', 'checkpoints', 'settings'];
        tabs.forEach(t => {
          const el = document.getElementById('tab-' + t);
          const btn = document.getElementById('tab-btn-' + t);
          if (!el) return;
          if (t === tabId) {
            el.classList.remove('hidden');
            el.style.display = 'block';
            if (btn) {
              btn.classList.add('tab-active');
              btn.classList.remove('text-slate-400');
            }
          } else {
            el.classList.add('hidden');
            el.style.display = 'none';
            if (btn) {
              btn.classList.remove('tab-active');
              btn.classList.add('text-slate-400');
            }
          }
        });

        // Trigger tab-specific initializations safely if main script is loaded
        try {
          if (tabId === 'training' && typeof initChartsIfNeeded === 'function') {
            setTimeout(initChartsIfNeeded, 100);
          } else if (tabId === 'overview' && typeof initOverviewChartsIfNeeded === 'function') {
            setTimeout(initOverviewChartsIfNeeded, 100);
          } else if (tabId === 'network_studio' && typeof refreshNetworkStudio === 'function') {
            refreshNetworkStudio();
          } else if (tabId === 'checkpoints') {
            if (typeof refreshCheckpointsTable === 'function') refreshCheckpointsTable();
            if (typeof refreshRunsTable === 'function') refreshRunsTable();
          } else if (tabId === 'datasets' && typeof refreshRegisteredDatasets === 'function') {
            refreshRegisteredDatasets();
          } else if (tabId === 'evaluation') {
            if (typeof refreshComparisonCheckpoints === 'function') refreshComparisonCheckpoints();
            if (typeof refreshBenchmarkHistory === 'function') refreshBenchmarkHistory();
            if (typeof refreshCheckpointsTable === 'function') refreshCheckpointsTable();
          } else if (tabId === 'settings') {
            if (typeof refreshAlertsTable === 'function') refreshAlertsTable();
            if (typeof loadWebhookSettings === 'function') loadWebhookSettings();
          }
        } catch (tabErr) {
          console.warn('Tab initialization warning:', tabErr);
        }
      } catch (err) {
        console.error('switchTab error:', err);
      }
    }

    function setTargetSteps(n) {
      const el = document.getElementById('config-target-steps');
      if (el) {
        el.value = n;
        onTargetStepsChange(n);
      }
    }

    function onTargetStepsChange(val) {
      const steps = parseInt(val, 10) || 1000;
      const elSCnt = document.getElementById('step-counter-text');
      const stepEl = document.getElementById('status-step');
      const currStep = stepEl ? stepEl.innerText : '0';
      if (elSCnt) elSCnt.innerText = 'Step ' + currStep + ' / ' + steps.toLocaleString();
    }
  </script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: #050811;
      background-image: 
        linear-gradient(to right, rgba(56, 189, 248, 0.03) 1px, transparent 1px),
        linear-gradient(to bottom, rgba(56, 189, 248, 0.03) 1px, transparent 1px);
      background-size: 32px 32px;
    }
    .font-mono { font-family: 'JetBrains Mono', monospace; }
    .tab-active {
      border-color: #00f0ff !important;
      color: #00f0ff !important;
      background-color: rgba(0, 240, 255, 0.12) !important;
      box-shadow: 0 0 12px rgba(0, 240, 255, 0.2) !important;
    }
    .glow-cyan { box-shadow: 0 0 20px -3px rgba(0, 240, 255, 0.3); }
    .glass-card {
      background: rgba(10, 16, 31, 0.75);
      backdrop-filter: blur(12px);
      -webkit-backdrop-filter: blur(12px);
      border: 1px solid rgba(56, 189, 248, 0.16);
    }
    .hidden { display: none !important; }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #050811; }
    ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #0284c7; }
  </style>
</head>
<body class="text-slate-200 antialiased min-h-screen flex flex-col justify-between">

  <!-- ================= TOP HEADER ================= -->
  <header class="border-b border-darkborder bg-darkcard/95 backdrop-blur sticky top-0 z-50 px-6 py-3">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3.5">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-blue-600 to-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20 text-lg">
          RC
        </div>
        <div>
          <div class="flex items-center gap-2.5">
            <h1 class="text-lg font-bold text-white tracking-tight">RoCanon-48.5M</h1>
            <span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-cyan-400 font-semibold border border-cyan-500/20">Studio v2</span>
            <span id="badge-connection" class="px-2 py-0.5 text-[11px] font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">CONNECTING...</span>
            <span id="badge-gpu" class="px-2 py-0.5 text-[11px] font-mono rounded bg-slate-800 text-slate-300">GPU: RTX 3060 (6GB)</span>
          </div>
          <p class="text-xs text-slate-400">Deep-Thin Canon Transformer++ & Custom Architecture Studio</p>
        </div>
      </div>

      <!-- Live VRAM Safety & Quick Master Controls -->
      <div class="flex items-center gap-3">
        <!-- VRAM Safety Indicator Pill -->
        <div id="vram-pill" class="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 text-xs font-semibold">
          <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
          <span id="vram-status-text">VRAM: Normal</span>
        </div>

        <!-- Master Run Controls -->
        <div class="flex items-center gap-2">
          <button id="btn-start" onclick="actionStartRun()" class="px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold transition flex items-center gap-1.5 shadow-lg shadow-emerald-600/20">
            <span>▶</span> <span id="start-btn-text">Start</span>
          </button>
          <button id="btn-pause" onclick="actionPause()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-white border border-slate-700 flex items-center gap-1.5 transition">
            <span id="pause-icon">⏸</span> <span id="pause-text">Pause</span>
          </button>
          <button onclick="actionValidate()" class="px-3 py-1.5 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 text-xs font-semibold transition flex items-center gap-1.5">
            🔍 <span>Validate</span>
          </button>
          <button onclick="actionSnapshot()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-semibold transition flex items-center gap-1.5">
            💾 <span>Snapshot</span>
          </button>
          <button id="btn-stop" onclick="actionStop()" class="px-3 py-1.5 rounded-lg bg-rose-500/10 hover:bg-rose-500/20 text-rose-400 border border-rose-500/30 text-xs font-semibold transition flex items-center gap-1.5">
            🛑 <span id="stop-btn-text">Stop</span>
          </button>
        </div>
      </div>
    </div>

    <!-- PRIMARY 7-TAB NAVIGATION -->
    <div class="max-w-7xl mx-auto flex items-center gap-1 mt-3 pt-2 border-t border-slate-800/80 overflow-x-auto text-xs font-medium">
      <button onclick="switchTab('overview')" id="tab-btn-overview" class="px-3.5 py-1.5 rounded-lg border border-transparent hover:text-white transition tab-active flex items-center gap-1.5">
        📊 <span>Overview</span>
      </button>
      <button onclick="switchTab('network_studio')" id="tab-btn-network_studio" class="px-3.5 py-1.5 rounded-lg border border-transparent text-slate-400 hover:text-white transition flex items-center gap-1.5">
        🧠 <span class="font-bold text-cyan-400">Network Studio</span>
      </button>
      <button onclick="switchTab('training')" id="tab-btn-training" class="px-3.5 py-1.5 rounded-lg border border-transparent text-slate-400 hover:text-white transition flex items-center gap-1.5">
        📈 <span>Training & Loss</span>
      </button>
      <button onclick="switchTab('datasets')" id="tab-btn-datasets" class="px-3.5 py-1.5 rounded-lg border border-transparent text-slate-400 hover:text-white transition flex items-center gap-1.5">
        📚 <span>Datasets</span>
      </button>
      <button onclick="switchTab('evaluation')" id="tab-btn-evaluation" class="px-3.5 py-1.5 rounded-lg border border-transparent text-slate-400 hover:text-white transition flex items-center gap-1.5">
        🧪 <span>Evaluation Lab</span>
      </button>
      <button onclick="switchTab('checkpoints')" id="tab-btn-checkpoints" class="px-3.5 py-1.5 rounded-lg border border-transparent text-slate-400 hover:text-white transition flex items-center gap-1.5">
        💾 <span>Runs & Checkpoints</span>
      </button>
      <button onclick="switchTab('settings')" id="tab-btn-settings" class="px-3.5 py-1.5 rounded-lg border border-transparent text-slate-400 hover:text-white transition flex items-center gap-1.5">
        ⚙️ <span>Settings & Telemetry</span>
      </button>
    </div>
  </header>

  <!-- ================= SUB-HEADER STATUS BAR ================= -->
  <section class="border-b border-darkborder bg-darkcard2/80 px-6 py-2.5 backdrop-blur text-xs">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-4">
        <div class="flex items-center gap-1.5">
          <span class="text-slate-400">Run State:</span>
          <span id="status-run-state" class="font-mono font-bold text-cyan-400 px-2 py-0.5 rounded bg-slate-900 border border-slate-800">IDLE</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span class="text-slate-400">Phase:</span>
          <span id="status-phase" class="font-semibold text-slate-200">1. Pretraining</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span class="text-slate-400">Step:</span>
          <span id="status-step" class="font-mono font-bold text-white">0</span>
        </div>
        <div class="flex items-center gap-1.5">
          <span class="text-slate-400">Tokens:</span>
          <span id="status-tokens" class="font-mono text-slate-300">0</span>
        </div>
      </div>

      <!-- Dynamic Learning Rate Multiplier Slider -->
      <div class="flex items-center gap-3">
        <span class="text-slate-400">Dynamic LR Multiplier:</span>
        <input type="range" id="slider-lr-mult" min="0.1" max="3.0" step="0.05" value="1.0" oninput="updateLrMultiplier(this.value)" class="w-28 accent-cyan-400 cursor-pointer">
        <span id="label-lr-mult" class="font-mono font-bold text-cyan-400">1.00x</span>
      </div>
    </div>
  </section>

  <!-- ================= MAIN CONTENT WRAPPER ================= -->
  <main class="max-w-7xl mx-auto w-full px-6 py-6 flex-1 space-y-6">

    <!-- ================= TAB 1: OVERVIEW ================= -->
    <div id="tab-overview" class="space-y-6" style="display: block;">
      <!-- KPI Top Metrics Cards -->
      <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3.5">
        <div class="bg-darkcard border border-darkborder rounded-xl p-3.5 flex flex-col justify-between">
          <span class="text-slate-400 text-[11px] font-semibold uppercase">Train Loss</span>
          <div class="flex items-baseline gap-1 mt-1">
            <span id="kpi-train-loss" class="text-xl font-mono font-bold text-white">--</span>
          </div>
          <span class="text-[10px] text-slate-500">Chunked CE</span>
        </div>

        <div class="bg-darkcard border border-darkborder rounded-xl p-3.5 flex flex-col justify-between">
          <span class="text-slate-400 text-[11px] font-semibold uppercase">Validation Loss</span>
          <div class="flex items-baseline gap-1 mt-1">
            <span id="kpi-val-loss" class="text-xl font-mono font-bold text-cyan-400">--</span>
          </div>
          <span id="kpi-gen-gap" class="text-[10px] text-slate-400">Gap: --</span>
        </div>

        <div class="bg-darkcard border border-darkborder rounded-xl p-3.5 flex flex-col justify-between">
          <span class="text-slate-400 text-[11px] font-semibold uppercase">Gradient Norm</span>
          <div class="flex items-baseline gap-1 mt-1">
            <span id="kpi-grad-norm" class="text-xl font-mono font-bold text-indigo-400">--</span>
          </div>
          <span class="text-[10px] text-slate-500">Threshold: 1.0</span>
        </div>

        <div class="bg-darkcard border border-darkborder rounded-xl p-3.5 flex flex-col justify-between">
          <span class="text-slate-400 text-[11px] font-semibold uppercase">Throughput</span>
          <div class="flex items-baseline gap-1 mt-1">
            <span id="kpi-tok-per-sec" class="text-xl font-mono font-bold text-emerald-400">--</span>
            <span class="text-xs text-slate-400">tok/s</span>
          </div>
          <span id="kpi-tflops" class="text-[10px] text-slate-500">-- TFLOPS</span>
        </div>

        <div class="bg-darkcard border border-darkborder rounded-xl p-3.5 flex flex-col justify-between">
          <span class="text-slate-400 text-[11px] font-semibold uppercase">GPU Temp & Util</span>
          <div class="flex items-baseline gap-1 mt-1">
            <span id="kpi-gpu-temp" class="text-xl font-mono font-bold text-amber-400">--°C</span>
            <span id="kpi-gpu-util" class="text-xs text-slate-400 font-mono">(--%)</span>
          </div>
          <span id="kpi-gpu-power" class="text-[10px] text-slate-500">-- W</span>
        </div>

        <div class="bg-darkcard border border-darkborder rounded-xl p-3.5 flex flex-col justify-between">
          <span class="text-slate-400 text-[11px] font-semibold uppercase">Free VRAM</span>
          <div class="flex items-baseline gap-1 mt-1">
            <span id="kpi-free-vram" class="text-xl font-mono font-bold text-white">--</span>
            <span class="text-xs text-slate-400">MB</span>
          </div>
          <span id="kpi-vram-pt" class="text-[10px] text-slate-500">PT: -- GB</span>
        </div>
      </div>

      <!-- LIVE PIPELINE & TOKENIZATION MONITOR -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <!-- Header row -->
        <div class="flex flex-wrap items-center justify-between gap-2 border-b border-darkborder/70 pb-3">
          <div class="flex items-center gap-2.5">
            <span class="flex h-2.5 w-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <h3 class="text-xs font-bold uppercase tracking-wider text-white">Live Pipeline & Tokenization Monitor</h3>
            <span id="pipeline-active-badge" class="px-2 py-0.5 rounded text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">READY</span>
          </div>
          <div class="flex items-center gap-4 text-xs text-slate-400">
            <div>Dataset: <span id="pipeline-dataset-name" class="font-mono text-slate-200 font-semibold">tech_wiki_ro.txt</span></div>
            <div>Phase: <span id="pipeline-phase-name" class="font-mono text-cyan-400 font-semibold">Pretrain</span></div>
            <div>Early Stop: <span id="pipeline-es-status" class="font-mono text-emerald-400 font-semibold">Armed (0/5)</span></div>
            <div>ETA: <span id="pipeline-eta" class="font-mono text-emerald-400 font-semibold">--:--:--</span></div>
          </div>
        </div>

        <!-- 3 Progress Bar Modules Grid -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <!-- 1. Dataset & Tokenization Stage -->
          <div class="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-2">
            <div class="flex items-center justify-between text-xs">
              <span class="text-slate-400 font-medium">1. Tokenization & Shards</span>
              <span id="tok-progress-pct" class="font-mono font-bold text-cyan-400">100%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div id="tok-progress-bar" class="bg-gradient-to-r from-blue-500 to-cyan-400 h-2 rounded-full transition-all duration-300" style="width: 100%;"></div>
            </div>
            <div class="flex items-center justify-between text-[11px] text-slate-400 pt-0.5">
              <span id="tok-status-text" class="truncate max-w-[200px]">48,419,639 tokens cached</span>
              <span id="tok-cache-badge" class="text-emerald-400 font-mono text-[10px]">DISK CACHED</span>
            </div>
          </div>

          <!-- 2. Dataset Epoch Progress -->
          <div class="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-2">
            <div class="flex items-center justify-between text-xs">
              <span class="text-slate-400 font-medium">2. Dataset Epoch Cycle</span>
              <span id="epoch-progress-pct" class="font-mono font-bold text-indigo-400">0.0%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div id="epoch-progress-bar" class="bg-gradient-to-r from-cyan-500 to-indigo-500 h-2 rounded-full transition-all duration-300" style="width: 0%;"></div>
            </div>
            <div class="flex items-center justify-between text-[11px] text-slate-400 pt-0.5">
              <span id="epoch-details-text">Epoch 1</span>
              <span id="epoch-tokens-text" class="font-mono text-[10px]">0 / 48.4M tok</span>
            </div>
          </div>

          <!-- 3. Target Training Steps & Tokens -->
          <div class="bg-slate-900/60 border border-slate-800 rounded-lg p-3.5 space-y-2">
            <div class="flex items-center justify-between text-xs">
              <span class="text-slate-400 font-medium">3. Run Step Target</span>
              <span id="step-progress-pct" class="font-mono font-bold text-emerald-400">0.0%</span>
            </div>
            <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
              <div id="step-progress-bar" class="bg-gradient-to-r from-indigo-500 to-emerald-400 h-2 rounded-full transition-all duration-300" style="width: 0%;"></div>
            </div>
            <div class="flex items-center justify-between text-[11px] text-slate-400 pt-0.5">
              <span id="step-counter-text">Step 0 / 1,000</span>
              <span id="speed-indicator-text" class="font-mono text-[10px] text-slate-400">-- tok/s</span>
            </div>
          </div>
        </div>

        <!-- Real-time Activity Terminal Line -->
        <div class="bg-black/40 rounded border border-slate-800/80 px-3 py-2 flex items-center justify-between text-xs font-mono">
          <div class="flex items-center gap-2 truncate">
            <span class="text-cyan-400">λ</span>
            <span class="text-slate-500">Pipeline:</span>
            <span id="pipeline-live-log" class="text-slate-300 truncate">Ready. Click Start to stream tokens and train RoCanon-48.5M on CUDA.</span>
          </div>
          <div class="flex items-center gap-3 text-[11px] text-slate-400 flex-shrink-0">
            <span>Vocab: <span class="text-slate-200">12,288</span></span>
            <span>Ctx: <span class="text-slate-200">256</span></span>
            <span>Batch: <span class="text-slate-200">2,048 tok/step</span></span>
          </div>
        </div>
      </div>

      <!-- Quick Setup & Run Launcher -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div class="lg:col-span-2 bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <span>🚀</span> Active Session Configuration
            </h2>
            <span class="text-xs text-slate-400">Local-first RTX 3060 Safe</span>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
            <div>
              <label class="text-[11px] font-semibold text-slate-400 uppercase">Architecture</label>
              <select id="config-network" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-cyan-300 font-semibold focus:outline-none focus:border-cyan-500">
                <option value="rocanon_48m">RoCanon-48.5M (Default Baseline)</option>
              </select>
            </div>
            <div>
              <label class="text-[11px] font-semibold text-slate-400 uppercase">Phase</label>
              <select id="config-phase" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500">
                <option value="pretrain">1. General Pretraining</option>
                <option value="midtrain">2. Domain Adaptation</option>
                <option value="chat_sft">3. Instruction Tuning (SFT)</option>
                <option value="persona">4. Creator Persona</option>
              </select>
            </div>
            <div>
              <label class="text-[11px] font-semibold text-slate-400 uppercase">Dataset Source</label>
              <select id="config-dataset" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500">
                <option value="tech_wiki_ro">tech_wiki_ro (Wikipedia & Tech)</option>
                <option value="alpaca_ro">alpaca_ro (Instruction Tuning)</option>
                <option value="chat_persona_ro">chat_persona_ro (Creator Persona)</option>
              </select>
            </div>
            <div>
              <label class="text-[11px] font-semibold text-slate-400 uppercase">Resume Checkpoint</label>
              <select id="config-checkpoint" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500">
                <option value="">Start from Scratch</option>
              </select>
            </div>
          </div>

          <!-- Row 2: Training Budget & Steps Controls -->
          <div class="border-t border-slate-800/80 pt-3">
            <div class="flex items-center justify-between mb-2">
              <span class="text-[11px] font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                <span>⏱</span> Training Target & Early Stopping
              </span>
              <span class="text-[10px] text-slate-400">Step-driven training with automatic generalization guard</span>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
              <!-- Target Steps with Preset Buttons -->
              <div>
                <div class="flex items-center justify-between">
                  <label class="text-[11px] font-semibold text-slate-400 uppercase">Target Steps</label>
                  <div class="flex items-center gap-1">
                    <button type="button" onclick="setTargetSteps(100)" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-[9px] text-slate-300">100</button>
                    <button type="button" onclick="setTargetSteps(500)" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-[9px] text-slate-300">500</button>
                    <button type="button" onclick="setTargetSteps(1000)" class="px-1.5 py-0.5 rounded bg-cyan-950 border border-cyan-500/40 hover:bg-cyan-900 text-[9px] text-cyan-300 font-bold">1k</button>
                    <button type="button" onclick="setTargetSteps(5000)" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-[9px] text-slate-300">5k</button>
                    <button type="button" onclick="setTargetSteps(10000)" class="px-1.5 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-[9px] text-slate-300">10k</button>
                  </div>
                </div>
                <input type="number" id="config-target-steps" value="1000" min="1" max="1000000" step="50" oninput="onTargetStepsChange(this.value)" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono font-bold focus:outline-none focus:border-cyan-500">
              </div>

              <!-- Validation Frequency -->
              <div>
                <label class="text-[11px] font-semibold text-slate-400 uppercase">Validate Every (Steps)</label>
                <input type="number" id="config-eval-freq" value="25" min="1" max="10000" step="5" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-cyan-500">
              </div>

              <!-- Save Checkpoint Frequency -->
              <div>
                <label class="text-[11px] font-semibold text-slate-400 uppercase">Save Checkpoint Every</label>
                <input type="number" id="config-save-freq" value="50" min="5" max="50000" step="25" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono focus:outline-none focus:border-cyan-500">
              </div>

              <!-- Early Stopping Controls -->
              <div class="bg-slate-900/80 border border-slate-700/80 rounded-lg p-2 space-y-1.5">
                <div class="flex items-center justify-between">
                  <label class="text-[11px] font-semibold text-emerald-400 uppercase flex items-center gap-1">
                    <span>🛑</span> Early Stopping
                  </label>
                  <label class="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" id="config-es-enabled" class="sr-only peer">
                    <div class="w-8 h-4 bg-slate-800 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:bg-emerald-500"></div>
                  </label>
                </div>
                <div class="grid grid-cols-2 gap-1.5">
                  <div>
                    <span class="text-[9px] text-slate-400 uppercase block">Patience</span>
                    <input type="number" id="config-es-patience" value="20" min="1" max="500" class="w-full bg-slate-950 border border-slate-700 rounded px-1.5 py-0.5 text-xs text-white font-mono focus:outline-none focus:border-emerald-500">
                  </div>
                  <div>
                    <span class="text-[9px] text-slate-400 uppercase block">Min Delta</span>
                    <input type="number" id="config-es-min-delta" value="0.001" min="0" step="0.0005" class="w-full bg-slate-950 border border-slate-700 rounded px-1.5 py-0.5 text-xs text-white font-mono focus:outline-none focus:border-emerald-500">
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Row 3: Optimization & GPU Hyperparameters -->
          <div class="border-t border-slate-800/80 pt-3">
            <div class="flex items-center justify-between mb-2">
              <span class="text-[11px] font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <span>⚙️</span> Compute & Optimizer Hyperparameters
              </span>
              <span class="text-[10px] text-emerald-400 font-mono">RTX 3060 6GB Safe Defaults</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div>
                <div class="flex items-center justify-between">
                  <label class="text-[10px] text-slate-400 uppercase">Micro-Batch</label>
                  <div class="flex items-center gap-1">
                    <button type="button" onclick="setMicroBatch(4)" class="px-1 py-0.2 rounded bg-slate-800 hover:bg-slate-700 text-[9px] text-slate-300">4</button>
                    <button type="button" onclick="setMicroBatch(8)" class="px-1 py-0.2 rounded bg-cyan-950 border border-cyan-500/40 hover:bg-cyan-900 text-[9px] text-cyan-300 font-bold">8</button>
                    <button type="button" onclick="setMicroBatch(12)" class="px-1 py-0.2 rounded bg-slate-800 hover:bg-slate-700 text-[9px] text-slate-300">12</button>
                    <button type="button" onclick="setMicroBatch(16)" class="px-1 py-0.2 rounded bg-slate-800 hover:bg-slate-700 text-[9px] text-slate-300">16</button>
                  </div>
                </div>
                <input type="number" id="config-batch-size" value="8" min="1" max="16" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white font-mono focus:outline-none focus:border-cyan-500">
              </div>
              <div>
                <label class="text-[10px] text-slate-400 uppercase block">Grad Accum Steps</label>
                <input type="number" id="config-grad-accum" value="2" min="1" max="32" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-white font-mono focus:outline-none focus:border-cyan-500">
              </div>
              <div>
                <div class="flex items-center justify-between">
                  <label class="text-[10px] text-slate-400 uppercase">Muon LR</label>
                  <span id="label-live-muon" class="text-[9px] font-mono text-cyan-400">Live</span>
                </div>
                <input type="number" id="config-muon-lr" value="0.003" step="0.0005" onchange="applyLiveLR()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-cyan-300 font-mono focus:outline-none focus:border-cyan-500">
              </div>
              <div>
                <div class="flex items-center justify-between">
                  <label class="text-[10px] text-slate-400 uppercase">AdamW LR</label>
                  <span id="label-live-adamw" class="text-[9px] font-mono text-indigo-400">Live</span>
                </div>
                <input type="number" id="config-adamw-lr" value="0.0006" step="0.0001" onchange="applyLiveLR()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-indigo-300 font-mono focus:outline-none focus:border-cyan-500">
              </div>
            </div>

            <!-- Dynamic Learning Rate Live Speed / Multiplier Ribbon -->
            <div class="mt-3 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 text-xs">
              <label class="flex items-center gap-1.5 cursor-pointer text-[11px] text-slate-300 select-none">
                <input type="checkbox" id="config-auto-lr" checked onchange="applyLiveLR()" class="accent-cyan-500 rounded">
                <span>⚡ Auto-Adaptive LR (Boost on stability / Protect on spikes)</span>
              </label>
              <div class="flex items-center gap-1.5">
                <span class="text-[10px] text-slate-400 font-mono">Live Speed:</span>
                <button type="button" onclick="adjustLiveLRMultiplier(0.75)" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-amber-300 font-mono text-[10px] border border-slate-700">-25%</button>
                <button type="button" onclick="adjustLiveLRMultiplier(1.0)" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 font-mono text-[10px] border border-slate-700">1.0x</button>
                <button type="button" onclick="adjustLiveLRMultiplier(1.25)" class="px-2 py-0.5 rounded bg-cyan-950 hover:bg-cyan-900 text-cyan-300 font-mono text-[10px] border border-cyan-500/40 font-bold">+25%</button>
                <button type="button" onclick="adjustLiveLRMultiplier(1.5)" class="px-2 py-0.5 rounded bg-slate-800 hover:bg-slate-700 text-emerald-300 font-mono text-[10px] border border-slate-700">+50%</button>
              </div>
            </div>
          </div>
        </div>

        <!-- Latest Alerts & System Events -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-5 flex flex-col justify-between">
          <div>
            <div class="flex items-center justify-between mb-3">
              <h2 class="text-sm font-bold text-white flex items-center gap-1.5">
                <span>⚠️</span> Real-Time Alerts
              </h2>
              <span id="alerts-count-badge" class="px-2 py-0.5 rounded text-[10px] bg-slate-800 text-slate-400">0 Alerts</span>
            </div>
            <div id="alerts-container" class="space-y-2 max-h-48 overflow-y-auto pr-1 text-xs">
              <div class="text-slate-500 italic text-center py-4">No critical alerts detected. System healthy.</div>
            </div>
          </div>
          <button onclick="clearAlerts()" class="mt-3 text-[11px] text-slate-400 hover:text-white transition text-right">Clear Alerts</button>
        </div>
      </div>

      <!-- Live Overview Telemetry & Training Charts -->
      <div class="space-y-6 pt-2">
        <!-- Main Interactive Loss Chart Card -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
          <div class="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 class="text-sm font-bold text-white flex items-center gap-2">
                <span>📈</span> Live Loss &amp; Convergence Monitor
              </h2>
              <p class="text-xs text-slate-400">Real-time Train Loss, Validation Loss, and Exponential Moving Average (EMA)</p>
            </div>

            <div class="flex flex-wrap items-center gap-3 text-xs">
              <!-- EMA Smoothing Slider -->
              <div class="flex items-center gap-2 bg-slate-900 px-3 py-1 rounded-lg border border-slate-800">
                <span class="text-slate-400 text-[11px]">EMA Smoothing:</span>
                <input type="range" id="slider-ema-alpha-ov" min="0.01" max="0.30" step="0.01" value="0.08" oninput="updateEmaAlpha(this.value)" class="w-20 accent-cyan-400 cursor-pointer">
                <span id="label-ema-alpha-ov" class="font-mono text-cyan-400 font-semibold">0.08</span>
              </div>

              <!-- Linear vs Logarithmic Toggle -->
              <button onclick="toggleLogScale()" id="btn-scale-toggle-ov" class="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">
                Scale: <span id="label-scale-type-ov" class="font-semibold text-white">Linear</span>
              </button>

              <!-- Reset Zoom -->
              <button onclick="resetChartZoom()" class="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">
                Reset Zoom
              </button>
            </div>
          </div>

          <div class="h-80 w-full relative">
            <canvas id="chart-loss-overview"></canvas>
          </div>
        </div>

        <!-- Auxiliary Telemetry Charts Grid -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <!-- Chart 1: Gradient Norm -->
          <div class="bg-darkcard border border-darkborder rounded-xl p-4">
            <div class="flex items-center justify-between mb-2">
              <span class="text-xs font-bold text-white">Gradient Norm</span>
              <span class="text-[10px] text-slate-400 font-mono">Clip: 1.0</span>
            </div>
            <div class="h-44 w-full">
              <canvas id="chart-grad-overview"></canvas>
            </div>
          </div>

          <!-- Chart 2: Learning Rate (Muon + AdamW) -->
          <div class="bg-darkcard border border-darkborder rounded-xl p-4">
            <div class="flex items-center justify-between mb-2">
              <span class="text-xs font-bold text-white">Learning Rate</span>
              <span class="text-[10px] text-cyan-400 font-mono">Muon &amp; AdamW</span>
            </div>
            <div class="h-44 w-full">
              <canvas id="chart-lr-overview"></canvas>
            </div>
          </div>

          <!-- Chart 3: Throughput (tok/s) -->
          <div class="bg-darkcard border border-darkborder rounded-xl p-4">
            <div class="flex items-center justify-between mb-2">
              <span class="text-xs font-bold text-white">Throughput (tok/s)</span>
              <span class="text-[10px] text-emerald-400 font-mono">Active Speed</span>
            </div>
            <div class="h-44 w-full">
              <canvas id="chart-speed-overview"></canvas>
            </div>
          </div>

          <!-- Chart 4: VRAM Peak -->
          <div class="bg-darkcard border border-darkborder rounded-xl p-4">
            <div class="flex items-center justify-between mb-2">
              <span class="text-xs font-bold text-white">PyTorch Allocated VRAM</span>
              <span class="text-[10px] text-slate-400 font-mono">Max: 5.0 GB</span>
            </div>
            <div class="h-44 w-full">
              <canvas id="chart-vram-overview"></canvas>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ================= TAB: NETWORK STUDIO ================= -->
    <div id="tab-network_studio" class="hidden space-y-6" style="display: none;">
      <!-- Studio Header & Action Ribbon -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-4">
          <div class="flex items-center gap-3">
            <div class="w-10 h-10 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 text-lg font-bold">
              ⚡
            </div>
            <div>
              <div class="flex items-center gap-2">
                <h2 id="studio-active-title" class="text-base font-bold text-white">RoCanon-48.5M Baseline</h2>
                <span id="studio-active-family" class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 uppercase">DECODER_ONLY</span>
                <span id="studio-active-params" class="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-slate-800 text-slate-300">48,531,968 Params</span>
              </div>
              <p class="text-xs text-slate-400">Create, configure, validate, and train custom neural network architectures</p>
            </div>
          </div>

          <!-- Mode Toggles & Action Buttons -->
          <div class="flex flex-wrap items-center gap-2">
            <div class="inline-flex rounded-lg bg-slate-900 p-1 border border-slate-800 text-xs">
              <button id="btn-mode-visual" onclick="switchNetworkMode('visual')" class="px-3 py-1 rounded-md bg-cyan-600 text-white font-semibold transition">Visual Builder</button>
              <button id="btn-mode-json" onclick="switchNetworkMode('json')" class="px-3 py-1 rounded-md text-slate-400 hover:text-white transition">Config JSON</button>
            </div>
            <button onclick="runArchitecturePreflight()" class="px-3 py-1.5 rounded-lg bg-indigo-600/20 hover:bg-indigo-600/30 text-indigo-300 border border-indigo-500/30 text-xs font-semibold transition flex items-center gap-1.5">
              <span>🩺</span> <span>Preflight Check</span>
            </button>
            <button onclick="saveActiveNetwork()" class="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-bold transition flex items-center gap-1.5 shadow-lg shadow-emerald-600/20">
              <span>💾</span> <span>Save to Registry</span>
            </button>
            <button onclick="cloneActiveNetwork()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-semibold transition">
              <span>🧬</span> <span>Clone</span>
            </button>
          </div>
        </div>

        <!-- Architecture Family Templates & Parameter Budget Assistant -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2 border-t border-slate-800/80 text-xs">
          <div>
            <span class="text-slate-400 text-[11px] font-semibold block mb-1">Architecture Family Template:</span>
            <select id="studio-template-select" onchange="selectNetworkTemplate(this.value)" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-slate-200">
              <option value="rocanon_48m">RoCanon-48.5M (Deep-Thin Canon Transformer++)</option>
              <option value="mini_gpt_15m">Mini-GPT 15M (Lightweight Decoder-Only)</option>
              <option value="bert_encoder_25m">BERT Encoder 25M (Encoder-Only Transformer)</option>
              <option value="hybrid_cnn_transformer_20m">Hybrid CNN-Transformer 20M (1D Conv + GQA)</option>
              <option value="deep_mlp_5m">Deep MLP Network 5M (Feed-Forward Stack)</option>
              <option value="conv1d_network_4m">1D Causal CNN Network 4M (Pure Convolutions)</option>
            </select>
          </div>
          <div>
            <span class="text-slate-400 text-[11px] font-semibold block mb-1">Parameter Budget Assistant:</span>
            <div class="flex items-center gap-1.5 flex-wrap">
              <button onclick="applyBudgetPreset('1M')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 font-mono text-[11px]">1M</button>
              <button onclick="applyBudgetPreset('5M')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 font-mono text-[11px]">5M</button>
              <button onclick="applyBudgetPreset('15M')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 font-mono text-[11px]">15M</button>
              <button onclick="applyBudgetPreset('25M')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 font-mono text-[11px]">25M</button>
              <button onclick="applyBudgetPreset('48.5M')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 font-mono text-[11px]">48.5M</button>
              <button onclick="applyBudgetPreset('100M')" class="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 font-mono text-[11px]">100M</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Main Visual Builder / JSON Editor Grid -->
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <!-- Center Left 2 Cols: Visual Canvas or JSON Editor -->
        <div class="lg:col-span-2 space-y-6">

          <!-- Visual Mode Container -->
          <div id="studio-view-visual" class="space-y-4">
            <!-- Component Palette -->
            <div class="bg-darkcard border border-darkborder rounded-xl p-4">
              <span class="text-xs font-bold text-white block mb-2">Available Component Palette</span>
              <div class="flex flex-wrap gap-2 text-[11px]">
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-cyan-300 font-mono">Token Embedding</span>
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-cyan-300 font-mono">RoPE (Rotary)</span>
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-indigo-300 font-mono">Grouped Query Attention (GQA)</span>
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-indigo-300 font-mono">Multi-Head Attention (MHA)</span>
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-amber-300 font-mono">SwiGLU FFN</span>
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-amber-300 font-mono">Standard MLP</span>
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-emerald-300 font-mono">RMSNorm</span>
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-emerald-300 font-mono">LayerNorm</span>
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-rose-300 font-mono">Canon Conv1D (k=4)</span>
                <span class="px-2 py-1 rounded bg-slate-900 border border-slate-800 text-purple-300 font-mono">Custom Module (GatedConv)</span>
              </div>
            </div>

            <!-- Visual Transformer Architecture Configuration Form -->
            <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
              <h3 class="text-xs font-bold text-white uppercase tracking-wider">Architecture Dimensions & Properties</h3>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">Network Name</label>
                  <input type="text" id="builder-name" value="RoCanon-48.5M" oninput="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200">
                </div>
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">Model Dim (d_model)</label>
                  <input type="number" id="builder-dmodel" value="512" step="32" oninput="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
                </div>
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">Layers (n_layers)</label>
                  <input type="number" id="builder-layers" value="16" step="1" oninput="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
                </div>
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">Vocabulary Size</label>
                  <input type="number" id="builder-vocab" value="12288" step="256" oninput="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
                </div>
              </div>

              <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">Query Heads (n_q)</label>
                  <input type="number" id="builder-nq" value="8" step="1" oninput="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
                </div>
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">KV Heads (n_kv)</label>
                  <input type="number" id="builder-nkv" value="2" step="1" oninput="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
                </div>
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">FFN Dim (d_ff)</label>
                  <input type="number" id="builder-dff" value="1280" step="64" oninput="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
                </div>
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">Max Seq Length</label>
                  <input type="number" id="builder-seqlen" value="2048" step="128" oninput="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
                </div>
              </div>

              <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">Normalization</label>
                  <select id="builder-norm" onchange="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200">
                    <option value="rmsnorm">RMSNorm</option>
                    <option value="layernorm">LayerNorm</option>
                    <option value="batchnorm">BatchNorm</option>
                  </select>
                </div>
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">Activation</label>
                  <select id="builder-act" onchange="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200">
                    <option value="swiglu">SwiGLU</option>
                    <option value="geglu">GeGLU</option>
                    <option value="gelu">GELU</option>
                    <option value="silu">SiLU</option>
                    <option value="relu">ReLU</option>
                  </select>
                </div>
                <div>
                  <label class="text-[10px] text-slate-400 font-semibold uppercase">Positional Encoding</label>
                  <select id="builder-pos" onchange="updateArchitectureFromForm()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200">
                    <option value="rope">RoPE (Rotary)</option>
                    <option value="learned">Learned Positional</option>
                    <option value="sinusoidal">Sinusoidal</option>
                    <option value="none">None</option>
                  </select>
                </div>
                <div class="flex flex-col justify-end">
                  <label class="inline-flex items-center gap-2 cursor-pointer mt-2">
                    <input type="checkbox" id="builder-tied" checked onchange="updateArchitectureFromForm()" class="accent-cyan-400">
                    <span class="text-slate-300 text-xs">Tied Embeddings</span>
                  </label>
                  <label class="inline-flex items-center gap-2 cursor-pointer mt-1">
                    <input type="checkbox" id="builder-conv" checked onchange="updateArchitectureFromForm()" class="accent-cyan-400">
                    <span class="text-slate-300 text-xs">Canon Conv1D (k=4)</span>
                  </label>
                </div>
              </div>
            </div>

            <!-- Visual Computation Graph View -->
            <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
              <h3 class="text-xs font-bold text-white uppercase tracking-wider flex items-center justify-between">
                <span>Visual Computation Graph Flow</span>
                <span id="graph-tensor-shape-info" class="text-cyan-400 font-mono text-[11px] font-normal">Shape: [B, S, 512]</span>
              </h3>
              <div id="studio-graph-canvas" class="space-y-2 max-h-96 overflow-y-auto pr-1"></div>
            </div>
          </div>

          <!-- Advanced JSON Mode Container -->
          <div id="studio-view-json" class="hidden space-y-4">
            <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
              <div class="flex items-center justify-between">
                <h3 class="text-xs font-bold text-white uppercase tracking-wider">Raw Architecture Configuration (JSON)</h3>
                <button onclick="formatActiveJson()" class="text-xs text-cyan-400 hover:text-cyan-300">Format JSON</button>
              </div>
              <textarea id="studio-json-editor" rows="18" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500" oninput="syncJsonToForm()"></textarea>
            </div>
          </div>

        </div>

        <!-- Right Col: Live Parameter Calculator & RTX 3060 Memory Estimator -->
        <div class="space-y-6">

          <!-- Live Parameter Calculator Card -->
          <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
            <h3 class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
              <span>🧮</span> Live Parameter Breakdown
            </h3>
            <div class="space-y-2 text-xs font-mono">
              <div class="flex justify-between py-1 border-b border-slate-800">
                <span class="text-slate-400">Total Parameters:</span>
                <span id="calc-total-params" class="text-cyan-400 font-bold text-sm">--</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800">
                <span class="text-slate-400">Non-Embedding:</span>
                <span id="calc-non-emb-params" class="text-white">--</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800">
                <span class="text-slate-400">Tied Token Embedding:</span>
                <span id="calc-emb-params" class="text-slate-300">--</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800">
                <span class="text-slate-400">Attention (All Layers):</span>
                <span id="calc-attn-params" class="text-indigo-300">--</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800">
                <span class="text-slate-400">FFN (All Layers):</span>
                <span id="calc-ffn-params" class="text-amber-300">--</span>
              </div>
              <div class="flex justify-between py-1">
                <span class="text-slate-400">Canon Convolutions:</span>
                <span id="calc-conv-params" class="text-rose-300">--</span>
              </div>
            </div>
          </div>

          <!-- RTX 3060 6GB VRAM & Compute Estimator Card -->
          <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
            <div class="flex items-center justify-between">
              <h3 class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                <span>⚡</span> RTX 3060 (6GB) VRAM Estimator
              </h3>
              <span id="calc-vram-status-badge" class="px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">SAFE</span>
            </div>

            <div class="grid grid-cols-2 gap-2 text-xs">
              <div>
                <label class="text-[10px] text-slate-400 font-semibold uppercase">Micro-Batch</label>
                <select id="estimator-batch" onchange="calculateArchitectureEstimates()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200">
                  <option value="1">1</option>
                  <option value="2">2</option>
                  <option value="4" selected>4</option>
                  <option value="8">8</option>
                </select>
              </div>
              <div>
                <label class="text-[10px] text-slate-400 font-semibold uppercase">Seq Length</label>
                <select id="estimator-seqlen" onchange="calculateArchitectureEstimates()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200">
                  <option value="128">128</option>
                  <option value="256" selected>256</option>
                  <option value="512">512</option>
                  <option value="1024">1024</option>
                  <option value="2048">2048</option>
                </select>
              </div>
            </div>

            <div class="space-y-1.5 text-xs font-mono pt-1">
              <div class="flex justify-between py-1 border-b border-slate-800">
                <span class="text-slate-400">Weights (bfloat16):</span>
                <span id="calc-mem-weights" class="text-slate-200">-- MB</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800">
                <span class="text-slate-400">Gradients (FP32):</span>
                <span id="calc-mem-grads" class="text-slate-200">-- MB</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800">
                <span class="text-slate-400">Optimizer States:</span>
                <span id="calc-mem-opt" class="text-slate-200">-- MB</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800">
                <span class="text-slate-400">Activations:</span>
                <span id="calc-mem-acts" class="text-slate-200">-- MB</span>
              </div>
              <div class="flex justify-between py-1 border-b border-slate-800 font-bold">
                <span class="text-slate-300">Total Training VRAM:</span>
                <span id="calc-mem-total" class="text-cyan-400">-- GB</span>
              </div>
              <div class="flex justify-between py-1">
                <span class="text-slate-400">Inference VRAM:</span>
                <span id="calc-mem-infer" class="text-emerald-400">-- GB</span>
              </div>
            </div>

            <div id="calc-vram-recommendation" class="p-2.5 rounded-lg bg-slate-900 border border-slate-800 text-[11px] text-slate-300">
              Fits comfortably on RTX 3060 6GB with active safety margin.
            </div>
          </div>

          <!-- Preflight Check Result Card -->
          <div id="studio-preflight-card" class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
            <h3 class="text-xs font-bold text-white uppercase tracking-wider flex items-center justify-between">
              <span>🩺 Preflight Status</span>
              <span id="preflight-status-badge" class="text-[10px] font-mono text-slate-500">Not Run</span>
            </h3>
            <div id="preflight-output-text" class="text-xs text-slate-400 font-mono whitespace-pre-wrap leading-relaxed">
              Click "Preflight Check" to instantiate the model, perform a dummy forward pass, and test gradient propagation.
            </div>
          </div>

        </div>
      </div>

      <!-- Network Registry Catalog Table -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
        <div class="flex items-center justify-between">
          <h3 class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
            <span>📚</span> Network Registry Catalog (`networks/registry/`)
          </h3>
          <button onclick="refreshNetworkRegistryTable()" class="text-xs text-cyan-400 hover:text-cyan-300">Refresh Registry</button>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs font-mono">
            <thead class="text-slate-400 border-b border-darkborder">
              <tr>
                <th class="pb-2 font-semibold">Network ID</th>
                <th class="pb-2 font-semibold">Name</th>
                <th class="pb-2 font-semibold">Family</th>
                <th class="pb-2 font-semibold">Layers</th>
                <th class="pb-2 font-semibold">d_model</th>
                <th class="pb-2 font-semibold">Parameters</th>
                <th class="pb-2 font-semibold">Tasks</th>
                <th class="pb-2 font-semibold text-right">Actions</th>
              </tr>
            </thead>
            <tbody id="table-registry-body" class="divide-y divide-slate-800/60">
              <tr><td colspan="8" class="py-4 text-center text-slate-500 italic">Loading network registry...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ================= TAB 3: TRAINING & LOSS ================= -->
    <div id="tab-training" class="hidden space-y-6" style="display: none;">
      <!-- Main Interactive Loss Chart Card -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 class="text-sm font-bold text-white">Interactive Loss Monitor</h2>
            <p class="text-xs text-slate-400">Real-time Train Loss, Validation Loss, and Exponential Moving Average (EMA)</p>
          </div>

          <div class="flex flex-wrap items-center gap-3 text-xs">
            <!-- EMA Smoothing Slider -->
            <div class="flex items-center gap-2 bg-slate-900 px-3 py-1 rounded-lg border border-slate-800">
              <span class="text-slate-400 text-[11px]">EMA Smoothing:</span>
              <input type="range" id="slider-ema-alpha" min="0.01" max="0.30" step="0.01" value="0.08" oninput="updateEmaAlpha(this.value)" class="w-20 accent-cyan-400 cursor-pointer">
              <span id="label-ema-alpha" class="font-mono text-cyan-400 font-semibold">0.08</span>
            </div>

            <!-- Linear vs Logarithmic Toggle -->
            <button onclick="toggleLogScale()" id="btn-scale-toggle" class="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">
              Scale: <span id="label-scale-type" class="font-semibold text-white">Linear</span>
            </button>

            <!-- Reset Zoom -->
            <button onclick="resetChartZoom()" class="px-3 py-1 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition">
              Reset Zoom
            </button>
          </div>
        </div>

        <div class="h-80 w-full relative">
          <canvas id="chart-loss"></canvas>
        </div>
      </div>

      <!-- Auxiliary Telemetry Charts Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <!-- Chart 1: Gradient Norm -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-bold text-white">Gradient Norm</span>
            <span class="text-[10px] text-slate-400 font-mono">Clip: 1.0</span>
          </div>
          <div class="h-44 w-full">
            <canvas id="chart-grad-norm"></canvas>
          </div>
        </div>

        <!-- Chart 2: Learning Rate (Muon + AdamW) -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-bold text-white">Learning Rate</span>
            <span class="text-[10px] text-cyan-400 font-mono">Muon & AdamW</span>
          </div>
          <div class="h-44 w-full">
            <canvas id="chart-lr"></canvas>
          </div>
        </div>

        <!-- Chart 3: Throughput (tok/s) -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-bold text-white">Throughput (tok/s)</span>
            <span class="text-[10px] text-emerald-400 font-mono">Active Speed</span>
          </div>
          <div class="h-44 w-full">
            <canvas id="chart-speed"></canvas>
          </div>
        </div>

        <!-- Chart 4: VRAM Peak -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-4">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-bold text-white">PyTorch Allocated VRAM</span>
            <span class="text-[10px] text-slate-400 font-mono">Max: 5.0 GB</span>
          </div>
          <div class="h-44 w-full">
            <canvas id="chart-vram"></canvas>
          </div>
        </div>
      </div>

      <!-- Learning Rate Scheduler Visualizer & Editor -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <span>📐</span> Learning Rate Scheduler Visualizer
            </h2>
            <p class="text-xs text-slate-400">Exact mathematical PyTorch schedule curve (Warmup-Stable-Decay / Cosine / Linear)</p>
          </div>
          <button onclick="refreshSchedulerCurve()" class="px-3 py-1 rounded-lg bg-cyan-600/20 text-cyan-400 border border-cyan-500/30 text-xs font-semibold hover:bg-cyan-600/30 transition">
            Recalculate Curve
          </button>
        </div>

        <div class="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Schedule Type</label>
            <select id="sched-type" onchange="refreshSchedulerCurve()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200">
              <option value="wsd">Warmup-Stable-Decay (WSD)</option>
              <option value="cosine">Cosine Decay</option>
              <option value="linear">Linear Decay</option>
              <option value="constant">Constant</option>
            </select>
          </div>
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Peak LR</label>
            <input type="number" id="sched-peak-lr" value="0.003" step="0.0005" onchange="refreshSchedulerCurve()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
          </div>
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Warmup Steps</label>
            <input type="number" id="sched-warmup" value="200" step="50" onchange="refreshSchedulerCurve()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
          </div>
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Stable Steps</label>
            <input type="number" id="sched-stable" value="600" step="50" onchange="refreshSchedulerCurve()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
          </div>
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Total Steps</label>
            <input type="number" id="sched-total" value="1000" step="100" onchange="refreshSchedulerCurve()" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
          </div>
        </div>

        <div class="h-56 w-full">
          <canvas id="chart-scheduler-curve"></canvas>
        </div>
      </div>
    </div>

    <!-- ================= TAB 4: DATASETS ================= -->
    <div id="tab-datasets" class="hidden space-y-6" style="display: none;">
      <!-- Drag & Drop Uploader -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <span>📥</span> Dataset Upload & Ingestion
            </h2>
            <p class="text-xs text-slate-400">Upload .txt (pretraining) or .jsonl / .json (instruction tuning & persona)</p>
          </div>
          <span class="text-xs text-slate-400">Strict UTF-8 & structure verification</span>
        </div>

        <div id="drop-zone" class="border-2 border-dashed border-slate-700 hover:border-cyan-500 rounded-xl p-8 flex flex-col items-center justify-center transition cursor-pointer bg-slate-900/40">
          <input type="file" id="file-input" class="hidden" accept=".txt,.jsonl,.json" onchange="handleFileSelected(this.files[0])">
          <div class="w-12 h-12 rounded-full bg-cyan-500/10 flex items-center justify-center text-cyan-400 text-xl mb-2">
            ⬆
          </div>
          <p class="text-sm font-semibold text-white">Drag and drop dataset file here, or click to browse</p>
          <p class="text-xs text-slate-400 mt-1">Accepts UTF-8 encoded .txt, .jsonl, .json</p>
          <div id="upload-progress-container" class="w-full max-w-md mt-4 hidden">
            <div class="flex justify-between text-xs text-slate-400 mb-1">
              <span id="upload-file-name">Uploading...</span>
              <span id="upload-pct">0%</span>
            </div>
            <div class="w-full h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div id="upload-progress-bar" class="h-full bg-cyan-500 transition-all duration-200" style="width: 0%"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Tokenizer Inspector -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <span>🔬</span> Tokenizer Inspector
            </h2>
            <p class="text-xs text-slate-400">Audit exact Byte-Level BPE tokenization, single-digit atomization, and Romanian sedila unification</p>
          </div>
          <button onclick="runTokenizerInspection()" class="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold transition">
            Tokenize Input
          </button>
        </div>

        <textarea id="inspect-input-text" rows="3" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500" placeholder="Tastează sau lipește un text în limba română (ex: Proiectarea modelului RoCanon-48.5M de către Manu Cătălin cu ecuația 2 + 3 = 5)..."></textarea>

        <!-- Inspection Result Output -->
        <div id="inspect-results" class="hidden space-y-3 pt-2">
          <div class="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div class="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">Total Tokens</span>
              <span id="inspect-token-count" class="font-mono font-bold text-cyan-400 text-base">0</span>
            </div>
            <div class="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">Tokens per Word</span>
              <span id="inspect-tokens-per-word" class="font-mono font-bold text-white text-base">0.0</span>
            </div>
            <div class="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">Tokens per Character</span>
              <span id="inspect-tokens-per-char" class="font-mono font-bold text-white text-base">0.0</span>
            </div>
            <div class="bg-slate-900 p-2.5 rounded-lg border border-slate-800">
              <span class="text-slate-400 block text-[10px]">UTF-8 Round-Trip</span>
              <span id="inspect-exact-match" class="font-mono font-bold text-emerald-400 text-base">100% Invariant</span>
            </div>
          </div>

          <div>
            <span class="text-xs font-semibold text-slate-400 block mb-2">Token Sequence Breakdown:</span>
            <div id="inspect-token-pills" class="flex flex-wrap gap-1.5 max-h-48 overflow-y-auto p-3 bg-slate-900 rounded-lg border border-slate-800"></div>
          </div>
        </div>
      </div>

      <!-- Dataset Explorer & Statistics -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <h2 class="text-sm font-bold text-white flex items-center gap-2">
          <span>📊</span> Registered Datasets & Statistical Explorer
        </h2>
        <div id="registered-datasets-list" class="space-y-3">
          <div class="text-slate-500 italic text-xs py-2">Loading registered datasets...</div>
        </div>
      </div>
    </div>

    <!-- ================= TAB 5: EVALUATION LAB ================= -->
    <div id="tab-evaluation" class="hidden space-y-6" style="display: none;">
      <!-- Live Model Inference -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <span>💬</span> Live Model Inference
            </h2>
            <p class="text-xs text-slate-400">Direct generation on compute device without mocks</p>
          </div>
          <button onclick="runLiveInference()" id="btn-generate" class="px-4 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold transition flex items-center gap-1.5 shadow-lg shadow-cyan-600/20">
            <span>✨</span> <span>Generate</span>
          </button>
        </div>

        <textarea id="infer-prompt" rows="3" class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-cyan-500" placeholder="Scrie un prompt pentru RoCanon-48.5M (ex: Explică arhitectura RoCanon și avantajele filtrelor Canon...)..."></textarea>

        <div class="grid grid-cols-2 md:grid-cols-6 gap-3 text-xs">
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Active Checkpoint</label>
            <select id="infer-checkpoint" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono text-[11px]">
              <option value="">Baseline / Active Model</option>
            </select>
          </div>
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Max Tokens</label>
            <input type="number" id="infer-max-tokens" value="128" step="16" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
          </div>
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Temperature</label>
            <input type="number" id="infer-temp" value="0.7" step="0.1" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
          </div>
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Top-P</label>
            <input type="number" id="infer-top-p" value="0.9" step="0.05" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
          </div>
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Top-K</label>
            <input type="number" id="infer-top-k" value="50" step="5" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
          </div>
          <div>
            <label class="text-[10px] text-slate-400 uppercase font-semibold">Repetition Penalty</label>
            <input type="number" id="infer-rep-pen" value="1.1" step="0.05" class="w-full mt-1 bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-slate-200 font-mono">
          </div>
        </div>

        <!-- Output response -->
        <div id="infer-output-container" class="hidden p-4 rounded-lg bg-slate-900 border border-slate-800 space-y-2">
          <div class="flex items-center justify-between text-xs text-slate-400 border-b border-slate-800 pb-1.5">
            <div class="flex items-center gap-2">
              <span>Model Output:</span>
              <span id="infer-device-badge" class="px-1.5 py-0.5 rounded text-[10px] bg-slate-800 font-mono text-cyan-400"></span>
              <span id="infer-checkpoint-badge" class="text-[10px] text-slate-400 font-mono"></span>
            </div>
            <div class="flex items-center gap-3 font-mono">
              <span id="infer-time-label">-- s</span>
              <span id="infer-speed-label" class="text-cyan-400">-- tok/s</span>
            </div>
          </div>
          <div id="infer-response-text" class="text-xs text-slate-200 font-mono whitespace-pre-wrap leading-relaxed"></div>
        </div>
      </div>

      <!-- Side-by-Side Checkpoint Comparison -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <span>⚖️</span> Side-by-Side Checkpoint Comparison
            </h2>
            <p class="text-xs text-slate-400">Evaluates checkpoints sequentially on a single GPU instance to prevent CUDA OOM</p>
          </div>
          <button onclick="runCheckpointComparison()" class="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition">
            Compare Selected
          </button>
        </div>

        <div id="compare-checkpoints-selector" class="flex flex-wrap gap-2 text-xs">
          <span class="text-slate-500 italic">Loading checkpoints...</span>
        </div>

        <div id="compare-results-grid" class="hidden grid grid-cols-1 md:grid-cols-2 gap-4 pt-2"></div>
      </div>

      <!-- Automatic Benchmark Suite -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4">
        <div class="flex items-center justify-between">
          <div>
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <span>🎯</span> Automated Benchmark Lab
            </h2>
            <p class="text-xs text-slate-400">Audits Creator Persona, Hardware Knowledge, Romanian Understanding, Code, and Math</p>
          </div>
          <button onclick="runBenchmarkSuite()" id="btn-run-benchmarks" class="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold transition">
            Run Benchmark Suite
          </button>
        </div>

        <div id="benchmark-results-container" class="space-y-3">
          <div class="text-slate-500 italic text-xs py-2">Click "Run Benchmark Suite" to evaluate the current model weights against standardized test sets.</div>
        </div>
      </div>

      <!-- Historical Benchmark Runs -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-bold text-white flex items-center gap-2">
            <span>📊</span> Benchmark Run History
          </h2>
          <button onclick="refreshBenchmarkHistory()" class="text-xs text-cyan-400 hover:text-cyan-300">Refresh History</button>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="text-slate-400 border-b border-darkborder">
              <tr>
                <th class="pb-2 font-semibold">Category</th>
                <th class="pb-2 font-semibold">Score</th>
                <th class="pb-2 font-semibold">Step</th>
                <th class="pb-2 font-semibold">Checkpoint</th>
                <th class="pb-2 font-semibold">Run ID</th>
                <th class="pb-2 font-semibold">Date</th>
              </tr>
            </thead>
            <tbody id="table-benchmark-history-body" class="divide-y divide-slate-800/60 font-mono">
              <tr><td colspan="6" class="py-4 text-center text-slate-500 italic">No benchmark runs recorded.</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ================= TAB 6: RUNS & CHECKPOINTS ================= -->
    <div id="tab-checkpoints" class="hidden space-y-6" style="display: none;">
      <!-- Runs History -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
        <h2 class="text-sm font-bold text-white flex items-center gap-2">
          <span>📜</span> Training Runs History
        </h2>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="text-slate-400 border-b border-darkborder">
              <tr>
                <th class="pb-2 font-semibold">Run ID</th>
                <th class="pb-2 font-semibold">Status</th>
                <th class="pb-2 font-semibold">Phase</th>
                <th class="pb-2 font-semibold">Total Steps</th>
                <th class="pb-2 font-semibold">Tokens</th>
                <th class="pb-2 font-semibold">Best Val Loss</th>
                <th class="pb-2 font-semibold">Created</th>
              </tr>
            </thead>
            <tbody id="table-runs-body" class="divide-y divide-slate-800/60 font-mono">
              <tr><td colspan="7" class="py-4 text-center text-slate-500 italic">No runs recorded yet.</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Checkpoints Table -->
      <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
        <div class="flex items-center justify-between">
          <h2 class="text-sm font-bold text-white flex items-center gap-2">
            <span>💾</span> Checkpoints on Disk
          </h2>
          <button onclick="refreshCheckpointsTable()" class="text-xs text-cyan-400 hover:text-cyan-300">Refresh List</button>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs">
            <thead class="text-slate-400 border-b border-darkborder">
              <tr>
                <th class="pb-2 font-semibold">Filename</th>
                <th class="pb-2 font-semibold">Step</th>
                <th class="pb-2 font-semibold">Val Loss</th>
                <th class="pb-2 font-semibold">Size</th>
                <th class="pb-2 font-semibold">Modified</th>
                <th class="pb-2 font-semibold text-right">Actions</th>
              </tr>
            </thead>
            <tbody id="table-checkpoints-body" class="divide-y divide-slate-800/60 font-mono">
              <tr><td colspan="6" class="py-4 text-center text-slate-500 italic">Scanning checkpoints...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ================= TAB 7: SETTINGS & TELEMETRY ================= -->
    <div id="tab-settings" class="hidden space-y-6" style="display: none;">
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <!-- Hardware Telemetry Diagnostic Card -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
          <h2 class="text-sm font-bold text-white flex items-center gap-2">
            <span>🖥️</span> NVIDIA NVML Hardware Diagnostic
          </h2>
          <div class="space-y-2 text-xs font-mono">
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Device Name:</span>
              <span id="diag-gpu-name" class="text-white font-semibold">--</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Driver Version:</span>
              <span id="diag-driver-version" class="text-slate-300">--</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Total VRAM:</span>
              <span id="diag-vram-total" class="text-slate-300">-- MB</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Used / Free VRAM:</span>
              <span id="diag-vram-used-free" class="text-cyan-400">-- / -- MB</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">SM / Memory Clock:</span>
              <span id="diag-clocks" class="text-slate-300">-- / -- MHz</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Power Draw / Limit:</span>
              <span id="diag-power" class="text-slate-300">-- / -- W</span>
            </div>
            <div class="flex justify-between py-1">
              <span class="text-slate-400">PyTorch Allocator (Alloc / Res):</span>
              <span id="diag-pt-alloc" class="text-emerald-400">-- / -- MB</span>
            </div>
          </div>
        </div>

        <!-- Architecture Verification Card -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3">
          <h2 class="text-sm font-bold text-white flex items-center gap-2">
            <span>🧬</span> RoCanon-48.5M Architecture Specifications
          </h2>
          <div class="space-y-2 text-xs font-mono">
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Exact Parameters:</span>
              <span class="text-cyan-400 font-bold">48,531,968</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Transformer Layers:</span>
              <span class="text-white">16 Layers</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Model Dimension (d_model):</span>
              <span class="text-white">512</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Attention Heads (Q : KV):</span>
              <span class="text-white">8 Query : 2 Key/Value (4:1 GQA)</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Feed-Forward (d_ff / SwiGLU):</span>
              <span class="text-white">1,280</span>
            </div>
            <div class="flex justify-between py-1 border-b border-slate-800">
              <span class="text-slate-400">Canon Depthwise Convolutions:</span>
              <span class="text-white">Kernel k=4 (A, B, C, D)</span>
            </div>
            <div class="flex justify-between py-1">
              <span class="text-slate-400">Vocabulary & Weight Tying:</span>
              <span class="text-emerald-400">12,288 (Tied In/Out Embeddings)</span>
            </div>
          </div>
        </div>

        <!-- Webhook Notifications Card (Discord & Telegram) -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-4 col-span-1 md:col-span-2">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <span>📢</span> Discord & Telegram Real-Time Webhooks
            </h2>
            <span class="text-xs text-slate-400">Notifies phone or server on Completion, Loss Spikes, & Best Checkpoints</span>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <!-- Discord Setup -->
            <div class="space-y-2 p-3 bg-slate-900/60 rounded-lg border border-slate-800">
              <div class="flex items-center justify-between">
                <span class="font-bold text-indigo-400 flex items-center gap-1.5">
                  <span>💬</span> Discord Webhook URL
                </span>
                <span class="text-[10px] text-slate-500">Channel Integration</span>
              </div>
              <input type="url" id="setting-discord-webhook" placeholder="https://discord.com/api/webhooks/..." class="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 font-mono text-[11px] focus:outline-none focus:border-cyan-500">
            </div>

            <!-- Telegram Setup -->
            <div class="space-y-2 p-3 bg-slate-900/60 rounded-lg border border-slate-800">
              <div class="flex items-center justify-between">
                <span class="font-bold text-cyan-400 flex items-center gap-1.5">
                  <span>✈️</span> Telegram Bot API
                </span>
                <span class="text-[10px] text-slate-500">Direct Push Notifications</span>
              </div>
              <div class="grid grid-cols-2 gap-2">
                <input type="text" id="setting-telegram-token" placeholder="Bot Token (123456:ABC...)" class="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 font-mono text-[11px] focus:outline-none focus:border-cyan-500">
                <input type="text" id="setting-telegram-chat" placeholder="Chat ID (-100... or @user)" class="w-full bg-slate-950 border border-slate-700 rounded px-2.5 py-1.5 text-slate-200 font-mono text-[11px] focus:outline-none focus:border-cyan-500">
              </div>
            </div>
          </div>

          <!-- Controls -->
          <div class="flex flex-wrap items-center justify-between gap-3 pt-1">
            <div class="flex items-center gap-4 text-xs text-slate-400">
              <label class="flex items-center gap-1.5 cursor-pointer">
                <input type="checkbox" id="setting-notify-complete" checked class="accent-cyan-400 rounded">
                <span>Run Completed</span>
              </label>
              <label class="flex items-center gap-1.5 cursor-pointer">
                <input type="checkbox" id="setting-notify-best" checked class="accent-cyan-400 rounded">
                <span>Best Checkpoint</span>
              </label>
              <label class="flex items-center gap-1.5 cursor-pointer">
                <input type="checkbox" id="setting-notify-spikes" checked class="accent-cyan-400 rounded">
                <span>Loss Spikes / OOM</span>
              </label>
            </div>
            <div class="flex items-center gap-2">
              <button onclick="testWebhookNotifications()" id="btn-test-webhook" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 text-xs font-semibold transition">
                <span>🧪 Test Webhook</span>
              </button>
              <button onclick="saveWebhookSettings()" class="px-3.5 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-bold transition shadow-lg shadow-cyan-600/20">
                <span>Save Notifications</span>
              </button>
            </div>
          </div>
          <div id="webhook-test-status" class="hidden text-xs font-mono p-2.5 rounded bg-slate-950 border border-slate-800"></div>
        </div>

        <!-- System Alerts History Table -->
        <div class="bg-darkcard border border-darkborder rounded-xl p-5 space-y-3 col-span-1 md:col-span-2">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-bold text-white flex items-center gap-2">
              <span>🔔</span> System & Training Alerts History
            </h2>
            <button onclick="refreshAlertsTable()" class="text-xs text-cyan-400 hover:text-cyan-300">Refresh Alerts</button>
          </div>
          <div class="overflow-x-auto max-h-64 overflow-y-auto">
            <table class="w-full text-left text-xs">
              <thead class="text-slate-400 border-b border-darkborder sticky top-0 bg-darkcard">
                <tr>
                  <th class="pb-2 font-semibold">Level</th>
                  <th class="pb-2 font-semibold">Category</th>
                  <th class="pb-2 font-semibold">Message</th>
                  <th class="pb-2 font-semibold">Step</th>
                  <th class="pb-2 font-semibold">Time</th>
                </tr>
              </thead>
              <tbody id="table-alerts-body" class="divide-y divide-slate-800/60 font-mono">
                <tr><td colspan="5" class="py-4 text-center text-slate-500 italic">No alerts recorded.</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  </main>

  <!-- ================= JAVASCRIPT LOGIC ================= -->
  <script>
    // Global State
    let ws = null;
    let isLogScale = false;
    let emaAlpha = 0.08;
    let stepHistory = [];
    let rawLossHistory = [];
    let emaLossHistory = [];
    let valLossHistory = [];
    let gradHistory = [];
    let speedHistory = [];
    let vramHistory = [];
    let muonLrHistory = [];
    let adamwLrHistory = [];

    // Training tab charts
    let chartLoss = null;
    let chartGrad = null;
    let chartLr = null;
    let chartSpeed = null;
    let chartVram = null;
    let chartSched = null;

    // Overview tab charts
    let chartLossOverview = null;
    let chartGradOverview = null;
    let chartLrOverview = null;
    let chartSpeedOverview = null;
    let chartVramOverview = null;

    // Active Network Studio Architecture State
    let activeNetworkConfig = {
      id: "rocanon_48m",
      name: "RoCanon-48.5M Baseline",
      architecture_family: "decoder_only",
      vocab_size: 12288,
      d_model: 512,
      n_layers: 16,
      n_q: 8,
      n_kv: 2,
      d_h: 64,
      d_ff: 1280,
      norm_type: "rmsnorm",
      activation: "swiglu",
      pos_encoding: "rope",
      max_seq_len: 2048,
      has_conv1d: true,
      conv_kernel_size: 4,
      qk_norm: true,
      pre_norm: true,
      tie_word_embeddings: true,
      supported_tasks: ["causal_lm"],
      version: "1.0"
    };

    // Tab Switching
    function switchTab(tabId) {
      try {
        const tabs = ['overview', 'network_studio', 'training', 'datasets', 'evaluation', 'checkpoints', 'settings'];
        tabs.forEach(t => {
          const el = document.getElementById('tab-' + t);
          const btn = document.getElementById('tab-btn-' + t);
          if (!el) return;
          if (t === tabId) {
            el.classList.remove('hidden');
            el.style.display = 'block';
            if (btn) {
              btn.classList.add('tab-active');
              btn.classList.remove('text-slate-400');
            }
          } else {
            el.classList.add('hidden');
            el.style.display = 'none';
            if (btn) {
              btn.classList.remove('tab-active');
              btn.classList.add('text-slate-400');
            }
          }
        });

        // Tab-specific initializations protected in try-catch
        try {
          if (tabId === 'training') {
            refreshRegisteredDatasets();
            setTimeout(() => {
              initChartsIfNeeded();
              if (chartLoss) chartLoss.resize();
              if (chartGrad) chartGrad.resize();
              if (chartLr) chartLr.resize();
              if (chartSpeed) chartSpeed.resize();
              if (chartVram) chartVram.resize();
            }, 60);
          } else if (tabId === 'overview') {
            setTimeout(() => {
              initOverviewChartsIfNeeded();
              if (chartLossOverview) chartLossOverview.resize();
              if (chartGradOverview) chartGradOverview.resize();
              if (chartLrOverview) chartLrOverview.resize();
              if (chartSpeedOverview) chartSpeedOverview.resize();
              if (chartVramOverview) chartVramOverview.resize();
            }, 60);
          } else if (tabId === 'network_studio') {
            refreshNetworkStudio();
          } else if (tabId === 'checkpoints') {
            refreshCheckpointsTable();
            refreshRunsTable();
          } else if (tabId === 'datasets') {
            refreshRegisteredDatasets();
          } else if (tabId === 'evaluation') {
            refreshComparisonCheckpoints();
            refreshBenchmarkHistory();
            refreshCheckpointsTable();
          } else if (tabId === 'settings') {
            refreshAlertsTable();
          }
        } catch (initErr) {
          console.warn('Tab initialization warning:', initErr);
        }
      } catch (err) {
        console.error('switchTab error:', err);
      }
    }

    // Chart.js Setup
    const chartCommonOptions = {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      scales: {
        x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', font: { size: 10 } } },
        y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', font: { size: 10 } } }
      },
      plugins: {
        legend: { labels: { color: '#94a3b8', boxWidth: 12, font: { size: 11 } } }
      }
    };

    function initOverviewChartsIfNeeded() {
      if (typeof Chart === 'undefined') return;
      if (chartLossOverview) return;
      const elLoss = document.getElementById('chart-loss-overview');
      if (!elLoss) return;

      try {
        chartLossOverview = new Chart(elLoss.getContext('2d'), {
          type: 'line',
          data: {
            labels: [...stepHistory],
            datasets: [
              { label: 'Train Loss (Raw)', data: [...rawLossHistory], borderColor: 'rgba(56, 189, 248, 0.4)', borderWidth: 1, pointRadius: 0, tension: 0.1 },
              { label: 'EMA Smoothed', data: [...emaLossHistory], borderColor: '#0ea5e9', borderWidth: 2, pointRadius: 0, tension: 0.1 },
              { label: 'Validation Loss', data: [...valLossHistory], borderColor: '#10b981', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#10b981', tension: 0 }
            ]
          },
          options: {
            ...chartCommonOptions,
            scales: {
              ...chartCommonOptions.scales,
              y: {
                ...chartCommonOptions.scales.y,
                type: isLogScale ? 'logarithmic' : 'linear'
              }
            }
          }
        });

        const elGrad = document.getElementById('chart-grad-overview');
        if (elGrad) {
          chartGradOverview = new Chart(elGrad.getContext('2d'), {
            type: 'line',
            data: { labels: [...stepHistory], datasets: [{ label: 'Grad Norm', data: [...gradHistory], borderColor: '#818cf8', borderWidth: 1.5, pointRadius: 0 }] },
            options: chartCommonOptions
          });
        }

        const elLr = document.getElementById('chart-lr-overview');
        if (elLr) {
          chartLrOverview = new Chart(elLr.getContext('2d'), {
            type: 'line',
            data: {
              labels: [...stepHistory],
              datasets: [
                { label: 'Muon LR', data: [...muonLrHistory], borderColor: '#06b6d4', borderWidth: 1.5, pointRadius: 0 },
                { label: 'AdamW LR', data: [...adamwLrHistory], borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0 }
              ]
            },
            options: chartCommonOptions
          });
        }

        const elSpeed = document.getElementById('chart-speed-overview');
        if (elSpeed) {
          chartSpeedOverview = new Chart(elSpeed.getContext('2d'), {
            type: 'line',
            data: { labels: [...stepHistory], datasets: [{ label: 'tok/s', data: [...speedHistory], borderColor: '#10b981', borderWidth: 1.5, pointRadius: 0 }] },
            options: chartCommonOptions
          });
        }

        const elVram = document.getElementById('chart-vram-overview');
        if (elVram) {
          chartVramOverview = new Chart(elVram.getContext('2d'), {
            type: 'line',
            data: { labels: [...stepHistory], datasets: [{ label: 'Allocated (GB)', data: [...vramHistory], borderColor: '#f43f5e', borderWidth: 1.5, pointRadius: 0 }] },
            options: chartCommonOptions
          });
        }
      } catch (err) {
        console.error('Failed to init overview charts:', err);
      }
    }

    function initChartsIfNeeded() {
      initOverviewChartsIfNeeded();
      if (typeof Chart === 'undefined') {
        console.warn('Chart.js not loaded or offline mode active.');
        return;
      }
      if (chartLoss) return;

      const elLoss = document.getElementById('chart-loss');
      if (!elLoss) return;

      // 1. Loss Chart
      chartLoss = new Chart(elLoss.getContext('2d'), {
        type: 'line',
        data: {
          labels: [...stepHistory],
          datasets: [
            { label: 'Train Loss (Raw)', data: [...rawLossHistory], borderColor: 'rgba(56, 189, 248, 0.4)', borderWidth: 1, pointRadius: 0, tension: 0.1 },
            { label: 'EMA Smoothed', data: [...emaLossHistory], borderColor: '#0ea5e9', borderWidth: 2, pointRadius: 0, tension: 0.1 },
            { label: 'Validation Loss', data: [...valLossHistory], borderColor: '#10b981', borderWidth: 2, pointRadius: 3, pointBackgroundColor: '#10b981', tension: 0 }
          ]
        },
        options: {
          ...chartCommonOptions,
          scales: {
            ...chartCommonOptions.scales,
            y: {
              ...chartCommonOptions.scales.y,
              type: isLogScale ? 'logarithmic' : 'linear'
            }
          }
        }
      });

      // 2. Grad Norm Chart
      const elGrad = document.getElementById('chart-grad-norm');
      if (elGrad) {
        chartGrad = new Chart(elGrad.getContext('2d'), {
          type: 'line',
          data: {
            labels: [...stepHistory],
            datasets: [{ label: 'Grad Norm', data: [...gradHistory], borderColor: '#818cf8', borderWidth: 1.5, pointRadius: 0 }]
          },
          options: chartCommonOptions
        });
      }

      // 3. LR Chart
      const elLr = document.getElementById('chart-lr');
      if (elLr) {
        chartLr = new Chart(elLr.getContext('2d'), {
          type: 'line',
          data: {
            labels: [...stepHistory],
            datasets: [
              { label: 'Muon LR', data: [...muonLrHistory], borderColor: '#06b6d4', borderWidth: 1.5, pointRadius: 0 },
              { label: 'AdamW LR', data: [...adamwLrHistory], borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0 }
            ]
          },
          options: chartCommonOptions
        });
      }

      // 4. Throughput Chart
      const elSpeed = document.getElementById('chart-speed');
      if (elSpeed) {
        chartSpeed = new Chart(elSpeed.getContext('2d'), {
          type: 'line',
          data: {
            labels: [...stepHistory],
            datasets: [{ label: 'tok/s', data: [...speedHistory], borderColor: '#10b981', borderWidth: 1.5, pointRadius: 0 }]
          },
          options: chartCommonOptions
        });
      }

      // 5. VRAM Chart
      const elVram = document.getElementById('chart-vram');
      if (elVram) {
        chartVram = new Chart(elVram.getContext('2d'), {
          type: 'line',
          data: {
            labels: [...stepHistory],
            datasets: [{ label: 'Allocated (GB)', data: [...vramHistory], borderColor: '#f43f5e', borderWidth: 1.5, pointRadius: 0 }]
          },
          options: chartCommonOptions
        });
      }

      refreshSchedulerCurve();
    }

    // Toggle Logarithmic Scale
    function toggleLogScale() {
      isLogScale = !isLogScale;
      const scaleText = isLogScale ? 'Logarithmic' : 'Linear';
      const el1 = document.getElementById('label-scale-type');
      if (el1) el1.innerText = scaleText;
      const el2 = document.getElementById('label-scale-type-ov');
      if (el2) el2.innerText = scaleText;
      [chartLoss, chartLossOverview].forEach(c => {
        if (c && c.options && c.options.scales && c.options.scales.y) {
          c.options.scales.y.type = isLogScale ? 'logarithmic' : 'linear';
          c.update();
        }
      });
    }

    // Reset Zoom
    function resetChartZoom() {
      [chartLoss, chartLossOverview].forEach(c => {
        if (c) {
          if (typeof c.resetZoom === 'function') c.resetZoom();
          else c.update();
        }
      });
    }

    // Update EMA smoothing
    function updateEmaAlpha(val) {
      emaAlpha = parseFloat(val);
      const strVal = emaAlpha.toFixed(2);
      const el1 = document.getElementById('label-ema-alpha');
      if (el1) el1.innerText = strVal;
      const el2 = document.getElementById('label-ema-alpha-ov');
      if (el2) el2.innerText = strVal;
      const sl1 = document.getElementById('slider-ema-alpha');
      if (sl1) sl1.value = val;
      const sl2 = document.getElementById('slider-ema-alpha-ov');
      if (sl2) sl2.value = val;
      recalculateEmaSeries();
    }

    function recalculateEmaSeries() {
      let currentEma = null;
      const newEma = [];
      for (let i = 0; i < rawLossHistory.length; i++) {
        const val = rawLossHistory[i];
        if (currentEma === null) currentEma = val;
        else currentEma = (1 - emaAlpha) * currentEma + emaAlpha * val;
        newEma.push(currentEma);
      }
      [chartLoss, chartLossOverview].forEach(c => {
        if (c && c.data && c.data.datasets && c.data.datasets[1]) {
          c.data.datasets[1].data = [...newEma];
          c.update();
        }
      });
    }

    // Scheduler Curve Visualizer
    async function refreshSchedulerCurve() {
      const payload = {
        schedule_type: document.getElementById('sched-type').value,
        peak_lr: parseFloat(document.getElementById('sched-peak-lr').value),
        warmup_steps: parseInt(document.getElementById('sched-warmup').value),
        stable_steps: parseInt(document.getElementById('sched-stable').value),
        total_steps: parseInt(document.getElementById('sched-total').value),
      };
      try {
        const res = await fetch('/api/scheduler/curve', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const points = await res.json();
        const steps = points.map(p => p.step);
        const lrs = points.map(p => p.lr);

        const ctx = document.getElementById('chart-scheduler-curve').getContext('2d');
        if (chartSched) chartSched.destroy();
        chartSched = new Chart(ctx, {
          type: 'line',
          data: {
            labels: steps,
            datasets: [{ label: 'Learning Rate Curve', data: lrs, borderColor: '#06b6d4', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(6, 182, 212, 0.08)' }]
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
              x: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', font: { size: 10 } } },
              y: { grid: { color: '#1e293b' }, ticks: { color: '#64748b', font: { size: 10 } } }
            },
            plugins: { legend: { labels: { color: '#94a3b8' } } }
          }
        });
      } catch (e) {
        console.error('Failed to load scheduler curve:', e);
      }
    }

    // ================= NETWORK STUDIO LOGIC =================

    function switchNetworkMode(mode) {
      const vVis = document.getElementById('studio-view-visual');
      const vJson = document.getElementById('studio-view-json');
      const bVis = document.getElementById('btn-mode-visual');
      const bJson = document.getElementById('btn-mode-json');

      if (mode === 'visual') {
        vVis.classList.remove('hidden');
        vJson.classList.add('hidden');
        bVis.className = 'px-3 py-1 rounded-md bg-cyan-600 text-white font-semibold transition';
        bJson.className = 'px-3 py-1 rounded-md text-slate-400 hover:text-white transition';
      } else {
        vVis.classList.add('hidden');
        vJson.classList.remove('hidden');
        bVis.className = 'px-3 py-1 rounded-md text-slate-400 hover:text-white transition';
        bJson.className = 'px-3 py-1 rounded-md bg-cyan-600 text-white font-semibold transition';
        document.getElementById('studio-json-editor').value = JSON.stringify(activeNetworkConfig, null, 2);
      }
    }

    async function refreshNetworkStudio() {
      await refreshNetworkRegistryTable();
      syncFormToConfig();
      renderComputationGraph();
      await calculateArchitectureEstimates();
    }

    function syncFormToConfig() {
      const c = activeNetworkConfig;
      document.getElementById('studio-active-title').innerText = c.name || c.id;
      document.getElementById('studio-active-family').innerText = (c.architecture_family || 'DECODER_ONLY').toUpperCase();
      document.getElementById('builder-name').value = c.name || c.id;
      document.getElementById('builder-dmodel').value = c.d_model || 512;
      document.getElementById('builder-layers').value = c.n_layers || 8;
      document.getElementById('builder-vocab').value = c.vocab_size || 12288;
      document.getElementById('builder-nq').value = c.n_q || 8;
      document.getElementById('builder-nkv').value = c.n_kv || 2;
      document.getElementById('builder-dff').value = c.d_ff || 1280;
      document.getElementById('builder-seqlen').value = c.max_seq_len || 1024;
      document.getElementById('builder-norm').value = c.norm_type || 'rmsnorm';
      document.getElementById('builder-act').value = c.activation || 'swiglu';
      document.getElementById('builder-pos').value = c.pos_encoding || 'rope';
      document.getElementById('builder-tied').checked = c.tie_word_embeddings !== false;
      document.getElementById('builder-conv').checked = Boolean(c.has_conv1d);
      document.getElementById('studio-json-editor').value = JSON.stringify(c, null, 2);
    }

    function updateArchitectureFromForm() {
      activeNetworkConfig.name = document.getElementById('builder-name').value;
      activeNetworkConfig.d_model = parseInt(document.getElementById('builder-dmodel').value);
      activeNetworkConfig.n_layers = parseInt(document.getElementById('builder-layers').value);
      activeNetworkConfig.vocab_size = parseInt(document.getElementById('builder-vocab').value);
      activeNetworkConfig.n_q = parseInt(document.getElementById('builder-nq').value);
      activeNetworkConfig.n_kv = parseInt(document.getElementById('builder-nkv').value);
      activeNetworkConfig.d_h = Math.floor(activeNetworkConfig.d_model / activeNetworkConfig.n_q);
      activeNetworkConfig.d_ff = parseInt(document.getElementById('builder-dff').value);
      activeNetworkConfig.max_seq_len = parseInt(document.getElementById('builder-seqlen').value);
      activeNetworkConfig.norm_type = document.getElementById('builder-norm').value;
      activeNetworkConfig.activation = document.getElementById('builder-act').value;
      activeNetworkConfig.pos_encoding = document.getElementById('builder-pos').value;
      activeNetworkConfig.tie_word_embeddings = document.getElementById('builder-tied').checked;
      activeNetworkConfig.has_conv1d = document.getElementById('builder-conv').checked;

      document.getElementById('studio-active-title').innerText = activeNetworkConfig.name;
      document.getElementById('studio-json-editor').value = JSON.stringify(activeNetworkConfig, null, 2);
      renderComputationGraph();
      calculateArchitectureEstimates();
    }

    function syncJsonToForm() {
      try {
        const parsed = JSON.parse(document.getElementById('studio-json-editor').value);
        activeNetworkConfig = parsed;
        syncFormToConfig();
        renderComputationGraph();
        calculateArchitectureEstimates();
      } catch (e) {
        // Syntax in progress
      }
    }

    function formatActiveJson() {
      try {
        const parsed = JSON.parse(document.getElementById('studio-json-editor').value);
        document.getElementById('studio-json-editor').value = JSON.stringify(parsed, null, 2);
      } catch (e) {
        alert('Invalid JSON syntax: ' + e);
      }
    }

    async function selectNetworkTemplate(templateId) {
      try {
        const res = await fetch('/api/networks/templates');
        const list = await res.json();
        const tmpl = list.find(t => t.id === templateId);
        if (tmpl) {
          activeNetworkConfig = JSON.parse(JSON.stringify(tmpl));
          syncFormToConfig();
          renderComputationGraph();
          await calculateArchitectureEstimates();
        }
      } catch (e) {
        console.error('Failed to load template:', e);
      }
    }

    async function applyBudgetPreset(name) {
      try {
        const targets = { '1M': 1000000, '5M': 5000000, '15M': 15000000, '25M': 25000000, '48.5M': 48500000, '100M': 100000000 };
        const res = await fetch('/api/networks/budget_recommend', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_params: targets[name] || 15000000 })
        });
        const rec = await res.json();
        activeNetworkConfig.d_model = rec.d_model;
        activeNetworkConfig.n_layers = rec.n_layers;
        activeNetworkConfig.n_q = rec.n_q;
        activeNetworkConfig.n_kv = rec.n_kv;
        activeNetworkConfig.d_h = Math.floor(rec.d_model / rec.n_q);
        activeNetworkConfig.d_ff = rec.d_ff;
        activeNetworkConfig.vocab_size = rec.vocab_size;
        syncFormToConfig();
        renderComputationGraph();
        await calculateArchitectureEstimates();
      } catch (e) {
        console.error('Failed budget preset:', e);
      }
    }

    function renderComputationGraph() {
      const c = activeNetworkConfig;
      const canvas = document.getElementById('studio-graph-canvas');
      const d_model = c.d_model || 512;
      const n_layers = c.n_layers || 8;
      const n_q = c.n_q || 8;
      const n_kv = c.n_kv || 2;
      const d_ff = c.d_ff || 1280;
      const family = c.architecture_family || 'decoder_only';

      document.getElementById('graph-tensor-shape-info').innerText = `Hidden Dimension: [Batch, SeqLen, ${d_model}]`;

      let html = '';

      // 1. Input Embedding Block
      html += `
        <div class="p-3 rounded-lg bg-slate-900 border border-cyan-500/30 flex items-center justify-between text-xs font-mono">
          <div class="flex items-center gap-2">
            <span class="w-6 h-6 rounded bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold">1</span>
            <div>
              <span class="font-bold text-white">${family === 'mlp' || family === 'cnn' ? 'Input Linear Projection' : 'Token & Positional Embedding'}</span>
              <span class="text-[10px] text-slate-400 block">${family === 'mlp' || family === 'cnn' ? 'Features -> ' + d_model : 'Vocab (' + (c.vocab_size || 12288) + ') + ' + (c.pos_encoding || 'ROPE').toUpperCase()}</span>
            </div>
          </div>
          <span class="text-cyan-400 font-semibold">[B, S, ${d_model}]</span>
        </div>
        <div class="flex justify-center text-slate-600 text-xs">↓</div>
      `;

      // 2. Transformer Blocks (Collapsed summary with individual block sample)
      html += `
        <div class="p-3 rounded-lg bg-slate-900 border border-indigo-500/30 space-y-2 text-xs font-mono">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span class="w-6 h-6 rounded bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold">2</span>
              <div>
                <span class="font-bold text-white">${(family || 'decoder_only').toUpperCase()} Backbone (${n_layers} Stacked Blocks)</span>
                <span class="text-[10px] text-slate-400 block">Pre-Norm • ${(c.norm_type || 'RMSNORM').toUpperCase()} • Residual Scaling</span>
              </div>
            </div>
            <span class="text-indigo-400 font-semibold">${n_layers}x Blocks</span>
          </div>

          <!-- Internal Block Operations Expansion -->
          <div class="p-2.5 rounded bg-slate-950/80 border border-slate-800 space-y-1.5 text-[11px] text-slate-300">
            <div class="flex justify-between items-center text-slate-400 border-b border-slate-800 pb-1">
              <span>Block Internal Operations</span>
              <span>Channels: ${d_model}</span>
            </div>
            <div class="flex justify-between items-center text-indigo-300">
              <span>• Attention: Grouped-Query Attention (GQA ${n_q}:${n_kv}, d_h=${c.d_h || Math.floor(d_model / n_q)})</span>
              <span class="text-slate-500">[B, S, ${d_model}]</span>
            </div>
            ${c.has_conv1d ? `<div class="flex justify-between items-center text-rose-300"><span>• Convolutions: Causal Depthwise Conv1D (k=${c.conv_kernel_size || 4})</span><span class="text-slate-500">[B, S, ${d_model}]</span></div>` : ''}
            <div class="flex justify-between items-center text-amber-300">
              <span>• Feed-Forward: ${(c.activation || 'swiglu').toUpperCase()} FFN (Dim: ${d_ff})</span>
              <span class="text-slate-500">[B, S, ${d_model}]</span>
            </div>
          </div>
        </div>
        <div class="flex justify-center text-slate-600 text-xs">↓</div>
      `;

      // 3. Final Normalization
      html += `
        <div class="p-3 rounded-lg bg-slate-900 border border-emerald-500/30 flex items-center justify-between text-xs font-mono">
          <div class="flex items-center gap-2">
            <span class="w-6 h-6 rounded bg-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold">3</span>
            <div>
              <span class="font-bold text-white">Final Normalization Layer</span>
              <span class="text-[10px] text-slate-400 block">${(c.norm_type || 'RMSNORM').toUpperCase()} (Dim: ${d_model})</span>
            </div>
          </div>
          <span class="text-emerald-400 font-semibold">[B, S, ${d_model}]</span>
        </div>
        <div class="flex justify-center text-slate-600 text-xs">↓</div>
      `;

      // 4. Output Head
      html += `
        <div class="p-3 rounded-lg bg-slate-900 border border-cyan-500/30 flex items-center justify-between text-xs font-mono">
          <div class="flex items-center gap-2">
            <span class="w-6 h-6 rounded bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold">4</span>
            <div>
              <span class="font-bold text-white">Output Projection Head</span>
              <span class="text-[10px] text-slate-400 block">${c.tie_word_embeddings ? 'Tied with Token Embedding' : 'Linear Projection'} -> Vocab (${c.vocab_size})</span>
            </div>
          </div>
          <span class="text-cyan-400 font-semibold">[B, S, ${c.vocab_size}]</span>
        </div>
      `;

      canvas.innerHTML = html;
    }

    async function calculateArchitectureEstimates() {
      const payload = {
        config: activeNetworkConfig,
        micro_batch_size: parseInt(document.getElementById('estimator-batch').value || 4),
        seq_len: parseInt(document.getElementById('estimator-seqlen').value || 256),
        dtype: 'bfloat16',
        optimizer: 'muon_adamw'
      };

      try {
        const res = await fetch('/api/networks/estimate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const d = await res.json();
        const p = d.parameters;
        const m = d.memory;

        // Update Parameters card
        document.getElementById('studio-active-params').innerText = `${Number(p.total_parameters).toLocaleString()} Params`;
        document.getElementById('calc-total-params').innerText = Number(p.total_parameters).toLocaleString();
        document.getElementById('calc-non-emb-params').innerText = Number(p.non_embedding_parameters).toLocaleString();
        document.getElementById('calc-emb-params').innerText = `${Number(p.embedding_parameters).toLocaleString()} (${p.tied_embeddings ? 'Tied' : 'Separate'})`;
        document.getElementById('calc-attn-params').innerText = Number(p.attention_parameters).toLocaleString();
        document.getElementById('calc-ffn-params').innerText = Number(p.ffn_parameters).toLocaleString();
        document.getElementById('calc-conv-params').innerText = Number(p.conv_parameters).toLocaleString();

        // Update Memory Estimator card
        document.getElementById('calc-mem-weights').innerText = `${m.model_weights_mb} MB`;
        document.getElementById('calc-mem-grads').innerText = `${m.gradients_mb} MB`;
        document.getElementById('calc-mem-opt').innerText = `${m.optimizer_states_mb} MB`;
        document.getElementById('calc-mem-acts').innerText = `${m.activations_mb} MB`;
        document.getElementById('calc-mem-total').innerText = `${m.total_training_vram_gb} GB`;
        document.getElementById('calc-mem-infer').innerText = `${m.inference_vram_gb} GB`;

        const statusBadge = document.getElementById('calc-vram-status-badge');
        statusBadge.innerText = m.safety_status;
        if (m.safety_status === 'SAFE') {
          statusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30';
        } else if (m.safety_status === 'TIGHT') {
          statusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-amber-500/10 text-amber-400 border border-amber-500/30';
        } else {
          statusBadge.className = 'px-2 py-0.5 rounded text-[10px] font-bold font-mono bg-rose-500/10 text-rose-400 border border-rose-500/30';
        }

        document.getElementById('calc-vram-recommendation').innerText = m.recommendation;
      } catch (e) {
        console.error('Calculation error:', e);
      }
    }

    async function runArchitecturePreflight() {
      const badge = document.getElementById('preflight-status-badge');
      const out = document.getElementById('preflight-output-text');
      badge.innerText = 'TESTING...';
      badge.className = 'text-[10px] font-mono text-cyan-400';
      out.innerText = 'Running CPU shape validation, forward pass, and gradient backprop...';

      try {
        const res = await fetch('/api/networks/validate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(activeNetworkConfig)
        });
        const d = await res.json();
        if (d.valid) {
          badge.innerText = 'VERIFIED';
          badge.className = 'text-[10px] font-mono text-emerald-400 font-bold';
          out.innerText = `SUCCESS: ${d.message}\nOutput Tensor Shape: [${d.output_shape.join(', ')}]\nTotal Verified Parameters: ${Number(d.total_parameters).toLocaleString()}`;
        } else {
          badge.innerText = 'FAILED';
          badge.className = 'text-[10px] font-mono text-rose-400 font-bold';
          out.innerText = `ERROR [${d.stage}]:\n` + (d.errors ? d.errors.join(`\n`) : '');
        }
      } catch (e) {
        badge.innerText = 'ERROR';
        out.innerText = 'Validation failed: ' + e;
      }
    }

    async function saveActiveNetwork() {
      const id = prompt('Enter unique Network ID for registry:', activeNetworkConfig.id || 'custom_model');
      if (!id) return;
      activeNetworkConfig.id = id;
      try {
        const res = await fetch('/api/networks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(activeNetworkConfig)
        });
        const saved = await res.json();
        alert(`Network ${saved.name || saved.id} saved to registry!`);
        refreshNetworkRegistryTable();
        refreshOverviewNetworkOptions();
      } catch (e) {
        alert('Save error: ' + e);
      }
    }

    async function cloneActiveNetwork() {
      const newId = prompt('Enter ID for cloned network:', `${activeNetworkConfig.id}_clone`);
      if (!newId) return;
      const newName = prompt('Enter Name for cloned network:', `Clone of ${activeNetworkConfig.name}`);
      try {
        const res = await fetch(`/api/networks/${activeNetworkConfig.id}/clone`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ new_id: newId, new_name: newName || newId })
        });
        const cloned = await res.json();
        activeNetworkConfig = cloned;
        syncFormToConfig();
        alert(`Cloned network ${cloned.name} successfully!`);
        refreshNetworkRegistryTable();
        refreshOverviewNetworkOptions();
      } catch (e) {
        alert('Clone error: ' + e);
      }
    }

    async function refreshNetworkRegistryTable() {
      try {
        const res = await fetch('/api/networks');
        const list = await res.json();
        const tbody = document.getElementById('table-registry-body');
        if (!list || list.length === 0) {
          tbody.innerHTML = '<tr><td colspan="8" class="py-4 text-center text-slate-500 italic">No networks registered yet.</td></tr>';
          return;
        }

        tbody.innerHTML = '';
        list.forEach(net => {
          const tr = document.createElement('tr');
          tr.className = 'hover:bg-slate-900/60 transition';
          tr.innerHTML = `
            <td class="py-2.5 font-bold text-white">${net.id}</td>
            <td class="py-2.5 text-slate-200">${net.name}</td>
            <td class="py-2.5 text-cyan-400 font-semibold">${net.architecture_family.toUpperCase()}</td>
            <td class="py-2.5 text-slate-300">${net.n_layers}</td>
            <td class="py-2.5 text-slate-300">${net.d_model}</td>
            <td class="py-2.5 text-emerald-400 font-bold">${Number(net.total_parameters).toLocaleString()}</td>
            <td class="py-2.5 text-slate-400 text-[10px]">${(net.supported_tasks || ['causal_lm']).join(', ')}</td>
            <td class="py-2.5 text-right space-x-2">
              <button onclick="loadNetworkFromRegistry('${net.id}')" class="text-cyan-400 hover:text-cyan-300">Edit</button>
              <button onclick="cloneNetworkFromRegistry('${net.id}')" class="text-indigo-400 hover:text-indigo-300">Clone</button>
              ${net.id !== 'rocanon_48m' ? `<button onclick="deleteNetworkFromRegistry('${net.id}')" class="text-rose-400 hover:text-rose-300">Delete</button>` : ''}
            </td>
          `;
          tbody.appendChild(tr);
        });
        refreshOverviewNetworkOptions();
      } catch (e) {
        console.error('Failed to load network registry:', e);
      }
    }

    async function loadNetworkFromRegistry(id) {
      try {
        const res = await fetch(`/api/networks/${id}`);
        const net = await res.json();
        activeNetworkConfig = net;
        syncFormToConfig();
        renderComputationGraph();
        await calculateArchitectureEstimates();
        alert(`Loaded network ${net.name} into editor!`);
      } catch (e) {
        alert('Load error: ' + e);
      }
    }

    async function cloneNetworkFromRegistry(id) {
      await loadNetworkFromRegistry(id);
      await cloneActiveNetwork();
    }

    async function deleteNetworkFromRegistry(id) {
      if (!confirm(`Are you sure you want to delete network ${id} from registry?`)) return;
      try {
        await fetch(`/api/networks/${id}`, { method: 'DELETE' });
        refreshNetworkRegistryTable();
      } catch (e) {
        alert('Delete error: ' + e);
      }
    }

    async function refreshOverviewNetworkOptions() {
      try {
        const res = await fetch('/api/networks');
        const list = await res.json();
        const sel = document.getElementById('config-network');
        if (!sel) return;
        const currentVal = sel.value;
        sel.innerHTML = '';
        list.forEach(net => {
          const opt = document.createElement('option');
          opt.value = net.id;
          opt.innerText = `${net.name} (${Number(net.total_parameters).toLocaleString()}p - ${net.architecture_family.toUpperCase()})`;
          sel.appendChild(opt);
        });
        if (currentVal && list.find(n => n.id === currentVal)) {
          sel.value = currentVal;
        }
      } catch (e) {
        console.error('Failed to refresh overview network options:', e);
      }
    }

    // ================= WEBSOCKET FEED =================

    function connectWebSocket() {
      const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${proto}//${window.location.host}/ws`;
      ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        document.getElementById('badge-connection').className = 'px-2 py-0.5 text-[11px] font-semibold rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
        document.getElementById('badge-connection').innerText = 'LIVE WS';
      };

      ws.onclose = () => {
        document.getElementById('badge-connection').className = 'px-2 py-0.5 text-[11px] font-semibold rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20';
        document.getElementById('badge-connection').innerText = 'OFFLINE';
        setTimeout(connectWebSocket, 2000);
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          handleIncomingMessage(msg);
        } catch (e) {
          console.error('WS Parse Error:', e);
        }
      };
    }

    let lastChartStep = -1;

    function updateRunStateUI(s) {
      if (!s) return;
      const elState = document.getElementById('status-run-state');
      if (elState) elState.innerText = s;
      const btnStartText = document.getElementById('start-btn-text');
      const btnStart = document.getElementById('btn-start');
      const pauseText = document.getElementById('pause-text');
      const elBadge = document.getElementById('pipeline-active-badge');

      if (s === 'RUNNING') {
        if (btnStartText) btnStartText.innerText = 'Running';
        if (btnStart) btnStart.disabled = true;
        if (pauseText) pauseText.innerText = 'Pause';
        if (elBadge) {
          elBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20';
          elBadge.innerText = 'TRAINING LIVE';
        }
      } else if (s === 'STARTING') {
        if (btnStartText) btnStartText.innerText = 'Starting...';
        if (btnStart) btnStart.disabled = true;
        if (elBadge) {
          elBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/20';
          elBadge.innerText = 'TOKENIZING...';
        }
      } else if (s === 'PAUSED') {
        if (pauseText) pauseText.innerText = 'Resume';
        if (elBadge) {
          elBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/10 text-amber-400 border border-amber-500/20';
          elBadge.innerText = 'PAUSED';
        }
      } else if (s === 'STOPPING') {
        const btnStopText = document.getElementById('stop-btn-text');
        if (btnStopText) btnStopText.innerText = 'Stopping...';
        if (btnStart) btnStart.disabled = true;
        if (elBadge) {
          elBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-500/10 text-rose-300 border border-rose-500/20 animate-pulse';
          elBadge.innerText = 'STOPPING...';
        }
      } else if (s === 'STOPPED' || s === 'IDLE' || s === 'COMPLETED') {
        if (btnStartText) btnStartText.innerText = 'Start';
        if (btnStart) btnStart.disabled = false;
        const btnStopText = document.getElementById('stop-btn-text');
        if (btnStopText) btnStopText.innerText = 'Stop';
        const btnStop = document.getElementById('btn-stop');
        if (btnStop) btnStop.disabled = false;
        if (pauseText) pauseText.innerText = 'Pause';
        if (elBadge) {
          elBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20';
          elBadge.innerText = s === 'IDLE' ? 'READY' : s;
        }
        const elLog = document.getElementById('pipeline-live-log');
        if (elLog && elLog.innerText.includes('Stop requested')) {
          elLog.innerText = 'Training session stopped cleanly. System ready.';
        }
      }
    }

    function updateMetricsUI(m) {
      if (!m || typeof m !== 'object') return;

      if (m.step !== undefined && m.step !== null) {
        const el = document.getElementById('status-step');
        if (el) el.innerText = m.step;
      }
      if (m.tokens !== undefined && m.tokens !== null) {
        const el = document.getElementById('status-tokens');
        if (el) el.innerText = Number(m.tokens).toLocaleString();
      }
      if (m.train_loss !== undefined && m.train_loss !== null) {
        const el = document.getElementById('kpi-train-loss');
        if (el) el.innerText = Number(m.train_loss).toFixed(4);
      }
      if (m.grad_norm !== undefined && m.grad_norm !== null) {
        const el = document.getElementById('kpi-grad-norm');
        if (el) el.innerText = Number(m.grad_norm).toFixed(2);
      }
      if (m.tok_per_sec !== undefined && m.tok_per_sec !== null) {
        const el = document.getElementById('kpi-tok-per-sec');
        if (el) el.innerText = Math.round(Number(m.tok_per_sec));
      }
      if (m.tflops !== undefined && m.tflops !== null) {
        const el = document.getElementById('kpi-tflops');
        if (el) el.innerText = Number(m.tflops).toFixed(2) + ' TFLOPS';
      }
      if (m.vram_used_gb !== undefined && m.vram_used_gb !== null) {
        const el = document.getElementById('kpi-vram-pt');
        if (el) el.innerText = 'PT: ' + Number(m.vram_used_gb).toFixed(2) + ' GB';
      }
      if (m.val_loss !== null && m.val_loss !== undefined) {
        const el = document.getElementById('kpi-val-loss');
        if (el) el.innerText = Number(m.val_loss).toFixed(4);
        if (m.generalization_gap !== null && m.generalization_gap !== undefined) {
          const elGap = document.getElementById('kpi-gen-gap');
          if (elGap) elGap.innerText = 'Gap: ' + Number(m.generalization_gap).toFixed(4);
        }
      }

      // Update Live Tokenization & Pipeline Monitor elements
      if (m.dataset_name) {
        const elDs = document.getElementById('pipeline-dataset-name');
        if (elDs) elDs.innerText = m.dataset_name;
      }
      if (m.phase) {
        const elPh = document.getElementById('pipeline-phase-name');
        if (elPh) elPh.innerText = m.phase;
        const elStPh = document.getElementById('status-phase');
        if (elStPh) elStPh.innerText = m.phase;
      }
      if (m.dataset_status) {
        const elLog = document.getElementById('pipeline-live-log');
        if (elLog) elLog.innerText = m.dataset_status;
        const elTokTxt = document.getElementById('tok-status-text');
        if (elTokTxt) elTokTxt.innerText = m.dataset_status;
      }
      if (m.dataset_progress_pct !== undefined && m.dataset_progress_pct !== null) {
        const pct = Math.min(100, Math.max(0, Number(m.dataset_progress_pct)));
        const elTokPct = document.getElementById('tok-progress-pct');
        if (elTokPct) elTokPct.innerText = pct.toFixed(1) + '%';
        const elTokBar = document.getElementById('tok-progress-bar');
        if (elTokBar) elTokBar.style.width = pct + '%';
        const elTokBadge = document.getElementById('tok-cache-badge');
        if (elTokBadge) {
          if (pct >= 100) {
            elTokBadge.className = 'text-emerald-400 font-mono text-[10px]';
            elTokBadge.innerText = 'DISK CACHED';
          } else {
            elTokBadge.className = 'text-amber-400 font-mono text-[10px] animate-pulse';
            elTokBadge.innerText = 'TOKENIZING ' + Math.round(pct) + '%';
          }
        }
      }
      if (m.epoch !== undefined && m.epoch !== null) {
        const epNum = Math.floor(Number(m.epoch)) + 1;
        const elEpDet = document.getElementById('epoch-details-text');
        if (elEpDet) elEpDet.innerText = 'Epoch ' + epNum;
      }
      if (m.epoch_pct !== undefined && m.epoch_pct !== null) {
        const epPct = Math.min(100, Math.max(0, Number(m.epoch_pct)));
        const elEpPct = document.getElementById('epoch-progress-pct');
        if (elEpPct) elEpPct.innerText = epPct.toFixed(1) + '%';
        const elEpBar = document.getElementById('epoch-progress-bar');
        if (elEpBar) elEpBar.style.width = epPct + '%';
      }
      if (m.tokens !== undefined && (m.dataset_tokens || m.train_tokens)) {
        const totalDsTokens = m.dataset_tokens || m.train_tokens;
        const elEpTok = document.getElementById('epoch-tokens-text');
        const curr = (Number(m.tokens) % Number(totalDsTokens));
        if (elEpTok) elEpTok.innerText = Number(curr).toLocaleString() + ' / ' + (Number(totalDsTokens) / 1000000).toFixed(1) + 'M tok';
      }
      if (m.target_steps) {
        const stepNum = Number(m.step || 0);
        const tgtSteps = Number(m.target_steps);
        const sPct = (m.step_pct !== undefined && m.step_pct !== null)
          ? Number(m.step_pct)
          : Math.min(100, Math.max(0, (stepNum / tgtSteps) * 100));
        const elSPct = document.getElementById('step-progress-pct');
        if (elSPct) elSPct.innerText = sPct.toFixed(1) + '%';
        const elSBar = document.getElementById('step-progress-bar');
        if (elSBar) elSBar.style.width = sPct + '%';
        const elSCnt = document.getElementById('step-counter-text');
        if (elSCnt) {
          if (m.start_step && m.session_target_steps) {
            const sessStep = m.session_step !== undefined ? m.session_step : (stepNum - m.start_step);
            elSCnt.innerText = `Step ${sessStep} / ${m.session_target_steps} (Global ${stepNum}/${tgtSteps})`;
          } else {
            elSCnt.innerText = 'Step ' + stepNum + ' / ' + tgtSteps.toLocaleString();
          }
        }
      }
      if (m.eta_seconds !== undefined && m.eta_seconds !== null) {
        const s = Number(m.eta_seconds);
        const hrs = Math.floor(s / 3600);
        const mins = Math.floor((s % 3600) / 60);
        const secs = s % 60;
        const pad = (n) => String(n).padStart(2, '0');
        const elEta = document.getElementById('pipeline-eta');
        if (elEta) elEta.innerText = pad(hrs) + ':' + pad(mins) + ':' + pad(secs);
      }
      if (m.tok_per_sec !== undefined) {
        const elSpd = document.getElementById('speed-indicator-text');
        if (elSpd) elSpd.innerText = Math.round(Number(m.tok_per_sec)) + ' tok/s';
      }

      if (m.early_stopping_enabled !== undefined) {
        const elEs = document.getElementById('pipeline-es-status');
        if (elEs) {
          if (!m.early_stopping_enabled) {
            elEs.innerText = 'Disabled';
            elEs.className = 'font-mono text-slate-400 font-semibold';
          } else {
            const cnt = m.early_stopping_counter !== undefined ? m.early_stopping_counter : 0;
            const pat = m.early_stopping_patience || 5;
            elEs.innerText = cnt > 0 ? ('Patience: ' + cnt + '/' + pat) : ('Armed (0/' + pat + ')');
            elEs.className = cnt > 0 ? 'font-mono text-amber-400 font-semibold' : 'font-mono text-emerald-400 font-semibold';
          }
        }
      }

      if (m.muon_lr !== undefined && m.muon_lr !== null) {
        const elMuon = document.getElementById('label-live-muon');
        if (elMuon) elMuon.innerText = Number(m.muon_lr).toExponential(3);
      }
      if (m.adamw_lr !== undefined && m.adamw_lr !== null) {
        const elAdamw = document.getElementById('label-live-adamw');
        if (elAdamw) elAdamw.innerText = Number(m.adamw_lr).toExponential(3);
      }

      // Charts update
      if (m.step !== undefined && m.step !== null && m.step !== lastChartStep && m.train_loss !== undefined && m.train_loss !== null) {
        lastChartStep = m.step;
        const trainLossNum = Number(m.train_loss);
        const valLossVal = (m.val_loss !== undefined && m.val_loss !== null) ? Number(m.val_loss) : null;

        stepHistory.push(m.step);
        rawLossHistory.push(trainLossNum);
        const prevEma = emaLossHistory.length > 0 ? emaLossHistory[emaLossHistory.length - 1] : trainLossNum;
        const currEma = (1 - emaAlpha) * prevEma + emaAlpha * trainLossNum;
        emaLossHistory.push(currEma);
        valLossHistory.push(valLossVal);

        if (stepHistory.length > 500) {
          stepHistory.shift();
          rawLossHistory.shift();
          emaLossHistory.shift();
          valLossHistory.shift();
        }

        [chartLoss, chartLossOverview].forEach(c => {
          if (!c) return;
          c.data.labels.push(m.step);
          c.data.datasets[0].data.push(trainLossNum);
          c.data.datasets[1].data.push(currEma);
          c.data.datasets[2].data.push(valLossVal);
          if (c.data.labels.length > 500) {
            c.data.labels.shift();
            c.data.datasets[0].data.shift();
            c.data.datasets[1].data.shift();
            c.data.datasets[2].data.shift();
          }
          c.update();
        });

        if (m.grad_norm !== undefined && m.grad_norm !== null) {
          const gn = Number(m.grad_norm);
          gradHistory.push(gn);
          if (gradHistory.length > 100) gradHistory.shift();
          [chartGrad, chartGradOverview].forEach(c => {
            if (!c) return;
            c.data.labels.push(m.step);
            c.data.datasets[0].data.push(gn);
            if (c.data.labels.length > 100) { c.data.labels.shift(); c.data.datasets[0].data.shift(); }
            c.update();
          });
        }

        if (m.muon_lr !== undefined && m.adamw_lr !== undefined) {
          const mlr = Number(m.muon_lr);
          const alr = Number(m.adamw_lr);
          muonLrHistory.push(mlr);
          adamwLrHistory.push(alr);
          if (muonLrHistory.length > 100) { muonLrHistory.shift(); adamwLrHistory.shift(); }
          [chartLr, chartLrOverview].forEach(c => {
            if (!c) return;
            c.data.labels.push(m.step);
            c.data.datasets[0].data.push(mlr);
            c.data.datasets[1].data.push(alr);
            if (c.data.labels.length > 100) { c.data.labels.shift(); c.data.datasets[0].data.shift(); }
            c.update();
          });
        }

        if (m.tok_per_sec !== undefined && m.tok_per_sec !== null) {
          const spd = Number(m.tok_per_sec);
          speedHistory.push(spd);
          if (speedHistory.length > 100) speedHistory.shift();
          [chartSpeed, chartSpeedOverview].forEach(c => {
            if (!c) return;
            c.data.labels.push(m.step);
            c.data.datasets[0].data.push(spd);
            if (c.data.labels.length > 100) { c.data.labels.shift(); c.data.datasets[0].data.shift(); }
            c.update();
          });
        }

        if (m.vram_used_gb !== undefined && m.vram_used_gb !== null) {
          const vr = Number(m.vram_used_gb);
          vramHistory.push(vr);
          if (vramHistory.length > 100) vramHistory.shift();
          [chartVram, chartVramOverview].forEach(c => {
            if (!c) return;
            c.data.labels.push(m.step);
            c.data.datasets[0].data.push(vr);
            if (c.data.labels.length > 100) { c.data.labels.shift(); c.data.datasets[0].data.shift(); }
            c.update();
          });
        }
      }
    }

    function handleIncomingMessage(msg) {
      if (msg.event_type === 'STEP_COMPLETED' || msg.event_type === 'METRICS_UPDATED' || msg.event_type === 'RUN_STARTED') {
        updateMetricsUI(msg.payload);
      } else if (msg.event_type === 'STATE_CHANGED') {
        const s = msg.payload.new_state;
        updateRunStateUI(s);
      } else if (msg.event_type === 'LOSS_SPIKE' || msg.event_type === 'GRADIENT_WARNING') {
        addAlertBadge(msg.event_type, msg.payload.loss || msg.payload.alert || 'Spike detected', msg.step);
      }
    }

    // Dynamic Learning Rate Live Controls
    let currentLiveLRMultiplier = 1.0;

    async function applyLiveLR(multiplier) {
      if (multiplier !== undefined && multiplier !== null) {
        currentLiveLRMultiplier = multiplier;
      }
      const muonEl = document.getElementById('config-muon-lr');
      const adamwEl = document.getElementById('config-adamw-lr');
      const autoEl = document.getElementById('config-auto-lr');
      const payload = {
        muon_lr: muonEl ? parseFloat(muonEl.value) : 0.003,
        adamw_lr: adamwEl ? parseFloat(adamwEl.value) : 0.0006,
        multiplier: currentLiveLRMultiplier,
        auto_adjust: autoEl ? autoEl.checked : true,
      };
      try {
        const res = await fetch('/api/runs/lr', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const lblM = document.getElementById('label-live-muon');
        if (lblM && payload.muon_lr) lblM.innerText = (payload.muon_lr * currentLiveLRMultiplier).toExponential(3);
        const lblA = document.getElementById('label-live-adamw');
        if (lblA && payload.adamw_lr) lblA.innerText = (payload.adamw_lr * currentLiveLRMultiplier).toExponential(3);
      } catch (e) {
        console.warn('Live LR update failed:', e);
      }
    }

    async function adjustLiveLRMultiplier(mult) {
      await applyLiveLR(mult);
    }

    // Telemetry polling
    async function pollTelemetry() {
      // 1. Hardware telemetry
      try {
        const res = await fetch('/api/hardware');
        const d = await res.json();
        if (d.is_available) {
          if (d.free_vram_mb !== null && d.free_vram_mb !== undefined) {
            const elVram = document.getElementById('kpi-free-vram');
            if (elVram) elVram.innerText = Math.round(d.free_vram_mb);
            const pill = document.getElementById('vram-pill');
            const txt = document.getElementById('vram-status-text');
            if (pill && txt) {
              if (d.free_vram_mb < 400) {
                pill.className = 'flex items-center gap-2 px-3 py-1.5 rounded-lg border border-rose-500/30 bg-rose-500/10 text-rose-300 text-xs font-semibold';
                txt.innerText = 'VRAM: EMERGENCY';
              } else if (d.free_vram_mb < 700) {
                pill.className = 'flex items-center gap-2 px-3 py-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-300 text-xs font-semibold';
                txt.innerText = 'VRAM: Critical';
              } else if (d.free_vram_mb < 1000) {
                pill.className = 'flex items-center gap-2 px-3 py-1.5 rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-300 text-xs font-semibold';
                txt.innerText = 'VRAM: Warning';
              } else {
                pill.className = 'flex items-center gap-2 px-3 py-1.5 rounded-lg border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 text-xs font-semibold';
                txt.innerText = 'VRAM: ' + Math.round(d.free_vram_mb) + ' MB Free';
              }
            }
          }
          if (d.temperature_c !== null && d.temperature_c !== undefined) {
            const el = document.getElementById('kpi-gpu-temp');
            if (el) el.innerText = d.temperature_c + '°C';
          }
          if (d.gpu_util_percent !== null && d.gpu_util_percent !== undefined) {
            const el = document.getElementById('kpi-gpu-util');
            if (el) el.innerText = '(' + d.gpu_util_percent + '%)';
          }
          if (d.power_draw_watts !== null && d.power_draw_watts !== undefined) {
            const el = document.getElementById('kpi-gpu-power');
            if (el) el.innerText = d.power_draw_watts + ' W';
          }

          // Diagnostic tab fields
          const elName = document.getElementById('diag-gpu-name');
          if (elName) elName.innerText = d.device_name || 'N/A';
          const elDrv = document.getElementById('diag-driver-version');
          if (elDrv) elDrv.innerText = d.driver_version || 'N/A';
          const elTot = document.getElementById('diag-vram-total');
          if (elTot) elTot.innerText = d.total_vram_mb ? (d.total_vram_mb + ' MB') : 'N/A';
          const elUF = document.getElementById('diag-vram-used-free');
          if (elUF) elUF.innerText = (d.used_vram_mb || 0) + ' / ' + (d.free_vram_mb || 0) + ' MB';
          const elClk = document.getElementById('diag-clocks');
          if (elClk) elClk.innerText = (d.sm_clock_mhz || '--') + ' / ' + (d.memory_clock_mhz || '--') + ' MHz';
          const elPwr = document.getElementById('diag-power');
          if (elPwr) elPwr.innerText = (d.power_draw_watts || '--') + ' / ' + (d.power_limit_watts || '--') + ' W';
          const elPt = document.getElementById('diag-pt-alloc');
          if (elPt) elPt.innerText = (d.pytorch_allocated_mb || 0) + ' / ' + (d.pytorch_reserved_mb || 0) + ' MB';
        }
      } catch (e) {
        console.error('Telemetry Poll Error:', e);
      }

      // 2. System Status & Training Metrics polling fallback
      try {
        const sysRes = await fetch('/api/system/status');
        const sys = await sysRes.json();
        if (sys) {
          if (sys.state) {
            updateRunStateUI(sys.state);
          }
          if (sys.latest_metrics && Object.keys(sys.latest_metrics).length > 0) {
            updateMetricsUI(sys.latest_metrics);
          }
        }
      } catch (e) {
        console.error('System Status Poll Error:', e);
      }
    }

    // Alerts
    let alertCount = 0;
    function addAlertBadge(type, message, step) {
      alertCount++;
      document.getElementById('alerts-count-badge').innerText = alertCount + ' Alerts';
      const container = document.getElementById('alerts-container');
      const div = document.createElement('div');
      div.className = 'p-2 rounded bg-rose-500/10 border border-rose-500/30 text-rose-300 flex justify-between items-center';
      div.innerHTML = '<span><strong>Step ' + step + ':</strong> ' + message + '</span><span class="text-[10px] text-slate-400 font-mono">' + type + '</span>';
      container.prepend(div);
    }
    function clearAlerts() {
      alertCount = 0;
      document.getElementById('alerts-count-badge').innerText = '0 Alerts';
      document.getElementById('alerts-container').innerHTML = '<div class="text-slate-500 italic text-center py-4">No critical alerts detected. System healthy.</div>';
    }

    function setMicroBatch(n) {
      const el = document.getElementById('config-batch-size');
      if (el) el.value = n;
    }

    function setTargetSteps(n) {
      const el = document.getElementById('config-target-steps');
      if (el) {
        el.value = n;
        onTargetStepsChange(n);
      }
    }

    function onTargetStepsChange(val) {
      const steps = parseInt(val, 10) || 1000;
      const elSCnt = document.getElementById('step-counter-text');
      const stepEl = document.getElementById('status-step');
      const currStep = stepEl ? stepEl.innerText : '0';
      if (elSCnt) elSCnt.innerText = 'Step ' + currStep + ' / ' + steps.toLocaleString();
    }

    // Actions
    async function actionStartRun() {
      const btnText = document.getElementById('start-btn-text');
      const origText = btnText ? btnText.innerText : 'Start';
      if (btnText) btnText.innerText = 'Starting...';

      const elBadge = document.getElementById('pipeline-active-badge');
      if (elBadge) {
        elBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/20';
        elBadge.innerText = 'STARTING...';
      }
      const elLog = document.getElementById('pipeline-live-log');
      if (elLog) elLog.innerText = 'Initializing training worker and verifying dataset tokenization cache...';
      const elTokBar = document.getElementById('tok-progress-bar');
      if (elTokBar) elTokBar.style.width = '25%';
      const elTokPct = document.getElementById('tok-progress-pct');
      if (elTokPct) elTokPct.innerText = '25%';
      const elTokTxt = document.getElementById('tok-status-text');
      if (elTokTxt) elTokTxt.innerText = 'Verifying pre-tokenized cache on disk...';

      const netEl = document.getElementById('config-network');
      const phaseEl = document.getElementById('config-phase');
      const dsEl = document.getElementById('config-dataset');
      const chkEl = document.getElementById('config-checkpoint');
      const stepsEl = document.getElementById('config-target-steps');
      const evalEl = document.getElementById('config-eval-freq');
      const saveEl = document.getElementById('config-save-freq');
      const esEnabledEl = document.getElementById('config-es-enabled');
      const esPatienceEl = document.getElementById('config-es-patience');
      const esDeltaEl = document.getElementById('config-es-min-delta');
      const batchEl = document.getElementById('config-batch-size');
      const accumEl = document.getElementById('config-grad-accum');
      const muonEl = document.getElementById('config-muon-lr');
      const adamwEl = document.getElementById('config-adamw-lr');

      const payload = {
        network_id: (netEl && netEl.value) ? netEl.value : 'rocanon_48m',
        phase: (phaseEl && phaseEl.value) ? phaseEl.value : '1. General Pretraining',
        dataset_source: (dsEl && dsEl.value) ? dsEl.value : 'tech_wiki_ro',
        checkpoint_to_load: (chkEl && chkEl.value) ? chkEl.value : null,
        target_steps: (stepsEl && stepsEl.value) ? parseInt(stepsEl.value, 10) : 1000,
        eval_frequency: (evalEl && evalEl.value) ? parseInt(evalEl.value, 10) : 25,
        save_frequency: (saveEl && saveEl.value) ? parseInt(saveEl.value, 10) : 50,
        early_stopping_enabled: esEnabledEl ? esEnabledEl.checked : false,
        early_stopping_patience: (esPatienceEl && esPatienceEl.value) ? parseInt(esPatienceEl.value, 10) : 20,
        early_stopping_min_delta: (esDeltaEl && esDeltaEl.value) ? parseFloat(esDeltaEl.value) : 0.001,
        micro_batch_size: (batchEl && batchEl.value) ? parseInt(batchEl.value, 10) : 8,
        grad_accum_steps: (accumEl && accumEl.value) ? parseInt(accumEl.value, 10) : 2,
        muon_lr: (muonEl && muonEl.value) ? parseFloat(muonEl.value) : 0.003,
        adamw_lr: (adamwEl && adamwEl.value) ? parseFloat(adamwEl.value) : 0.0006,
        device: 'cuda',
      };

      try {
        const res = await fetch('/api/runs', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const d = await res.json();
        if (!res.ok) {
          alert('Could not start run: ' + (d.error || res.statusText));
          if (btnText) btnText.innerText = origText;
        } else {
          if (btnText) btnText.innerText = 'Running';
        }
      } catch (err) {
        alert('Network error starting run: ' + err);
        if (btnText) btnText.innerText = origText;
      }
    }

    async function actionPause() {
      const s = document.getElementById('status-run-state').innerText;
      if (s === 'PAUSED') {
        await fetch('/api/action', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'resume' }) });
      } else {
        await fetch('/api/action', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'pause' }) });
      }
    }

    async function actionStop() {
      const btnStopText = document.getElementById('stop-btn-text');
      if (btnStopText) btnStopText.innerText = 'Stopping...';
      const btnStop = document.getElementById('btn-stop');
      if (btnStop) btnStop.disabled = true;

      const elBadge = document.getElementById('pipeline-active-badge');
      if (elBadge) {
        elBadge.className = 'px-2 py-0.5 rounded text-[10px] font-semibold bg-rose-500/10 text-rose-300 border border-rose-500/20 animate-pulse';
        elBadge.innerText = 'STOPPING...';
      }
      const elLog = document.getElementById('pipeline-live-log');
      if (elLog) elLog.innerText = 'Stop requested. Halting worker and saving safe checkpoint...';

      try {
        await fetch('/api/action', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'stop' })
        });
        await fetch('/api/runs/active/stop', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({})
        });
      } catch (e) {
        console.error('Stop error:', e);
      }
    }

    async function actionSnapshot() {
      await fetch('/api/action', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'manual_checkpoint' }) });
    }

    async function actionValidate() {
      await fetch('/api/action', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'trigger_val' }) });
    }

    async function updateLrMultiplier(val) {
      document.getElementById('label-lr-mult').innerText = parseFloat(val).toFixed(2) + 'x';
      await fetch('/api/action', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action: 'set_lr_multiplier', multiplier: parseFloat(val) })
      });
    }

    // Tokenizer Inspector
    async function runTokenizerInspection() {
      const text = document.getElementById('inspect-input-text').value;
      if (!text.trim()) return;
      try {
        const res = await fetch('/api/tokenizer/inspect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text })
        });
        const d = await res.json();
        document.getElementById('inspect-results').classList.remove('hidden');
        document.getElementById('inspect-token-count').innerText = d.token_count;
        document.getElementById('inspect-tokens-per-word').innerText = d.tokens_per_word;
        document.getElementById('inspect-tokens-per-char').innerText = d.tokens_per_char;
        document.getElementById('inspect-exact-match').innerText = d.exact_match ? '100% Invariant' : 'Mismatch';

        const container = document.getElementById('inspect-token-pills');
        container.innerHTML = '';
        d.tokens.forEach(tok => {
          const pill = document.createElement('span');
          pill.className = tok.is_special
            ? 'px-2 py-1 rounded bg-amber-500/20 text-amber-300 font-mono text-[11px] border border-amber-500/40 flex items-center gap-1'
            : 'px-2 py-1 rounded bg-slate-800 text-cyan-300 font-mono text-[11px] border border-slate-700 flex items-center gap-1';
          pill.innerHTML = `<span class="text-white">${escapeHtml(tok.token)}</span><span class="text-[9px] text-slate-500">#${tok.id}</span>`;
          container.appendChild(pill);
        });
      } catch (e) {
        console.error('Tokenizer inspect error:', e);
      }
    }

    function escapeHtml(str) {
      return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    // Live Inference
    async function runLiveInference() {
      const prompt = document.getElementById('infer-prompt').value;
      if (!prompt.trim()) return;

      const btn = document.getElementById('btn-generate');
      btn.disabled = true;
      btn.innerText = 'Generating...';

      const chk = document.getElementById('infer-checkpoint') ? document.getElementById('infer-checkpoint').value : '';
      const payload = {
        prompt: prompt,
        checkpoint: chk,
        max_new_tokens: parseInt(document.getElementById('infer-max-tokens').value),
        temperature: parseFloat(document.getElementById('infer-temp').value),
        top_p: parseFloat(document.getElementById('infer-top-p').value),
        top_k: parseInt(document.getElementById('infer-top-k').value),
        repetition_penalty: parseFloat(document.getElementById('infer-rep-pen').value),
      };

      try {
        const outContainer = document.getElementById('infer-output-container');
        const respText = document.getElementById('infer-response-text');
        outContainer.classList.remove('hidden');
        respText.innerText = '';
        btn.innerHTML = '<span class="animate-spin">⏳</span> <span>Streaming...</span>';

        const res = await fetch('/api/evaluation/stream', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) {
          throw new Error('Stream request failed with status ' + res.status);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed.startsWith('data:')) {
              try {
                const chunkData = JSON.parse(trimmed.substring(5).trim());
                if (chunkData.token) {
                  respText.innerText += chunkData.token;
                  respText.scrollTop = respText.scrollHeight;
                }
                if (chunkData.done) {
                  if (chunkData.generation_time_sec !== undefined) {
                    document.getElementById('infer-time-label').innerText = `${chunkData.generation_time_sec} s`;
                  }
                  if (chunkData.tokens_per_sec !== undefined) {
                    document.getElementById('infer-speed-label').innerText = `${chunkData.tokens_per_sec} tok/s`;
                  }
                  if (document.getElementById('infer-device-badge')) {
                    document.getElementById('infer-device-badge').innerText = chunkData.device ? `Device: ${chunkData.device}` : '';
                  }
                }
              } catch (err) {
                console.error('SSE Chunk parse error:', err);
              }
            }
          }
        }

        if (document.getElementById('infer-checkpoint-badge')) {
          document.getElementById('infer-checkpoint-badge').innerText = payload.checkpoint ? `Checkpoint: ${payload.checkpoint}` : 'Base Model';
        }
      } catch (e) {
        console.warn('Streaming failed, falling back to batch generate:', e);
        try {
          const fallbackRes = await fetch('/api/evaluation/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const d = await fallbackRes.json();
          document.getElementById('infer-output-container').classList.remove('hidden');
          document.getElementById('infer-time-label').innerText = `${d.generation_time_sec} s`;
          document.getElementById('infer-speed-label').innerText = `${d.tokens_per_sec} tok/s`;
          document.getElementById('infer-response-text').innerText = d.generated_text;
        } catch (fErr) {
          console.error('Generation error:', fErr);
        }
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span>✨</span> <span>Generate</span>';
      }
    }

    // Checkpoints Refresh
    async function refreshCheckpointsTable() {
      try {
        const res = await fetch('/api/checkpoints');
        const list = await res.json();
        const tbody = document.getElementById('table-checkpoints-body');
        const select = document.getElementById('config-checkpoint');
        const inferSelect = document.getElementById('infer-checkpoint');
        select.innerHTML = '<option value="">Start from Scratch</option>';
        if (inferSelect) {
          inferSelect.innerHTML = '<option value="">Baseline / Active Model</option>';
        }

        if (!list || list.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-slate-500 italic">No checkpoints found.</td></tr>';
          return;
        }

        tbody.innerHTML = '';
        list.forEach(chk => {
          // Add to config select
          const opt = document.createElement('option');
          opt.value = chk.path || chk.filename;
          opt.innerText = `${chk.filename} (Step ${chk.step || 0})`;
          select.appendChild(opt);

          // Add to evaluation inference select
          if (inferSelect) {
            const opt2 = document.createElement('option');
            opt2.value = chk.path || chk.filename;
            opt2.innerText = `${chk.filename} (Step ${chk.step || 0}${chk.val_loss ? ', Val ' + chk.val_loss.toFixed(3) : ''})`;
            inferSelect.appendChild(opt2);
          }

          // Add row
          const tr = document.createElement('tr');
          const mod = chk.modified ? new Date(chk.modified * 1000).toLocaleString() : '--';
          tr.innerHTML = `
            <td class="py-2.5 font-semibold text-white">${chk.filename}</td>
            <td class="py-2.5 text-cyan-400">${chk.step || 0}</td>
            <td class="py-2.5 text-slate-300">${chk.val_loss !== null && chk.val_loss !== undefined ? chk.val_loss.toFixed(4) : '--'}</td>
            <td class="py-2.5 text-slate-400">${chk.size_mb || 0} MB</td>
            <td class="py-2.5 text-slate-500 text-[11px]">${mod}</td>
            <td class="py-2.5 text-right space-x-2">
              <button onclick="deleteCheckpoint('${chk.filename}')" class="text-rose-400 hover:text-rose-300 text-xs">Delete</button>
            </td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.error('Checkpoints fetch error:', e);
      }
    }

    async function refreshBenchmarkHistory() {
      try {
        const res = await fetch('/api/evaluation/results');
        const list = await res.json();
        const tbody = document.getElementById('table-benchmark-history-body');
        if (!tbody) return;
        if (!list || list.length === 0) {
          tbody.innerHTML = '<tr><td colspan="6" class="py-4 text-center text-slate-500 italic">No benchmark runs recorded.</td></tr>';
          return;
        }
        tbody.innerHTML = list.map(b => `
          <tr class="hover:bg-slate-900/40">
            <td class="py-2.5 font-semibold text-white">${b.category || '--'}</td>
            <td class="py-2.5 font-bold ${b.score >= 0.7 ? 'text-emerald-400' : (b.score >= 0.4 ? 'text-amber-400' : 'text-rose-400')}">${(b.score !== undefined && b.score !== null) ? (b.score * 100).toFixed(1) + '%' : '--'}</td>
            <td class="py-2.5 text-slate-300">${b.step || 0}</td>
            <td class="py-2.5 text-slate-400 text-[11px] truncate max-w-xs">${b.checkpoint_id || '--'}</td>
            <td class="py-2.5 text-cyan-400 text-[11px]">${b.run_id || '--'}</td>
            <td class="py-2.5 text-slate-500 text-[10px]">${b.created_at ? new Date(b.created_at * 1000).toLocaleString() : '--'}</td>
          </tr>
        `).join('');
      } catch (e) {
        console.error('Failed to load benchmark history:', e);
      }
    }

    async function refreshAlertsTable() {
      try {
        const res = await fetch('/api/alerts');
        const list = await res.json();
        const tbody = document.getElementById('table-alerts-body');
        if (!tbody) return;
        if (!list || list.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" class="py-4 text-center text-slate-500 italic">No alerts recorded.</td></tr>';
          return;
        }
        tbody.innerHTML = list.map(a => `
          <tr class="hover:bg-slate-900/40">
            <td class="py-2 font-bold ${a.level === 'CRITICAL' ? 'text-rose-400' : (a.level === 'WARNING' ? 'text-amber-400' : 'text-cyan-400')}">${a.level}</td>
            <td class="py-2 text-slate-300 font-semibold">${a.category}</td>
            <td class="py-2 text-slate-200 text-[11px]">${a.message}</td>
            <td class="py-2 text-slate-400">${a.step || 0}</td>
            <td class="py-2 text-slate-500 text-[10px]">${a.timestamp ? new Date(a.timestamp * 1000).toLocaleTimeString() : '--'}</td>
          </tr>
        `).join('');
      } catch (e) {
        console.error('Failed to load alerts:', e);
      }
    }

    async function deleteCheckpoint(filename) {
      if (!confirm(`Are you sure you want to delete checkpoint ${filename}?`)) return;
      await fetch(`/api/checkpoints/${filename}`, { method: 'DELETE' });
      refreshCheckpointsTable();
    }

    async function refreshRunsTable() {
      try {
        const res = await fetch('/api/runs');
        const runs = await res.json();
        const tbody = document.getElementById('table-runs-body');
        if (!runs || runs.length === 0) {
          tbody.innerHTML = '<tr><td colspan="7" class="py-4 text-center text-slate-500 italic">No runs recorded.</td></tr>';
          return;
        }
        tbody.innerHTML = '';
        runs.forEach(r => {
          const tr = document.createElement('tr');
          const d = new Date(r.created_at * 1000).toLocaleTimeString();
          tr.innerHTML = `
            <td class="py-2.5 font-semibold text-white">${r.id}</td>
            <td class="py-2.5 text-cyan-400 font-bold">${r.status}</td>
            <td class="py-2.5 text-slate-300">${r.current_phase}</td>
            <td class="py-2.5 text-slate-200">${r.total_steps || 0}</td>
            <td class="py-2.5 text-slate-400">${Number(r.total_tokens || 0).toLocaleString()}</td>
            <td class="py-2.5 text-emerald-400">${r.best_val_loss ? r.best_val_loss.toFixed(4) : '--'}</td>
            <td class="py-2.5 text-slate-500 text-[11px]">${d}</td>
          `;
          tbody.appendChild(tr);
        });
      } catch (e) {
        console.error('Runs fetch error:', e);
      }
    }

    // Datasets
    async function refreshRegisteredDatasets() {
      try {
        const res = await fetch('/api/datasets');
        const list = await res.json();
        const container = document.getElementById('registered-datasets-list');
        const configDsSelect = document.getElementById('config-dataset');

        // Update Training Configuration Dataset dropdown
        if (configDsSelect) {
          const currentVal = configDsSelect.value;
          configDsSelect.innerHTML = `
            <optgroup label="Standard Built-in Presets">
              <option value="tech_wiki_ro">tech_wiki_ro (Wikipedia & Tech Romanian - Phase 1)</option>
              <option value="chat_persona_ro">chat_persona_ro (Creator Persona - Phase 2 & 4)</option>
              <option value="alpaca_ro">alpaca_ro (Instruction Tuning - Phase 3)</option>
            </optgroup>
            <optgroup label="Uploaded User Datasets" id="config-dataset-uploaded-group"></optgroup>
            <optgroup label="Testing & Synthetic">
              <option value="synthetic">synthetic (Synthetic Fallback Mix - Testing Only)</option>
            </optgroup>
          `;
          const upGroup = document.getElementById('config-dataset-uploaded-group');
          if (list && list.length > 0 && upGroup) {
            list.forEach(ds => {
              const opt = document.createElement('option');
              opt.value = ds.id;
              opt.textContent = `${ds.name} (${Number(ds.sample_count).toLocaleString()} samples, ${Number(ds.token_count).toLocaleString()} tok)`;
              upGroup.appendChild(opt);
            });
          }
          if (currentVal) {
            configDsSelect.value = currentVal;
          }
        }

        if (!container) return;
        if (!list || list.length === 0) {
          container.innerHTML = '<div class="text-slate-500 italic text-xs py-2">No custom datasets uploaded yet. Upload above to inspect.</div>';
          return;
        }
        container.innerHTML = '';
        list.forEach(ds => {
          const card = document.createElement('div');
          card.className = 'p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs';
          card.innerHTML = `
            <div class="flex items-center justify-between">
              <span class="font-bold text-white text-sm">${ds.name}</span>
              <span class="px-2 py-0.5 rounded text-[10px] bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">ID: ${ds.id}</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-slate-400 font-mono">
              <div>Format: <span class="text-slate-200">${ds.format}</span></div>
              <div>Samples: <span class="text-slate-200">${Number(ds.sample_count).toLocaleString()}</span></div>
              <div>Estimated Tokens: <span class="text-cyan-400">${Number(ds.token_count).toLocaleString()}</span></div>
              <div>File Size: <span class="text-slate-200">${(ds.file_size_bytes / (1024*1024)).toFixed(1)} MB</span></div>
            </div>
            <div class="flex items-center justify-between pt-2 border-t border-slate-800/80">
              <span class="text-[10px] text-slate-400">Quality Filter & MinHash Deduplication</span>
              <button onclick="actionCurateDataset('${ds.id}')" id="btn-curate-${ds.id}" class="px-2.5 py-1 rounded bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 text-[11px] font-semibold transition">
                🧹 Curate & Deduplicate
              </button>
            </div>
            <div id="curate-stats-${ds.id}" class="hidden p-2.5 rounded bg-slate-950 border border-slate-800 text-[11px] font-mono space-y-1"></div>
          `;
          container.appendChild(card);
        });
      } catch (e) {
        console.error('Datasets fetch error:', e);
      }
    }

    // Drag and Drop Upload
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) {
      dropZone.onclick = () => {
        const fi = document.getElementById('file-input');
        if (fi) fi.click();
      };
      dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('border-cyan-500'); };
      dropZone.ondragleave = () => { dropZone.classList.remove('border-cyan-500'); };
      dropZone.ondrop = (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-cyan-500');
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
          handleFileSelected(e.dataTransfer.files[0]);
        }
      };
    }

    async function handleFileSelected(file) {
      if (!file) return;
      const progressContainer = document.getElementById('upload-progress-container');
      const progressBar = document.getElementById('upload-progress-bar');
      const fileNameLabel = document.getElementById('upload-file-name');
      const pctLabel = document.getElementById('upload-pct');

      progressContainer.classList.remove('hidden');
      fileNameLabel.innerText = file.name;
      progressBar.style.width = '30%';
      pctLabel.innerText = '30%';

      const formData = new FormData();
      formData.append('file', file);

      try {
        progressBar.style.width = '60%';
        pctLabel.innerText = '60%';
        const res = await fetch('/api/datasets/upload', {
          method: 'POST',
          body: formData
        });
        const d = await res.json();
        progressBar.style.width = '100%';
        pctLabel.innerText = '100%';
        alert(`Dataset ${file.name} uploaded successfully! Estimated tokens: ${Number(d.token_count).toLocaleString()}`);
        refreshRegisteredDatasets();
      } catch (e) {
        alert(`Upload failed: ${e}`);
      } finally {
        setTimeout(() => progressContainer.classList.add('hidden'), 2000);
      }
    }

    // Benchmarking
    async function runBenchmarkSuite() {
      const btn = document.getElementById('btn-run-benchmarks');
      btn.disabled = true;
      btn.innerText = 'Benchmarking...';
      const container = document.getElementById('benchmark-results-container');
      container.innerHTML = '<div class="text-cyan-400 italic text-xs py-2">Running evaluation on model...</div>';

      try {
        const res = await fetch('/api/evaluation/benchmark', { method: 'POST' });
        const d = await res.json();
        container.innerHTML = `
          <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-3">
            <div class="flex items-center justify-between">
              <span class="text-sm font-bold text-white">Mean Benchmark Score: <span class="text-cyan-400 font-mono">${(d.mean_score * 100).toFixed(1)}%</span></span>
              <span class="text-xs text-slate-400 font-mono">Step: ${d.step}</span>
            </div>
            <div class="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs font-mono">
              ${Object.entries(d.category_scores).map(([cat, score]) => `
                <div class="bg-slate-800/80 p-2 rounded border border-slate-700/60">
                  <span class="text-slate-400 block text-[10px] uppercase">${cat.replace('_', ' ')}</span>
                  <span class="text-white font-bold text-sm">${(score * 100).toFixed(1)}%</span>
                </div>
              `).join('')}
            </div>
          </div>
        `;
        refreshBenchmarkHistory();
      } catch (e) {
        container.innerHTML = `<div class="text-rose-400 text-xs">Benchmark execution failed: ${e}</div>`;
      } finally {
        btn.disabled = false;
        btn.innerText = 'Run Benchmark Suite';
      }
    }

    // Side-by-Side Checkpoint Comparison
    async function refreshComparisonCheckpoints() {
      try {
        const res = await fetch('/api/checkpoints');
        const list = await res.json();
        const container = document.getElementById('compare-checkpoints-selector');
        if (!list || list.length === 0) {
          container.innerHTML = '<span class="text-slate-500 italic">No checkpoints found to compare.</span>';
          return;
        }
        container.innerHTML = '';
        list.slice(0, 5).forEach((chk, idx) => {
          const lbl = document.createElement('label');
          lbl.className = 'flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-900 border border-slate-800 cursor-pointer hover:border-cyan-500';
          lbl.innerHTML = `<input type="checkbox" name="chk-compare" value="${chk.path || chk.filename}" ${idx < 2 ? 'checked' : ''} class="accent-cyan-400"> <span>${chk.filename}</span>`;
          container.appendChild(lbl);
        });
      } catch (e) {
        console.error('Comparison checkpoints load error:', e);
      }
    }

    async function runCheckpointComparison() {
      const checkedBoxes = Array.from(document.querySelectorAll('input[name="chk-compare"]:checked'));
      const paths = checkedBoxes.map(b => b.value);
      if (paths.length === 0) {
        alert('Please select at least one checkpoint to evaluate.');
        return;
      }
      const prompt = document.getElementById('infer-prompt').value || 'Explică arhitectura RoCanon-48.5M.';
      const grid = document.getElementById('compare-results-grid');
      grid.classList.remove('hidden');
      grid.innerHTML = '<div class="text-cyan-400 italic text-xs col-span-2">Executing sequential GPU evaluation...</div>';

      try {
        const res = await fetch('/api/evaluation/compare', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: prompt, checkpoint_paths: paths })
        });
        const d = await res.json();
        grid.innerHTML = '';
        d.results.forEach(r => {
          const card = document.createElement('div');
          card.className = 'p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2 text-xs font-mono';
          card.innerHTML = `
            <div class="flex items-center justify-between border-b border-slate-800 pb-1.5">
              <span class="font-bold text-white text-sm">${r.checkpoint_name}</span>
              <span class="text-cyan-400">${r.tokens_per_sec} tok/s</span>
            </div>
            <div class="text-slate-300 whitespace-pre-wrap leading-relaxed">${escapeHtml(r.generated_text || r.error || '')}</div>
          `;
          grid.appendChild(card);
        });
      } catch (e) {
        grid.innerHTML = `<div class="text-rose-400 text-xs col-span-2">Comparison failed: ${e}</div>`;
      }
    }

    async function initDatasetMonitor() {
      try {
        const res = await fetch('/api/dataset_info');
        const d = await res.json();
        if (d) {
          const elDs = document.getElementById('pipeline-dataset-name');
          if (elDs && d.name) elDs.innerText = d.name;
          const elTokTxt = document.getElementById('tok-status-text');
          if (elTokTxt) elTokTxt.innerText = (d.train_tokens > 0 ? Number(d.train_tokens).toLocaleString() + ' tokens cached' : (d.size_mb + ' MB dataset'));
          const elTokBadge = document.getElementById('tok-cache-badge');
          if (elTokBadge) elTokBadge.innerText = d.cached ? 'DISK CACHED' : 'READY TO TOKENIZE';
          const elTokBar = document.getElementById('tok-progress-bar');
          if (elTokBar && d.cached) elTokBar.style.width = '100%';
          const elTokPct = document.getElementById('tok-progress-pct');
          if (elTokPct && d.cached) elTokPct.innerText = '100%';
        }
      } catch (e) {}
    }

    async function actionCurateDataset(dsId) {
      const btn = document.getElementById(`btn-curate-${dsId}`);
      const statsBox = document.getElementById(`curate-stats-${dsId}`);
      if (btn) {
        btn.disabled = true;
        btn.innerText = 'Curating...';
      }
      if (statsBox) {
        statsBox.classList.remove('hidden');
        statsBox.innerHTML = '<span class="text-cyan-400 italic">Running MinHash LSH deduplication & Romanian LID filters...</span>';
      }
      try {
        const res = await fetch(`/api/datasets/${dsId}/curate`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ jaccard_threshold: 0.80, lid_threshold: 0.80, save_curated: true })
        });
        const d = await res.json();
        if (d.error) {
          if (statsBox) statsBox.innerHTML = `<span class="text-rose-400">Error: ${d.error}</span>`;
          return;
        }
        if (statsBox) {
          statsBox.innerHTML = `
            <div class="text-emerald-400 font-bold mb-1">✓ Curation Complete: ${d.retention_rate}% retained (${d.curated_samples} / ${d.total_samples} samples)</div>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-2 text-slate-300">
              <div>Exact Dupes: <span class="text-amber-400 font-semibold">${d.exact_duplicates_dropped}</span></div>
              <div>MinHash Dupes: <span class="text-rose-400 font-semibold">${d.minhash_duplicates_dropped}</span></div>
              <div>Low LID Rej: <span class="text-amber-400 font-semibold">${d.low_lid_dropped}</span></div>
              <div>PII Masked: <span class="text-cyan-400 font-semibold">${d.pii_masked_count}</span></div>
            </div>
          `;
        }
        refreshRegisteredDatasets();
      } catch (e) {
        if (statsBox) statsBox.innerHTML = `<span class="text-rose-400">Curation failed: ${e}</span>`;
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerText = '🧹 Curate & Deduplicate';
        }
      }
    }

    async function loadWebhookSettings() {
      try {
        const res = await fetch('/api/settings');
        const s = await res.json();
        if (s) {
          const dUrl = document.getElementById('setting-discord-webhook');
          if (dUrl && s.discord_webhook_url) dUrl.value = s.discord_webhook_url;
          const tgToken = document.getElementById('setting-telegram-token');
          if (tgToken && s.telegram_bot_token) tgToken.value = s.telegram_bot_token;
          const tgChat = document.getElementById('setting-telegram-chat');
          if (tgChat && s.telegram_chat_id) tgChat.value = s.telegram_chat_id;
          const nComp = document.getElementById('setting-notify-complete');
          if (nComp && s.notify_on_complete !== undefined) nComp.checked = Boolean(s.notify_on_complete);
          const nBest = document.getElementById('setting-notify-best');
          if (nBest && s.notify_on_best_checkpoint !== undefined) nBest.checked = Boolean(s.notify_on_best_checkpoint);
          const nSpike = document.getElementById('setting-notify-spikes');
          if (nSpike && s.notify_on_loss_spike !== undefined) nSpike.checked = Boolean(s.notify_on_loss_spike);
        }
      } catch (e) {
        console.debug('Failed to load webhook settings:', e);
      }
    }

    async function saveWebhookSettings() {
      const payload = {
        discord_webhook_url: (document.getElementById('setting-discord-webhook')?.value || '').trim(),
        telegram_bot_token: (document.getElementById('setting-telegram-token')?.value || '').trim(),
        telegram_chat_id: (document.getElementById('setting-telegram-chat')?.value || '').trim(),
        notify_on_complete: Boolean(document.getElementById('setting-notify-complete')?.checked),
        notify_on_best_checkpoint: Boolean(document.getElementById('setting-notify-best')?.checked),
        notify_on_loss_spike: Boolean(document.getElementById('setting-notify-spikes')?.checked),
        enable_webhooks: true,
      };
      try {
        const res = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          alert('Webhook notification settings saved successfully!');
        } else {
          alert('Failed to save webhook settings.');
        }
      } catch (e) {
        alert('Error saving webhook settings: ' + e);
      }
    }

    async function testWebhookNotifications() {
      const btn = document.getElementById('btn-test-webhook');
      const statusDiv = document.getElementById('webhook-test-status');
      if (btn) btn.disabled = true;
      if (statusDiv) {
        statusDiv.classList.remove('hidden');
        statusDiv.innerHTML = '<span class="text-cyan-400">Sending test payload to Discord & Telegram...</span>';
      }
      const payload = {
        discord_webhook_url: (document.getElementById('setting-discord-webhook')?.value || '').trim(),
        telegram_bot_token: (document.getElementById('setting-telegram-token')?.value || '').trim(),
        telegram_chat_id: (document.getElementById('setting-telegram-chat')?.value || '').trim(),
      };
      try {
        const res = await fetch('/api/notifications/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const d = await res.json();
        if (statusDiv) {
          let html = '<div class="space-y-1">';
          html += `<div>Discord: <span class="${d.discord ? 'text-emerald-400 font-bold' : 'text-slate-400'}">${d.discord ? '✓ Success' : (payload.discord_webhook_url ? '✗ Failed' : 'Not configured')}</span></div>`;
          html += `<div>Telegram: <span class="${d.telegram ? 'text-emerald-400 font-bold' : 'text-slate-400'}">${d.telegram ? '✓ Success' : (payload.telegram_bot_token ? '✗ Failed' : 'Not configured')}</span></div>`;
          if (d.errors && d.errors.length > 0) {
            html += `<div class="text-rose-400">${d.errors.join('<br>')}</div>`;
          }
          html += '</div>';
          statusDiv.innerHTML = html;
        }
      } catch (e) {
        if (statusDiv) statusDiv.innerHTML = `<span class="text-rose-400">Dispatch error: ${e}</span>`;
      } finally {
        if (btn) btn.disabled = false;
      }
    }

    // Startup
    window.addEventListener('DOMContentLoaded', () => {
      connectWebSocket();
      initDatasetMonitor();
      initOverviewChartsIfNeeded();
      pollTelemetry();
      setInterval(pollTelemetry, 1500);
      refreshCheckpointsTable();
      refreshOverviewNetworkOptions();
      loadWebhookSettings();
    });
  </script>
</body>
</html>
"""

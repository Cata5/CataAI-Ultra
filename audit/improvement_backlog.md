# LLM Maker — Prioritized Improvement Backlog

**Author:** Senior Machine Learning Infrastructure Architect  
**Date:** November 16, 2026  
**Status:** Living Engineering Roadmap  

---

## 1. High Priority (Immediate Next Iteration)

1. **FlashAttention-2 / SDPA Integration**:
   - Integrate `torch.nn.functional.scaled_dot_product_attention` (SDPA) with causal mask flag for non-GQA templates, further reducing activation VRAM on RTX 3060.
2. **Automatic SFT Data Formatter & Validator**:
   - Create a dedicated UI tool inside Datasets view to convert raw Alpaca / ShareGPT / Persona JSON files directly into `<|user|>` / `<|assistant|>` formatted tokens with proper turn masking (calculating cross-entropy loss exclusively on assistant tokens).
3. **Loss Spike Auto-Checkpoint Rollback**:
   - Enhance the Early Stopping & Anomaly engine to automatically detect gradient NaN or >3x loss spikes, pause training, roll back 1 checkpoint, and decrease learning rate by 25%.

---

## 2. Medium Priority (Platform Maturity)

4. **Quantization & GGUF Export**:
   - Provide 1-click GGUF (Q4_K_M, Q8_0) export for trained checkpoints directly from the Evaluation / Checkpoint tab, allowing immediate deployment in `llama.cpp` or Ollama.
5. **Interactive Latent Attention Heatmaps**:
   - In Network Studio / Evaluation, render attention weight heatmaps for generated tokens across the 8 query heads and 2 KV heads.
6. **Multi-Run Comparison Dashboard**:
   - Allow overlaying 2 or more historical training runs on the same Chart.js canvas to evaluate hyperparameter search sweeps (e.g. comparing Muon `0.02` vs `0.005`).

---

## 3. Low Priority (Future Explorations)

7. **Multi-GPU / Distributed DDP Support**:
   - Add PyTorch DDP / FSDP multi-GPU support for users transitioning from RTX 3060 laptops to desktop dual-GPU or server environments.
8. **Direct Preference Optimization (DPO)**:
   - Implement Phase 3 DPO alignment worker for paired chosen/rejected Romanian dialogues.

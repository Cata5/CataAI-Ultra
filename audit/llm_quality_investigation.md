# RoCanon-48.5M vs 38M Model Quality & Instruction Tuning Investigation

**Author:** Senior Machine Learning Infrastructure Architect & LLM Training Researcher  
**Date:** November 16, 2026  
**Subject:** Comparative Quality Diagnostic between `RoCanon-48.5M` and User's 38M Model (`D:\proiecte\LLM from Scratch`)  

---

## 1. Problem Statement

During conversational evaluation in the WebUI, the RoCanon-48.5M model generated repetitive word-salad and hallucinatory Romanian text when given instruction/chat prompts (such as *"Salut, cine esti?"*), whereas a smaller 38M model in `D:\proiecte\LLM from Scratch` (`model_phase2_chat_ro.pt`) exhibited coherent, fluent Romanian dialogue responses.

This investigation was commissioned to identify the exact root cause across tokenizers, chat templating, dataset formatting, and training dynamics.

---

## 2. Forensic Comparison of Models

| Dimension | 38M Model (`LLM from Scratch`) | RoCanon-48.5M (`CataAI Ultra`) |
| :--- | :--- | :--- |
| **Model Weights File** | `model_phase2_chat_ro.pt` (Phase 2 SFT) | `snapshot_step_9900.pt` / `checkpoints/*.pt` |
| **Parameter Count** | ~38,000,000 | 48,531,968 (Exact RoCanon specification) |
| **Tokenizer Vocabulary** | `bpe_4096_special.pkl` (4,096 vocab) | `tokenizer.json` / `vocab.json` (12,288 vocab) |
| **ChatML Special Tokens** | `<|im_start|>` (256), `<|im_end|>` (258) | `<|user|>` (5), `<|assistant|>` (6), `<|end_turn|>` (7) |
| **Training Phase** | **Phase 2 SFT Dialogue Tuning** on Romanian Alpaca & Persona | **Phase 1 Pre-training** on Wikipedia and Web crawls |
| **Inference Stop Tokens** | Stopped strictly on `<|im_end|>` (token 258) | Stopped only on `<|eos|>` (token 1); ignored `<|end_turn|>` |
| **Input Wrapping** | Wrapped with `<|im_start|>user\n...\n<|im_end|>\n<|im_start|>assistant\n` | Raw prompt sent directly without conversational formatting |

---

## 3. Root Cause Analysis

### 3.1 Special Token Mismatch & Stop Token Blindness
1. **The 38M Model** was fine-tuned with ChatML markup (`<|im_start|>` and `<|im_end|>`). When prompt is formatted with ChatML, it recognizes its role as `assistant` and generates a response followed immediately by `<|im_end|>`.
2. **RoCanon-48.5M Tokenizer** defined `<|user|>` (token 5), `<|assistant|>` (token 6), and `<|end_turn|>` (token 7), but lacked `<|im_start|>`.
3. In `src/eval/inference.py`, the generation loop only checked `eos_token_id = 1` (`<|eos|>`). Even if a fine-tuned model emitted `<|end_turn|>`, inference did not halt, causing the model to ramble past turn boundaries and deteriorate into incoherent loops.

### 3.2 Pre-training vs Instruction Tuning Corpus
- The RoCanon-48.5M snapshots evaluated in the UI were pre-trained on Wikipedia (`tech_wiki_ro`) using causal next-token prediction.
- In causal pre-training, asking *"Salut, cine esti?"* is naturally completed by the model as a fictional or journalistic excerpt (e.g. *"...a întrebat personajul principal în timp ce privea cerul..."*), rather than an assistant-style greeting.
- Treating a raw pre-trained base model as an instruction-tuned dialogue agent without SFT yields poor perceived conversational quality.

### 3.3 Dataset Boundary Fragmentation in `custom_loader.py`
- In earlier versions of `custom_loader.py`, multiline conversational datasets were tokenized as an unpartitioned 1D stream and chunked into fixed 256-token slices.
- As a consequence, dialogue turns were sliced in the middle of utterances, training the model on truncated user/assistant segments without context resets.

### 3.4 Excessive SFT Learning Rates
- When training dialogue datasets using Muon with the default `muon_lr = 0.02` (and adaptive multiplier `1.14` -> `2.28e-2`), the learning rate was 10x-50x too aggressive for instruction fine-tuning.
- High learning rates wipe out subtle pre-trained Romanian language representations within 200 steps (catastrophic forgetting).

---

## 4. Remediation Implemented

1. **Inference Stop Token Set**:
   In `src/eval/inference.py`, extended the stop condition to halt on any of:
   - `<|eos|>` (1)
   - `<|end_turn|>` (7)
   - `<|im_end|>` (if present in custom vocab)
   - `<|endoftext|>` (0)
2. **Automatic Chat / Instruct Formatting**:
   Added `chat_mode` toggle in UI and backend:
   ```python
   if chat_mode and "<|user|>" not in prompt:
       prompt = f"<|user|>\n{prompt}\n<|end_turn|>\n<|assistant|>\n"
   ```
3. **Completion Extraction**:
   Generation returns separate `completion` field with prompt and stop tokens stripped for clean UI display.
4. **SFT Fine-Tuning Recipe**:
   Documented recommended hyperparameters for Romanian SFT:
   - `optimizer: muon_adamw`
   - `muon_lr: 0.002` (reduced by 10x from pre-training 0.02)
   - `peak_lr: 0.0003` (AdamW for embeddings & norms)
   - `scheduler: cosine` with 50 warmup steps

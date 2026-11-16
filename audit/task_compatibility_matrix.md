# Task Compatibility Matrix

**Component:** `src/training/tasks/`  
**Registry:** `TaskRegistry`  

---

## 1. Supported Architecture and Task Matrix

| Training Task Adapter | Family ID | Progress Unit | Compatible Architecture Families | Loss Function | Primary Evaluated Metric | UI Metric Cards |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Causal Language Modeling** | `causal_lm` | `tokens` | `decoder_only`, `hybrid`, `custom_pytorch` | Next-Token Cross-Entropy | `token_loss` | Token Loss, Perplexity, Throughput (tok/s) |
| **Instruction Fine-Tuning (SFT)** | `instruction_sft` | `tokens` | `decoder_only`, `hybrid`, `custom_pytorch` | Masked Prompt Cross-Entropy | `sft_loss` | SFT Loss, Perplexity, Tokens/sec |
| **Classification (Vision & Tabular)** | `classification` | `samples` | `cnn`, `mlp`, `encoder_only`, `rnn_lstm_gru`, `custom_pytorch` | Multi-class Cross-Entropy / BCE | `accuracy` | Cross-Entropy Loss, Accuracy (%), Samples/sec |
| **Regression (Continuous Numerical)** | `regression` | `samples` | `mlp`, `cnn`, `rnn_lstm_gru`, `encoder_only`, `custom_pytorch` | MSE / L1 (MAE) | `rmse` | Regression Loss, MAE, RMSE, Samples/sec |

---

## 2. Dynamic Metric Presentation

The UI dynamically adapts to the selected training task:
- For **Language Modeling tasks (`causal_lm`, `instruction_sft`)**: The UI displays Token Loss, Validation Loss, Perplexity, Cumulative Tokens Seen, and Tokens/sec.
- For **Computer Vision & Classification tasks (`classification`)**: The UI replaces token counts with Samples Processed, displays Accuracy (%), Classification Loss, and Samples/sec.
- For **Numerical & Time-Series tasks (`regression`)**: The UI displays MAE, RMSE, Regression Loss, and Processed Samples.

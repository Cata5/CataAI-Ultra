"""
train.py
Main entry point for RoCanon-48.5M training, validation, and pilot execution.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from src.model.architecture import RoCanonConfig, RoCanonForCausalLM
from src.tokenizer.tokenizer import RoCanonTokenizer
from src.training.loss import ChunkedCrossEntropyLoss
from src.training.optimizer import configure_optimizers
from src.training.scheduler import WSDScheduler
from src.training.trainer import Trainer, TrainingConfig
from src.utils.memory import MemoryTracker


def run_pilot(steps: int = 15, device_str: str | None = None) -> None:
    """
    Executes an end-to-end pilot / sanity training run for RoCanon-48.5M.
    Verifies forward, backward, dual optimizer steps, loss decrease,
    and memory usage below the 5.0 GB safety threshold.
    """
    print("=" * 70)
    print("RoCanon-48.5M: End-to-End Pilot Training Verification")
    print("=" * 70)

    # 1. Device selection
    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Compute Device: {device}")

    # 2. Model initialization
    print("\n[1/5] Initializing RoCanon-48.5M Model Architecture...")
    config = RoCanonConfig()
    model = RoCanonForCausalLM(config).to(device=device)

    total_params = sum(p.numel() for p in set(model.parameters()))
    non_emb_params = model.get_num_params(non_embedding=True)
    tied_emb_params = model.embed_tokens.weight.numel()
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Non-Embedding Backbone: {non_emb_params:,}")
    print(f"  Tied Embeddings: {tied_emb_params:,}")
    assert total_params == 48531968, f"Parameter count mismatch: got {total_params}"

    # 3. Tokenizer initialization
    print("\n[2/5] Initializing Tokenizer...")
    tokenizer = RoCanonTokenizer()
    print(f"  Vocabulary Size: {tokenizer.vocab_size} tokens")

    # 4. Setup Hybrid Optimizer & Trainer
    print("\n[3/5] Configuring Hybrid Optimizer (Muon + AdamW) & Trainer...")
    muon_opt, adamw_opt = configure_optimizers(
        model,
        muon_lr=3.0e-3,
        adamw_lr=6.0e-4,
        fused=True,
    )
    muon_params = sum(p.numel() for g in muon_opt.param_groups for p in g["params"])
    adamw_params = sum(p.numel() for g in adamw_opt.param_groups for p in g["params"])
    print(f"  Muon Optimizer (2D weight matrices): {muon_params:,} parameters")
    print(f"  AdamW Optimizer (1D/Norms/Embeddings/Canon): {adamw_params:,} parameters")

    scheduler = WSDScheduler(
        optimizers=(muon_opt, adamw_opt),
        warmup_steps=2,
        plateau_tokens=10000,
        total_tokens=50000,
    )

    training_cfg = TrainingConfig(
        device=str(device),
        dtype=torch.bfloat16,
        max_grad_norm=1.0,
        grad_accum_steps=2,
        micro_batch_size=4,
        seq_len=256,
        max_budget_gb=5.0,
    )

    trainer = Trainer(
        model=model,
        optimizers=(muon_opt, adamw_opt),
        scheduler=scheduler,
        config=training_cfg,
    )

    # 5. Synthetic Pilot Data
    print("\n[4/5] Preparing Bilingual Pilot Data Shards...")
    samples = [
        "Proiectarea și antrenarea unui Small Language Model românesc de 48.5M parametri.",
        "The RoCanon architecture unifies causal convolutions with Grouped-Query Attention.",
        "Modelul optimizează reprezentarea sintactică și semantică bilingvă română-engleză.",
        "Inference is highly efficient due to 4:1 GQA reducing KV cache memory footprint.",
        "Algoritmul de optimizare Muon folosește iterații Newton-Schulz de gradul cinci.",
        "Straturile Canon depthwise adaugă context local fără a mări costul pătratic.",
    ]
    packed_ids = []
    for s in samples:
        packed_ids.extend(tokenizer.encode(s))
        packed_ids.append(tokenizer.eos_token_id)

    # Repeat to fill sequence batches
    while len(packed_ids) < 4 * 256 * 2:
        packed_ids.extend(packed_ids)

    # 6. Execute Pilot Training Steps
    print(f"\n[5/5] Executing {steps} Pilot Training Steps...")
    print(f"{'Step':>6} | {'Loss':>10} | {'Grad Norm':>10} | {'VRAM Peak':>12} | {'Step Time':>10}")
    print("-" * 62)

    initial_loss = None
    final_loss = None
    step_losses = []

    for step in range(1, steps + 1):
        t0 = time.time()
        trainer.zero_grad()

        # Two accumulation micro-batches
        for accum_idx in range(training_cfg.grad_accum_steps):
            start_pos = ((step * 2 + accum_idx) * 256) % (len(packed_ids) - 257)
            sub_ids = packed_ids[start_pos : start_pos + 4 * 256]
            batch_tensor = torch.tensor(sub_ids[: 4 * 256], dtype=torch.long, device=device).view(4, 256)

            input_ids = batch_tensor[:, :-1]
            labels = batch_tensor[:, 1:]

            trainer.accumulate_micro_batch(input_ids=input_ids, labels=labels)

        # Optimizer step
        tokens_in_step = 4 * 255 * training_cfg.grad_accum_steps
        grad_norm = trainer.optimizer_step(num_tokens_in_step=tokens_in_step)
        loss_val = trainer.loss_accumulator.reset()

        dt = time.time() - t0
        step_losses.append(loss_val)

        if initial_loss is None:
            initial_loss = loss_val
        final_loss = loss_val

        # Memory tracking
        trainer.memory_tracker.assert_within_budget(5.0)
        if torch.cuda.is_available():
            peak_gb = trainer.memory_tracker.get_peak_allocated_gb()
            vram_str = f"{peak_gb:.2f} GB"
        else:
            vram_str = "N/A (CPU)"

        print(f"{step:>6} | {loss_val:>10.4f} | {grad_norm:>10.4f} | {vram_str:>12} | {dt:>9.2f}s")

    print("-" * 62)
    print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
    loss_drop = initial_loss - final_loss
    print(f"Total Loss Decrease: {loss_drop:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease during pilot run!"

    # Save pilot checkpoint
    chk_dir = Path("checkpoints")
    chk_dir.mkdir(exist_ok=True)
    chk_path = chk_dir / "pilot_checkpoint.pt"
    torch.save(
        {
            "global_step": steps,
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        chk_path,
    )
    print(f"\nPilot checkpoint successfully saved to: {chk_path}")
    print("\n>>> RoCanon-48.5M PILOT VERIFICATION SUCCESSFUL! <<<")


def main():
    parser = argparse.ArgumentParser(description="RoCanon-48.5M Training & Verification Engine")
    parser.add_argument("--pilot", action="store_true", help="Run 15-step pilot verification")
    parser.add_argument("--steps", type=int, default=15, help="Number of steps for pilot")
    parser.add_argument("--config", type=str, default="configs/stage1_pretrain.yaml", help="Path to YAML training config")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda')")
    args = parser.parse_args()

    if args.pilot:
        run_pilot(steps=args.steps, device_str=args.device)
    else:
        print(f"Loading configuration from {args.config}...")
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded config for stage: {cfg.get('name', 'unnamed')}")
        print("To launch a verification pilot, run: python train.py --pilot")


if __name__ == "__main__":
    main()

"""
src/checkpoints/charts.py
Generates publication-grade, dark-themed training and validation dynamics charts.
Saves val_loss_curve.png, training_dynamics.png, and learning_rate.png in the model directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Cyber / Neural Workstation Dark Theme Palette
STYLE_CONFIG = {
    "bg_color": "#090D14",
    "card_bg": "#0E1420",
    "plot_bg": "#131C29",
    "border_color": "#273548",
    "grid_color": "#1C2738",
    "text_primary": "#ECF3FC",
    "text_secondary": "#91A0B4",
    "text_muted": "#5E7188",
    "color_train_loss": "#45CFFF",    # Cyan
    "color_val_loss": "#35D49A",      # Emerald
    "color_muon_lr": "#9B8CFF",       # Violet
    "color_adamw_lr": "#F2B85B",      # Amber
    "color_tok_s": "#80DEFF",         # Light Blue
    "color_best_marker": "#35D49A",   # Bright Emerald
}


def _setup_dark_style(plt):
    """Configures matplotlib with the neural workstation dark aesthetic."""
    plt.rcParams["figure.facecolor"] = STYLE_CONFIG["bg_color"]
    plt.rcParams["axes.facecolor"] = STYLE_CONFIG["plot_bg"]
    plt.rcParams["axes.edgecolor"] = STYLE_CONFIG["border_color"]
    plt.rcParams["axes.labelcolor"] = STYLE_CONFIG["text_secondary"]
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = STYLE_CONFIG["grid_color"]
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.6
    plt.rcParams["xtick.color"] = STYLE_CONFIG["text_muted"]
    plt.rcParams["ytick.color"] = STYLE_CONFIG["text_muted"]
    plt.rcParams["text.color"] = STYLE_CONFIG["text_primary"]
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]


def generate_training_charts(
    metrics: Sequence[Dict[str, Any]],
    output_dir: Path | str,
    model_id: str = "model",
    best_step: Optional[int] = None,
    best_val_loss: Optional[float] = None,
) -> Dict[str, str]:
    """
    Renders high-resolution PNG charts from training metrics history.
    Returns mapping of chart name to relative file path.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    if not metrics:
        logger.warning("No metrics provided for chart generation.")
        return results

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        logger.warning("Matplotlib not available for chart generation: %s", e)
        return results

    _setup_dark_style(plt)

    # Extract series data
    steps: List[int] = []
    train_losses: List[float] = []
    val_steps: List[int] = []
    val_losses: List[float] = []
    muon_lrs: List[float] = []
    adamw_lrs: List[float] = []
    throughput_steps: List[int] = []
    tok_per_sec: List[float] = []
    step_times: List[float] = []

    for m in metrics:
        st = int(m.get("step") or 0)
        steps.append(st)

        # Loss
        t_loss = m.get("train_loss") if m.get("train_loss") is not None else m.get("loss")
        if t_loss is not None:
            train_losses.append(float(t_loss))
        else:
            train_losses.append(float("nan"))

        # Val loss
        v_loss = m.get("val_loss")
        if v_loss is not None:
            val_steps.append(st)
            val_losses.append(float(v_loss))

        # LRs
        m_lr = m.get("muon_lr") if m.get("muon_lr") is not None else m.get("learning_rate") or m.get("lr")
        muon_lrs.append(float(m_lr) if m_lr is not None else 0.0)

        a_lr = m.get("adamw_lr")
        adamw_lrs.append(float(a_lr) if a_lr is not None else 0.0)

        # Throughput & Latency
        tps = m.get("tokens_per_second") or m.get("tok_per_sec")
        if tps is not None and float(tps) > 0:
            throughput_steps.append(st)
            tok_per_sec.append(float(tps))
            st_ms = m.get("step_time_ms") or (float(m.get("dt", 0)) * 1000 if m.get("dt") else 0)
            step_times.append(float(st_ms))

    # Determine best val loss if not provided
    if val_losses:
        min_idx = int(np.argmin(val_losses))
        calc_best_val = val_losses[min_idx]
        calc_best_step = val_steps[min_idx]
        if best_val_loss is None:
            best_val_loss = calc_best_val
        if best_step is None:
            best_step = calc_best_step

    # -------------------------------------------------------------
    # 1. Validation & Training Loss Curve (val_loss_curve.png)
    # -------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
        fig.patch.set_facecolor(STYLE_CONFIG["bg_color"])

        # Plot Train Loss
        ax.plot(
            steps,
            train_losses,
            label="Training Loss",
            color=STYLE_CONFIG["color_train_loss"],
            linewidth=1.8,
            alpha=0.85,
        )

        # Plot Validation Loss
        if val_steps and val_losses:
            ax.plot(
                val_steps,
                val_losses,
                label="Validation Loss",
                color=STYLE_CONFIG["color_val_loss"],
                linewidth=2.4,
                linestyle="--",
                marker="o",
                markersize=5,
                markeredgecolor=STYLE_CONFIG["bg_color"],
                markeredgewidth=1.2,
            )

            # Highlight Best Validation Point
            if best_step is not None and best_val_loss is not None:
                ax.scatter(
                    [best_step],
                    [best_val_loss],
                    color=STYLE_CONFIG["color_best_marker"],
                    s=140,
                    zorder=5,
                    edgecolors="#FFFFFF",
                    linewidths=2.0,
                    label=f"Best Val: {best_val_loss:.4f} (Step {best_step})",
                )
                ax.annotate(
                    f"Best: {best_val_loss:.4f}\nStep {best_step}",
                    xy=(best_step, best_val_loss),
                    xytext=(best_step, best_val_loss + 0.15 * (max(val_losses) - min(val_losses) + 0.1)),
                    arrowprops=dict(
                        facecolor=STYLE_CONFIG["color_best_marker"],
                        edgecolor=STYLE_CONFIG["border_color"],
                        arrowstyle="->",
                        lw=1.5,
                    ),
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc=STYLE_CONFIG["card_bg"],
                        ec=STYLE_CONFIG["color_best_marker"],
                        lw=1.2,
                    ),
                    color=STYLE_CONFIG["text_primary"],
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_title(f"Model Loss Convergence — {model_id}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Optimization Step", fontsize=10, labelpad=8)
        ax.set_ylabel("Cross Entropy Loss", fontsize=10, labelpad=8)
        ax.legend(loc="upper right", framealpha=0.9, facecolor=STYLE_CONFIG["card_bg"], edgecolor=STYLE_CONFIG["border_color"])

        plt.tight_layout()
        val_chart_path = out_dir / "val_loss_curve.png"
        fig.savefig(val_chart_path, dpi=150)
        plt.close(fig)
        results["val_loss_curve"] = str(val_chart_path)
    except Exception as e:
        logger.warning("Failed to render val_loss_curve.png: %s", e)

    # -------------------------------------------------------------
    # 2. Comprehensive Training Dynamics (training_dynamics.png)
    # -------------------------------------------------------------
    try:
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), dpi=150, sharex=True)
        fig.patch.set_facecolor(STYLE_CONFIG["bg_color"])

        # Top: Loss Dynamics
        ax_loss = axes[0]
        ax_loss.plot(steps, train_losses, color=STYLE_CONFIG["color_train_loss"], label="Train Loss", lw=1.6)
        if val_steps and val_losses:
            ax_loss.plot(val_steps, val_losses, color=STYLE_CONFIG["color_val_loss"], linestyle="--", marker="o", label="Val Loss", lw=2.0)
            if best_step is not None and best_val_loss is not None:
                ax_loss.scatter([best_step], [best_val_loss], color="#35D49A", s=100, zorder=5, edgecolors="#FFF")
        ax_loss.set_title(f"{model_id} — Training & Validation Dynamics", fontsize=12, fontweight="bold")
        ax_loss.set_ylabel("Loss", fontsize=10)
        ax_loss.legend(loc="upper right", facecolor=STYLE_CONFIG["card_bg"], edgecolor=STYLE_CONFIG["border_color"])

        # Middle: Dual Optimizer Learning Rates
        ax_lr = axes[1]
        if any(muon_lrs):
            ax_lr.plot(steps, muon_lrs, color=STYLE_CONFIG["color_muon_lr"], label="Muon LR (Matrix)", lw=1.8)
        if any(adamw_lrs):
            ax_lr.plot(steps, adamw_lrs, color=STYLE_CONFIG["color_adamw_lr"], label="AdamW LR (Vector)", lw=1.8)
        ax_lr.set_ylabel("Learning Rate", fontsize=10)
        ax_lr.set_yscale("log")
        ax_lr.legend(loc="upper right", facecolor=STYLE_CONFIG["card_bg"], edgecolor=STYLE_CONFIG["border_color"])

        # Bottom: Throughput & Latency
        ax_perf = axes[2]
        if throughput_steps and tok_per_sec:
            ax_perf.plot(throughput_steps, tok_per_sec, color=STYLE_CONFIG["color_tok_s"], label="Tokens/sec", lw=1.6)
            ax_perf.set_ylabel("Tokens / Sec", color=STYLE_CONFIG["color_tok_s"], fontsize=10)
            ax_perf.tick_params(axis="y", labelcolor=STYLE_CONFIG["color_tok_s"])

            if step_times:
                ax_lat = ax_perf.twinx()
                ax_lat.plot(throughput_steps, step_times, color="#F2B85B", linestyle=":", label="Step Latency (ms)", lw=1.4)
                ax_lat.set_ylabel("Step Latency (ms)", color="#F2B85B", fontsize=10)
                ax_lat.tick_params(axis="y", labelcolor="#F2B85B")
                ax_lat.grid(False)

        ax_perf.set_xlabel("Optimization Step", fontsize=10)
        ax_perf.legend(loc="upper left", facecolor=STYLE_CONFIG["card_bg"], edgecolor=STYLE_CONFIG["border_color"])

        plt.tight_layout()
        dynamics_path = out_dir / "training_dynamics.png"
        fig.savefig(dynamics_path, dpi=150)
        plt.close(fig)
        results["training_dynamics"] = str(dynamics_path)
    except Exception as e:
        logger.warning("Failed to render training_dynamics.png: %s", e)

    # -------------------------------------------------------------
    # 3. Learning Rate Schedule Curve (learning_rate.png)
    # -------------------------------------------------------------
    try:
        fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
        fig.patch.set_facecolor(STYLE_CONFIG["bg_color"])
        if any(muon_lrs):
            ax.plot(steps, muon_lrs, color=STYLE_CONFIG["color_muon_lr"], label="Muon LR", lw=2.0)
        if any(adamw_lrs):
            ax.plot(steps, adamw_lrs, color=STYLE_CONFIG["color_adamw_lr"], label="AdamW LR", lw=2.0)
        ax.set_title("Dual Learning Rate Schedule (WSD)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Learning Rate", fontsize=9)
        ax.legend(loc="best", facecolor=STYLE_CONFIG["card_bg"], edgecolor=STYLE_CONFIG["border_color"])

        plt.tight_layout()
        lr_path = out_dir / "learning_rate.png"
        fig.savefig(lr_path, dpi=150)
        plt.close(fig)
        results["learning_rate"] = str(lr_path)
    except Exception as e:
        logger.warning("Failed to render learning_rate.png: %s", e)

    logger.info("Generated %d training charts in %s", len(results), out_dir)
    return results

"""Backtest evaluation, result export, visualization, and checkpointing."""

import os
import json
import math
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import TEST_IDS, OUTPUT_DIR, EPISODE_LEN, MAX_ITERATIONS


# ────────────────── Backtest ──────────────────

def backtest_series(policy, feats, rets, label=""):
    """Run backtest with parallel sliding window (same as train)."""
    device = next(policy.parameters()).device
    feats = feats.to(device)
    rets = rets.to(device)
    T = feats.shape[0] - 1                               # 可决策的时间步
    D = feats.shape[1]
    with torch.no_grad():
        f = feats[:T]                                    # (T, D)
        n_w = (T + EPISODE_LEN - 1) // EPISODE_LEN
        pad = n_w * EPISODE_LEN - T
        if pad:
            f = torch.cat([f, f.new_zeros(pad, D)], dim=0)

        # 所有窗口作为 batch 并行 forward
        windows = f.reshape(n_w, EPISODE_LEN, D)         # (n_w, EPISODE_LEN, D)
        logits, exit_iters_w, _, _ = policy(windows)     # logits: (n_w, n_actions)

        # Shift: window i's action → timesteps [(i+1)*EPISODE_LEN : (i+2)*EPISODE_LEN]
        # First EPISODE_LEN timesteps: neutral position 5
        actions = logits.argmax(dim=-1)                    # (n_w,)
        default = torch.full((EPISODE_LEN,), 5, dtype=actions.dtype, device=actions.device)
        shifted = actions[:-1].unsqueeze(-1).expand(-1, EPISODE_LEN).reshape(-1)
        positions = torch.cat([default, shifted])[:T]

        # per-window depth → per-token
        exit_iters = exit_iters_w.cpu().numpy()
        exit_iters = np.repeat(exit_iters, EPISODE_LEN)[:T]

    rets_f = rets[1:T + 1]                              # aligned with positions[0:T]
    pos_w = positions.float() / 10.0

    # strategy equity — all on GPU
    strat_daily = pos_w * rets_f
    strat_eq = torch.cumprod(1 + strat_daily, dim=0)
    bh_eq = torch.cumprod(1 + rets_f, dim=0)

    strat_total = strat_eq[-1].item()
    bh_total    = bh_eq[-1].item()
    strat_peak  = torch.cummax(strat_eq, dim=0).values
    strat_dd    = (strat_peak - strat_eq) / (strat_peak + 1e-8)
    bh_peak     = torch.cummax(bh_eq, dim=0).values
    bh_dd       = (bh_peak - bh_eq) / (bh_peak + 1e-8)

    strat_maxdd = min(strat_dd.max().item(), 1.0)
    bh_maxdd    = min(bh_dd.max().item(), 1.0)

    strat_sharpe = (strat_daily.mean() / (strat_daily.std() + 1e-8) * math.sqrt(252)).item()
    bh_sharpe    = (rets_f.mean() / (rets_f.std() + 1e-8) * math.sqrt(252)).item()

    n_days = positions.shape[0]
    strat_calmar = ((strat_total ** (252/n_days) - 1)) / (strat_maxdd + 1e-8)
    bh_calmar = 0.0 if bh_total < 1e-10 else ((bh_total ** (252/n_days) - 1)) / (bh_maxdd + 1e-8)

    metrics = {
        "strat_total": strat_total, "bh_total": bh_total,
        "strat_maxdd": strat_maxdd, "bh_maxdd": bh_maxdd,
        "strat_sharpe": strat_sharpe, "bh_sharpe": bh_sharpe,
        "strat_calmar": strat_calmar, "bh_calmar": bh_calmar,
    }

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  {'Metric':<20} {'Strategy':>14} {'Buy&Hold':>14}")
    print(f"  {'-'*48}")
    print(f"  {'Total Return':<20} {strat_total:>13.2f}x {bh_total:>13.2f}x")
    print(f"  {'Max Drawdown':<20} {strat_maxdd:>13.2%} {bh_maxdd:>13.2%}")
    print(f"  {'Sharpe (ann.)':<20} {strat_sharpe:>13.2f}  {bh_sharpe:>13.2f}")
    print(f"  {'Calmar':<20} {strat_calmar:>13.2f}  {bh_calmar:>13.2f}")

    # numpy only at the boundary for matplotlib
    return {
        "metrics": metrics,
        "positions": positions.cpu().numpy(),
        "exit_iters": exit_iters,
        "strat_eq": strat_eq.cpu().numpy(),
        "bh_eq": bh_eq.cpu().numpy(),
        "strat_dd": strat_dd.cpu().numpy(),
        "bh_dd": bh_dd.cpu().numpy(),
        "rets": rets_f.cpu().numpy(),
    }


def run_backtest(policy, test_feats, test_rets, test_ids=None):
    """Run backtest on all test series."""
    if test_ids is None:
        test_ids = TEST_IDS
    results = []
    for i, sid in enumerate(test_ids):
        r = backtest_series(policy, test_feats[i], test_rets[i],
                            label=f"Test Series {sid}")
        results.append(r)
    return results


# ────────────────── Model Checkpoint ──────────────────

def save_checkpoint(model, optimizer, scheduler, history, epoch, path):
    """Save full training state for resume."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
    }, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load training state. Returns (epoch, history)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Checkpoint loaded ← {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("epoch", 0), ckpt.get("history", {})


def save_model(model, path):
    """Save model weights only (for inference)."""
    torch.save(model.state_dict(), path)
    print(f"Model saved → {path}")


def load_model(model, path):
    """Load model weights for inference."""
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
    print(f"Model loaded ← {path}")
    return model


# ────────────────── Result Export ──────────────────

def export_results(results, history, test_ids=None, output_dir=None):
    """Export backtest details + training history to CSV and metrics to JSON."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if test_ids is None:
        test_ids = TEST_IDS
    os.makedirs(output_dir, exist_ok=True)

    # Per-series backtest CSV
    for i, sid in enumerate(test_ids):
        r = results[i]
        df = pd.DataFrame({
            "day": np.arange(len(r["strat_eq"])),
            "position": r["positions"],
            "loop_depth": r["exit_iters"] if "exit_iters" in r else np.nan,
            "daily_return": r["rets"],
            "strat_equity": r["strat_eq"],
            "bh_equity": r["bh_eq"],
            "strat_drawdown": r["strat_dd"],
            "bh_drawdown": r["bh_dd"],
        })
        csv_path = os.path.join(output_dir, f"backtest_series_{sid}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Exported → {csv_path}")

    # Training history CSV
    hist_path = os.path.join(output_dir, "training_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index_label="episode")
    print(f"Exported → {hist_path}")

    # Summary metrics JSON
    summary = {}
    for i, sid in enumerate(test_ids):
        m = results[i]["metrics"]
        summary[f"series_{sid}"] = {
            "strategy": {
                "total_return": m["strat_total"],
                "max_drawdown": m["strat_maxdd"],
                "sharpe_annual": m["strat_sharpe"],
                "calmar": m["strat_calmar"],
            },
            "buy_and_hold": {
                "total_return": m["bh_total"],
                "max_drawdown": m["bh_maxdd"],
                "sharpe_annual": m["bh_sharpe"],
                "calmar": m["bh_calmar"],
            },
        }
    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Exported → {json_path}")


# ────────────────── Plotting ──────────────────

def _save_fig(fig, name, output_dir=None):
    d = output_dir or "."
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_prices(prices_list, output_dir=None):
    n = len(prices_list)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5*n), sharex=True, squeeze=False)
    for i in range(n):
        axes[i, 0].plot(prices_list[i]["date"].to_numpy(), prices_list[i]["close"].to_numpy(), lw=.5)
        axes[i, 0].set_ylabel(f"S{i}")
    plt.suptitle("Synthetic A-Stock Price Series", y=1.01)
    plt.tight_layout()
    _save_fig(fig, "fig_01_prices.png", output_dir)


def plot_training_curves(history, output_dir=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["loss"]);       axes[0].set_title("GRPO Loss")
    axes[1].plot(history["reward_mean"]);axes[1].set_title("Mean Reward")
    axes[2].plot(history["reward_std"]); axes[2].set_title("Reward Std")
    for ax in axes: ax.set_xlabel("Episode")
    plt.tight_layout()
    _save_fig(fig, "fig_02_training.png", output_dir)


def plot_results(results, test_ids=None, output_dir=None):
    """Plot equity, drawdown, position heatmap, and position distribution."""
    if test_ids is None:
        test_ids = TEST_IDS

    for i, sid in enumerate(test_ids):
        r = results[i]
        T = len(r["strat_eq"])
        days = np.arange(T)

        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 3, 1, 1]})

        # 1) Equity curves
        axes[0].plot(days, r["bh_eq"],    lw=1, alpha=.6, label="Buy & Hold", color="gray")
        axes[0].plot(days, r["strat_eq"], lw=1, label="KDA+GRPO Strategy", color="tab:blue")
        axes[0].set_ylabel("Equity (start=1)")
        axes[0].legend()
        axes[0].set_title(f"Test Series {sid} — Equity Curves")

        # 2) Drawdown
        axes[1].fill_between(days, -r["bh_dd"],    alpha=.3, color="gray", label="B&H DD")
        axes[1].fill_between(days, -r["strat_dd"], alpha=.4, color="tab:blue", label="Strategy DD")
        axes[1].set_ylabel("Drawdown")
        axes[1].legend()

        # 3) Position — fill_between like depth display
        pos = r["positions"]
        pos_days = np.arange(len(pos))
        axes[2].fill_between(pos_days, pos, alpha=.7, color="tab:green")
        axes[2].set_ylabel("Position")
        axes[2].set_ylim(0, 10)
        axes[2].set_yticks(range(0, 11, 2))

        # 4) Loop depth (exit iteration per token)
        if "exit_iters" in r:
            exit_days = np.arange(len(r["exit_iters"]))
            axes[3].fill_between(exit_days, r["exit_iters"], alpha=.6, color="tab:orange")
            axes[3].set_ylabel("Loop Depth")
            axes[3].set_xlabel("Trading Day")
            axes[3].set_ylim(0, MAX_ITERATIONS)
            axes[3].set_yticks(range(0, MAX_ITERATIONS + 1, max(1, MAX_ITERATIONS // 4)))

        plt.tight_layout()
        _save_fig(fig, f"fig_03_test_series_{sid}.png", output_dir)

    # Position distribution
    n_test = len(test_ids)
    fig, axes = plt.subplots(1, n_test, figsize=(4*n_test, 4), squeeze=False)
    axes = axes[0]
    for i, sid in enumerate(test_ids):
        pos = results[i]["positions"]
        axes[i].hist(pos, bins=11, range=(-0.5, 10.5), edgecolor="black", align="mid")
        axes[i].set_title(f"Series {sid} — Position Distribution")
        axes[i].set_xlabel("Position (0–10)")
        axes[i].set_ylabel("Count")
    plt.tight_layout()
    _save_fig(fig, "fig_04_position_dist.png", output_dir)


def print_conclusions():
    print("\n" + "="*60)
    print("  CONCLUSIONS")
    print("="*60)
    print("""
Model: Mini-KDA Policy Network (adapted from Kimi Linear)
  - 3 stacked KDA layers with delta-rule recurrence
  - Channel-wise gating (α) for fine-grained memory control
  - PoPE position encoding (content-position decoupling)
  - AttnRes depth aggregation
  - ~26K parameters total

Training: GRPO (from DeepSeekMath)
  - No critic/value model needed
  - Group-relative advantage estimation (G=24 samples)
  - Reward = log_return − λ·max_drawdown  (λ=2.0)
  - DDP multi-GPU via torchrun

Key observations:
  1. MMn differential features capture short/long-term trend shifts
     through the rolling log-min and log-max dynamics.
  2. The KDA recurrence acts as an adaptive memory that tracks
     market regime changes across multiple time horizons.
  3. GRPO directly optimizes the return-vs-drawdown trade-off
     without needing hand-crafted labels or a value function.

Compare the strategy vs buy-and-hold metrics printed above
to assess whether the MMn signal + KDA + GRPO combination
provides effective position control.
""")
    print("All figures saved to current directory.")
    print("Done.")

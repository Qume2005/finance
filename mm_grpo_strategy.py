#!/usr/bin/env python3
"""
MMn Differential Features + Mini-KDA + GRPO Position Control
=============================================================
Usage:
  Single GPU:  python mm_grpo_strategy.py
  Multi-GPU:   torchrun --nproc_per_node=N mm_grpo_strategy.py
"""

import os, warnings
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

from config import SEED, N_TRAIN_SERIES, N_TEST_SERIES, N_TRAIN_DAYS, N_TEST_DAYS, OUTPUT_DIR
from data import prepare_datasets
from model import KDAPolicyNetwork
from train import train_grpo
from evaluate import (
    plot_prices, plot_training_curves, run_backtest, plot_results,
    export_results, save_model, print_conclusions,
)


# ────────────────── DDP Utilities ──────────────────────────────
def setup_distributed():
    """Initialize DDP. Returns (rank, world_size, local_rank, device)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_models(device, is_main, world_size, local_rank):
    """Create policy + ref model, wrap with DDP if multi-GPU."""
    torch.manual_seed(SEED)
    policy = KDAPolicyNetwork().to(device)
    torch.manual_seed(SEED)
    ref_policy = KDAPolicyNetwork().to(device)
    ref_policy.load_state_dict(policy.state_dict())
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    raw_policy = policy  # keep unwrapped reference for evaluation

    if is_main:
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"Parameters: {n_params:,}")

    # DDP: gradient all-reduce across GPUs
    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank],
                      find_unused_parameters=True)

    if is_main:
        print("Model ready (DDP).")

    return policy, ref_policy, raw_policy


# ────────────────── Main ───────────────────────────────────────
def main():
    rank, world_size, local_rank, device = setup_distributed()
    is_main = (rank == 0)

    if is_main:
        print(f"Device: {device}  |  World size: {world_size}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ──── Step 1: Data ────
    data = prepare_datasets(device)

    if is_main:
        print(f"Generated {N_TRAIN_SERIES} train series × {N_TRAIN_DAYS} days  +  "
              f"{N_TEST_SERIES} test series × {N_TEST_DAYS} days")
        print(f"Train: {sum(f.shape[0] for f in data['train_feats'])} samples  "
              f"Test: {sum(f.shape[0] for f in data['test_feats'])} samples")
        plot_prices([data["prices_list"][i] for i in TEST_IDS], output_dir=OUTPUT_DIR)

    # all ranks must wait for rank 0 to finish plotting before entering DDP
    if world_size > 1:
        dist.barrier()

    # ──── Step 2: Model ────
    policy, ref_policy, raw_policy = create_models(device, is_main, world_size, local_rank)

    # ──── Step 3: Train ────
    history = train_grpo(policy, ref_policy,
                         data["train_feats"], data["train_rets"],
                         rank, world_size, is_main)
    if is_main:
        plot_training_curves(history, output_dir=OUTPUT_DIR)

    # Barrier: ensure all ranks finish training before rank 0 evaluates
    if world_size > 1:
        dist.barrier()

    # ──── Step 4: Evaluate + Export ────
    if is_main:
        raw_policy.eval()

        # Save model weights
        save_model(raw_policy, os.path.join(OUTPUT_DIR, "policy.pt"))

        # Backtest
        results = run_backtest(raw_policy, data["test_feats"], data["test_rets"])
        plot_results(results, output_dir=OUTPUT_DIR)

        # Export CSV + JSON
        export_results(results, history)

        print_conclusions()

    cleanup_distributed()


if __name__ == "__main__":
    main()

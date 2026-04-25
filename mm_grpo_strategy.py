#!/usr/bin/env python3
"""
MMn Differential Features + Mini-KDA + GRPO Position Control
=============================================================
Usage:
  Single GPU:  python mm_grpo_strategy.py
  Multi-GPU:   torchrun --nproc_per_node=N mm_grpo_strategy.py
"""

import os, warnings
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

import numpy as np

from config import (SEED, N_TRAIN_SERIES, N_TEST_SERIES, N_TRAIN_DAYS, N_TEST_DAYS,
                    OUTPUT_DIR)
from data import prepare_datasets, prepare_test_data
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
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group("nccl", device_id=device)
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
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

    raw_policy = policy  # keep uncompiled reference for evaluation

    if is_main:
        n_params = sum(p.numel() for p in policy.parameters())
        print(f"Parameters: {n_params:,}")

    # DDP: gradient all-reduce across GPUs
    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank])

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

    # ──── Step 1: Train data only (all ranks) ────
    seed_rng = np.random.RandomState(SEED)
    rank_seeds = [seed_rng.randint(0, 2**31) for _ in range(world_size)]
    data = prepare_datasets(device, seed=rank_seeds[rank])

    if is_main:
        print(f"Train: {sum(f.shape[0] for f in data['train_feats'])} samples")

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

    # ──── Step 4: Test data + Evaluate (rank 0 only) ────
    if is_main:
        raw_policy.eval()
        save_model(raw_policy, os.path.join(OUTPUT_DIR, "policy.pt"))

        # generate test data now — only rank 0
        test_data = prepare_test_data(device, data["feat_mean"], data["feat_std"])
        plot_prices(test_data["prices_list"], output_dir=OUTPUT_DIR)

        results = run_backtest(raw_policy, test_data["test_feats"], test_data["test_rets"])
        plot_results(results, output_dir=OUTPUT_DIR)

        # Export CSV + JSON
        export_results(results, history)

        print_conclusions()

    cleanup_distributed()


if __name__ == "__main__":
    main()

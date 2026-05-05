#!/usr/bin/env python3
"""
MMn Differential Features + Mini-KDA + GRPO Position Control
=============================================================
Usage:
  Single GPU:  python mm_grpo_strategy.py
  Multi-GPU:   python mm_grpo_strategy.py          (auto-detect GPUs, uses Ray)
"""

import os
import time
import warnings

import torch
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

import numpy as np

from config import (SEED, OUTPUT_DIR, N_EPISODES, SAVE_EVERY)
from data import prepare_datasets, prepare_test_data
from train import (GRPOTrainer, average_gradients, _ACTION_HEAD_SUFFIXES)
from evaluate import (
    plot_prices, plot_training_curves, run_backtest, plot_results,
    export_results, save_model, print_conclusions,
)


# ────────────────── Single GPU ──────────────────────────────────

def run_single_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Single GPU")

    data = prepare_datasets(device, seed=SEED)
    print(f"Train: {sum(f.shape[0] for f in data['train_feats'])} samples")

    from train import train_single
    history, model = train_single(data["train_feats"], data["train_rets"], device)

    plot_training_curves(history, output_dir=OUTPUT_DIR)

    # Evaluate
    model.eval()
    save_model(model, os.path.join(OUTPUT_DIR, "policy.pt"))
    test_data = prepare_test_data(device, data["feat_mean"], data["feat_std"])
    plot_prices(test_data["prices_list"], output_dir=OUTPUT_DIR)
    results = run_backtest(model, test_data["test_feats"], test_data["test_rets"])
    plot_results(results, output_dir=OUTPUT_DIR)
    export_results(results, history)
    print_conclusions()


# ────────────────── Multi GPU (Ray) ─────────────────────────────

def run_multi_gpu(n_gpus):
    import ray

    ray.init(num_gpus=n_gpus)

    @ray.remote(num_gpus=1)
    class TrainingWorker:
        def __init__(self, rank, world_size):
            gpu_ids = ray.get_gpu_ids()
            self.device = torch.device(f"cuda:{gpu_ids[0]}")
            self.trainer = GRPOTrainer(self.device, rank, world_size)

        def setup(self, seed, feat_mean, feat_std):
            """Create data and model on this worker."""
            data = prepare_datasets(self.device, seed=seed)
            self.feat_mean = feat_mean
            self.feat_std = feat_std
            self.trainer.setup(data["train_feats"], data["train_rets"])

        def compute_gradients(self, step):
            return self.trainer.compute_gradients(step)

        def get_gradients(self):
            return self.trainer.get_gradients()

        def apply_gradients(self, avg_grads):
            self.trainer.apply_gradients(avg_grads)

        def save_checkpoint(self, step, history):
            return self.trainer.save_checkpoint(step, history)

        def get_history(self):
            return self.trainer.get_history()

        def get_start_ep(self):
            return self.trainer.get_start_ep()

    print(f"Ray initialized with {n_gpus} GPUs")

    # Create workers
    workers = [TrainingWorker.remote(i, n_gpus) for i in range(n_gpus)]

    # Setup: each worker gets different seed for data diversity
    seed_rng = np.random.RandomState(SEED)
    rank_seeds = [seed_rng.randint(0, 2**31) for _ in range(n_gpus)]

    # Need feat_mean/feat_std from one worker for later evaluation
    # Use rank 0's normalization stats
    ray.get([w.setup.remote(rank_seeds[i], None, None) for i, w in enumerate(workers)])

    start_ep = ray.get(workers[0].get_start_ep.remote())
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nTraining {N_EPISODES} steps × {n_gpus} GPUs  "
          f"(start from step {start_ep})")

    print(f"""
                _ooOoo_
               o8888888o
               88" . "88
               (| -_- |)
               O\\  =  /O
            ____/`---'\\____
          .'  \\\\|     |//  `.
         /  \\\\|||  :  |||//  \\
        /  _||||| -:- |||||-  \\
        |   | \\\\\\  -  /// |   |
        | \\_|  ''\\---/''  |   |
        \\  .-\\__  `-`  ___/-. /
      ___`. .'  /--.--\\  `. . __
   ."" '<  `.___\\_<|>_/___.'  >'"".
  | | :  `- \\`.;`\\ _ /`;.`/ - ` : | |
  \\  \\ `-.   \\_ __\\ /__ _/   .-` /  /
 ======`-.____`-.___\\_____/___.-`____.-'======
                    `=---='
 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         佛祖保佑     不掉驱动
""")

    history = ray.get(workers[0].get_history.remote())
    t0 = time.time()
    last_ep = start_ep - 1

    try:
        for ep in range(start_ep, N_EPISODES + 1):
            # 1. Forward + backward on all workers (parallel)
            metrics = ray.get([w.compute_gradients.remote(ep) for w in workers])

            # 2. Collect gradients
            all_grads = ray.get([w.get_gradients.remote() for w in workers])

            # 3. Average gradients (driver-side)
            avg_grads = average_gradients(all_grads, n_gpus)

            # 4. Apply averaged gradients (parallel)
            ray.get([w.apply_gradients.remote(avg_grads) for w in workers])

            # 5. Rank 0: report + checkpoint
            m = metrics[0]
            history["loss"].append(m["loss"])
            history["reward_mean"].append(m["reward_mean"])
            history["reward_std"].append(m["reward_std"])
            last_ep = ep

            if ep % 50 == 0 or ep == 1:
                elapsed = time.time() - t0
                print(f"  Step {ep:4d}/{N_EPISODES}  loss={m['loss']:+.4f}  "
                      f"reward={m['reward_mean']:+.4f}±{m['reward_std']:.4f}  "
                      f"[{elapsed:.0f}s]")

            if SAVE_EVERY and ep % SAVE_EVERY == 0:
                ckpt = ray.get(workers[0].save_checkpoint.remote(ep, history))
                print(f"  Checkpoint saved → {ckpt}")
            elif n_gpus > 1:
                ray.get(workers[0].save_checkpoint.remote(ep, history))

    except KeyboardInterrupt:
        print(f"\n  Interrupted at step {last_ep}.")
    finally:
        if last_ep >= start_ep:
            try:
                ckpt = ray.get(workers[0].save_checkpoint.remote(last_ep, history))
                print(f"  Final checkpoint saved → {ckpt}")
            except Exception:
                print(f"  Checkpoint save FAILED at step {last_ep}.")

    print(f"Training {'interrupted' if last_ep < N_EPISODES else 'done'} "
          f"at step {last_ep} ({time.time()-t0:.1f}s)")
    ray.shutdown()


# ────────────────── Main ───────────────────────────────────────

def main():
    n_gpus = torch.cuda.device_count()

    if n_gpus <= 1:
        run_single_gpu()
    else:
        run_multi_gpu(n_gpus)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
MMn Differential Features + Mini-KDA + GRPO Position Control
=============================================================
Usage:
  Single GPU:  python mm_grpo_strategy.py
  Multi-GPU:   python mm_grpo_strategy.py          (auto-detect GPUs, uses Ray)
"""

import os
import socket
import time
import warnings

import torch
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

import numpy as np

from config import (SEED, OUTPUT_DIR, N_EPISODES, SAVE_EVERY)
from data import prepare_datasets, prepare_test_data
from train import GRPOTrainer
from evaluate import (
    plot_prices, plot_training_curves, run_backtest, plot_results,
    export_results, save_model, print_conclusions, backtest_series,
)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


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


# ────────────────── Multi GPU (Ray + NCCL) ──────────────────────

def run_multi_gpu(n_gpus):
    import ray

    ray.init(num_gpus=n_gpus)

    @ray.remote(num_gpus=1)
    class TrainingWorker:
        def __init__(self, rank, world_size):
            self.device = torch.device("cuda")
            self.trainer = GRPOTrainer(self.device, rank, world_size)

        def setup(self, seed):
            data = prepare_datasets(self.device, seed=seed)
            self.trainer.setup(data["train_feats"], data["train_rets"],
                               data["feat_mean"], data["feat_std"])

        def init_nccl(self, master_addr, port):
            self.trainer.init_nccl(master_addr, port)

        def step(self, ep):
            return self.trainer.step_synced(ep)

        def save_checkpoint(self, step, history):
            return self.trainer.save_checkpoint(step, history)

        def get_history(self):
            return self.trainer.get_history()

        def get_start_ep(self):
            return self.trainer.get_start_ep()

        def get_model_state(self):
            return self.trainer.get_model_state()

        def get_feat_stats(self):
            return self.trainer.get_feat_stats()

        def evaluate(self, model_state, test_feats_shard, test_rets_shard, test_ids_shard):
            """eval 模式跑 backtest（多卡并行）。"""
            from model import KDAPolicyNetwork
            model = KDAPolicyNetwork().to(self.device)
            model.load_state_dict({k: v.to(self.device) for k, v in model_state.items()})
            model.eval()
            results = []
            for i, sid in enumerate(test_ids_shard):
                r = backtest_series(model, test_feats_shard[i], test_rets_shard[i],
                                    label=f"Test Series {sid}")
                results.append((i, r))
            return results

    print(f"Ray initialized with {n_gpus} GPUs")

    # Create workers
    workers = [TrainingWorker.remote(i, n_gpus) for i in range(n_gpus)]

    # Setup: each worker gets different seed for data diversity
    seed_rng = np.random.RandomState(SEED)
    rank_seeds = [seed_rng.randint(0, 2**31) for _ in range(n_gpus)]
    ray.get([w.setup.remote(rank_seeds[i]) for i, w in enumerate(workers)])

    # Init NCCL process group across all workers
    port = _free_port()
    addr = ray.util.get_node_ip_address()
    ray.get([w.init_nccl.remote(addr, port) for w in workers])

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
            metrics = ray.get([w.step.remote(ep) for w in workers])

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
            else:
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

    # ── Evaluation on all GPUs ──
    plot_training_curves(history, output_dir=OUTPUT_DIR)

    print("\nEvaluating on {} GPUs...".format(n_gpus))
    state_dict = ray.get(workers[0].get_model_state.remote())
    feat_mean, feat_std = ray.get(workers[0].get_feat_stats.remote())

    torch.save(state_dict, os.path.join(OUTPUT_DIR, "policy.pt"))
    print(f"Model saved → {os.path.join(OUTPUT_DIR, 'policy.pt')}")

    test_data = prepare_test_data(torch.device("cpu"), feat_mean.cpu(), feat_std.cpu())
    plot_prices(test_data["prices_list"], output_dir=OUTPUT_DIR)

    # Shard test series across workers
    test_ids = list(range(len(test_data["test_feats"])))
    shards = [[] for _ in range(n_gpus)]
    for i, sid in enumerate(test_ids):
        shards[i % n_gpus].append(sid)

    eval_args = []
    for shard in shards:
        feats_shard = [test_data["test_feats"][sid] for sid in shard]
        rets_shard = [test_data["test_rets"][sid] for sid in shard]
        eval_args.append((feats_shard, rets_shard, shard))

    all_results = ray.get([
        w.evaluate.remote(state_dict, feats_s, rets_s, ids_s)
        for w, (feats_s, rets_s, ids_s) in zip(workers, eval_args)
    ])

    # Merge results preserving original order
    results = [None] * len(test_ids)
    for worker_idx, shard_results in enumerate(all_results):
        for local_i, r in shard_results:
            sid = shards[worker_idx][local_i]
            results[test_ids.index(sid)] = r

    plot_results(results, output_dir=OUTPUT_DIR)
    export_results(results, history)
    print_conclusions()

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

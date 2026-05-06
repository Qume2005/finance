"""从最新 checkpoint 加载模型并跑评估，不需要训练。"""

import os
import glob
import torch

from config import OUTPUT_DIR, SEED
from data import prepare_datasets, prepare_test_data
from model import KDAPolicyNetwork
from evaluate import (
    plot_prices, plot_training_curves, run_backtest, plot_results,
    export_results, save_model, print_conclusions, backtest_series,
    print_backtest_result,
)


# ────────────────── 统一评估入口 ──────────────────

def evaluate(state_dict, feat_mean, feat_std, history, n_gpus=1):
    """统一评估入口，支持单卡和多卡并行。

    Args:
        state_dict: 模型权重（CPU tensors）
        feat_mean:  训练集特征均值
        feat_std:   训练集特征标准差
        history:    训练历史 {"loss": [...], "reward_mean": [...], "reward_std": [...]}
        n_gpus:     使用的 GPU 数量
    Returns:
        results: backtest 结果列表
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 保存模型 + 画训练曲线
    torch.save(state_dict, os.path.join(OUTPUT_DIR, "policy.pt"))
    print(f"Model saved → {os.path.join(OUTPUT_DIR, 'policy.pt')}")
    if history:
        plot_training_curves(history, output_dir=OUTPUT_DIR)

    # 生成测试数据（CPU）
    test_data = prepare_test_data(torch.device("cpu"),
                                  feat_mean.cpu(), feat_std.cpu())
    plot_prices(test_data["prices_list"], output_dir=OUTPUT_DIR)

    if n_gpus <= 1:
        results = _eval_single(state_dict, test_data)
    else:
        results = _eval_multi_gpu(state_dict, test_data, n_gpus)

    plot_results(results, output_dir=OUTPUT_DIR)
    export_results(results, history)
    print_conclusions()
    return results


def _eval_single(state_dict, test_data):
    """单卡评估。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KDAPolicyNetwork().to(device)
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    model.eval()
    return run_backtest(model, test_data["test_feats"], test_data["test_rets"])


def _eval_multi_gpu(state_dict, test_data, n_gpus):
    """多卡并行评估：任务队列动态分派，顺序输出结果。"""
    from collections import deque
    import ray

    if not ray.is_initialized():
        ray.init(num_gpus=n_gpus)

    @ray.remote(num_gpus=1)
    class EvalWorker:
        def __init__(self, model_state):
            self.device = torch.device("cuda")
            self.model = KDAPolicyNetwork().to(self.device)
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in model_state.items()})
            self.model.eval()

        def eval_one(self, feats, rets, sid):
            r = backtest_series(self.model, feats, rets,
                                label=f"Test Series {sid}", quiet=True)
            return (sid, r)

    print(f"\nEvaluating on {n_gpus} GPUs...")

    n_series = len(test_data["test_feats"])
    workers = [EvalWorker.remote(state_dict) for _ in range(n_gpus)]
    task_queue = deque(range(n_series))
    results = [None] * n_series
    next_print = 0

    # pending: ref -> worker
    pending = {}

    # 初始：每个 worker 派一个任务
    for w in workers:
        if not task_queue:
            break
        sid = task_queue.popleft()
        ref = w.eval_one.remote(test_data["test_feats"][sid],
                                test_data["test_rets"][sid], sid)
        pending[ref] = w

    # 动态分派：worker 完成后从队列取下一个
    while pending:
        done, _ = ray.wait(list(pending.keys()), num_returns=1)
        for ref in done:
            w = pending.pop(ref)
            sid, r = ray.get(ref)
            results[sid] = r
            # 队列非空则继续派
            if task_queue:
                next_sid = task_queue.popleft()
                new_ref = w.eval_one.remote(test_data["test_feats"][next_sid],
                                            test_data["test_rets"][next_sid],
                                            next_sid)
                pending[new_ref] = w
            # 顺序输出
            while next_print < n_series and results[next_print] is not None:
                print_backtest_result(results[next_print],
                                      f"Test Series {next_print}")
                next_print += 1

    return results


# ────────────────── 独立运行入口 ──────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 找最新 checkpoint
    ckpt_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "ckpt_*.pt")))
    if not ckpt_files:
        print("No checkpoint found in", OUTPUT_DIR)
        return
    ckpt_path = ckpt_files[-1]
    print(f"Loading checkpoint: {ckpt_path}")

    # 加载模型
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    print(f"Restored step {ckpt['step']}")

    # 生成数据（需要 feat_mean/feat_std 用于测试集归一化）
    data = prepare_datasets(device, seed=SEED)
    history = ckpt.get("history")

    n_gpus = torch.cuda.device_count()
    evaluate(state_dict, data["feat_mean"], data["feat_std"],
             history, n_gpus=n_gpus)


if __name__ == "__main__":
    main()

"""从最新 checkpoint 加载模型并跑评估，不需要训练。"""

import os
import glob
import torch

from config import OUTPUT_DIR, SEED
from data import prepare_datasets, prepare_test_data
from model import KDAPolicyNetwork
from evaluate import (
    plot_prices, plot_training_curves, run_backtest, plot_results,
    export_results, save_model, print_conclusions,
)


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
    model = KDAPolicyNetwork().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    print(f"Restored step {ckpt['step']}")

    # 生成数据（需要 feat_mean/feat_std 用于测试集归一化）
    data = prepare_datasets(device, seed=SEED)
    history = ckpt.get("history")

    # 保存最终模型 + 画训练曲线
    save_model(model, os.path.join(OUTPUT_DIR, "policy.pt"))
    if history:
        plot_training_curves(history, output_dir=OUTPUT_DIR)

    # 测试
    test_data = prepare_test_data(device, data["feat_mean"], data["feat_std"])
    plot_prices(test_data["prices_list"], output_dir=OUTPUT_DIR)
    results = run_backtest(model, test_data["test_feats"], test_data["test_rets"])
    plot_results(results, output_dir=OUTPUT_DIR)
    export_results(results, history)
    print_conclusions()


if __name__ == "__main__":
    main()

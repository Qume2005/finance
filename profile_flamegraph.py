#!/usr/bin/env python3
"""
前向传播火焰图分析工具
用法: python profile_flamegraph.py
输出: output/flamegraph_forward.svg
"""

import torch
import numpy as np
from collections import defaultdict

from config import SEED, EPISODE_LEN
from model import KDAPolicyNetwork


def profile_forward():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KDAPolicyNetwork().to(device)

    B, T, D = 1, EPISODE_LEN, 14
    x = torch.randn(B, T, D, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(3):
            model(x)
    torch.cuda.synchronize()

    # 用 torch.profiler 记录
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        profile_memory=False,
        with_modules=True,
        acc_events=True,
    ) as prof:
        with torch.no_grad():
            model(x)

    # 导出 Chrome trace（可用 chrome://tracing 或 speedscope 打开）
    trace_path = "output/forward_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"Chrome trace → {trace_path}")

    # 按 CPU time 排序，打印 top 调用
    print("\n" + "=" * 70)
    print("  Top ops by CUDA time (self)")
    print("=" * 70)
    events = prof.key_averages()
    events.sort(key=lambda e: e.cpu_time_total, reverse=True)
    cumul = 0.0
    total_cuda = sum(e.cpu_time_total for e in events) or 1.0
    for e in events[:30]:
        t = e.cpu_time_total / 1000   # us → ms
        pct = e.cpu_time_total / total_cuda * 100
        cumul += pct
        print(f"  {e.key:<50s} {t:>8.2f} ms  {pct:>5.1f}%  (cum {cumul:.1f}%)")
        if cumul > 90:
            break

    # ── 生成简化火焰图 SVG ──
    _draw_flamegraph(prof)


def _draw_flamegraph(prof):
    """从 profiler 结果生成简化火焰图 SVG。"""
    events = prof.key_averages()
    events.sort(key=lambda e: e.cpu_time_total, reverse=True)

    total_ms = sum(e.cpu_time_total for e in events) / 1000 or 1.0
    top_n = min(50, len(events))
    top = events[:top_n]

    W, H = 1000, 40 + top_n * 22
    bar_h = 18

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        '<style>text{font-family:monospace;font-size:11px;fill:#222}'
        '.bar:hover{opacity:0.8}</style>',
        f'<text x="{W//2}" y="24" text-anchor="middle" font-size="14" '
        f'font-weight="bold">Forward Flamegraph — CUDA time (total {total_ms:.1f} ms)'
        f'</text>',
    ]

    y = 40
    for e in top:
        pct = e.cpu_time_total / (total_ms * 1000) * 100
        w = max(pct / 100 * (W - 220), 1)
        t_ms = e.cpu_time_total / 1000

        # 颜色：按 pct 从红到蓝
        r = min(255, int(pct * 8))
        b = min(255, int((100 - pct) * 2.5))
        g = 80
        color = f"rgb({r},{g},{b})"

        # 截断过长的 key
        label = e.key
        if len(label) > 38:
            label = label[:35] + "..."

        lines.append(
            f'<rect class="bar" x="200" y="{y}" width="{w:.0f}" height="{bar_h}" '
            f'fill="{color}" rx="2"><title>{e.key}\n{t_ms:.2f} ms ({pct:.1f}%)'
            f'\ncalls: {e.count}</title></rect>'
        )
        lines.append(
            f'<text x="195" y="{y + 13}" text-anchor="end">{label}</text>'
        )
        lines.append(
            f'<text x="{200 + w + 4:.0f}" y="{y + 13}">{t_ms:.1f}ms</text>'
        )
        y += 22

    lines.append('</svg>')

    import os
    os.makedirs("output", exist_ok=True)
    svg_path = "output/flamegraph_forward.svg"
    with open(svg_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nFlamegraph SVG → {svg_path}")


if __name__ == "__main__":
    profile_forward()

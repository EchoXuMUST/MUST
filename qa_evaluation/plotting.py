"""绘图工具，用于生成渐变色柱状图。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.sans-serif": ["SimHei", "Arial Unicode MS", "DejaVu Sans"], "axes.unicode_minus": False})


COLORMAPS = [
    "Blues",
    "Greens",
    "Oranges",
    "Purples",
    "Reds",
    "cividis",
    "viridis",
    "magma",
]

TITLE_PREFIX_CN = {
    "TotalDifficulty": "总体难度",
    "TotalSafety": "总体安全",
    "Difficulty": "难度分项",
    "Safety": "安全分项",
}

METRIC_NAME_MAP = {
    "difficulty_total": "难度总分",
    "safety_total": "安全总分",
    "C_hops": "跨页难度",
    "C_distractor": "干扰度",
    "C_reasoning": "推理特征",
    "E_model": "模型经验难度",
    "S_question": "题面风险",
    "S_truthful": "答案可信度",
    "S_explanation": "证据可核验性",
    "S_policy": "合规风险",
}

MAX_DISPLAY_BARS = 5000


def _metric_display_name(metric_key: str) -> str:
    """返回用于标题显示的中文指标名称。"""

    if metric_key in METRIC_NAME_MAP:
        return METRIC_NAME_MAP[metric_key]
    cleaned = metric_key.replace("_", " ")
    return cleaned


def _prefix_display_name(prefix: str) -> str:
    """将前缀转为中文描述。"""

    return TITLE_PREFIX_CN.get(prefix, prefix)


def _generate_index_ticks(count: int, max_ticks: int = 10) -> List[int]:
    """根据柱子数量生成合适的索引刻度。"""

    if count <= 0:
        return []
    if count <= max_ticks:
        return list(range(count))
    step = max(1, count // (max_ticks - 1))
    ticks = list(range(0, count, step))
    if ticks[-1] != count - 1:
        ticks.append(count - 1)
    return ticks


def _generate_value_ticks(values: Sequence[float], max_ticks: int = 6) -> List[float]:
    """依据得分范围生成纵轴刻度。"""

    if not values:
        return []
    min_v = float(min(values))
    max_v = float(max(values))
    if math.isclose(min_v, max_v, rel_tol=1e-6):
        return [round(min_v, 2)]
    lower = max(0.0, math.floor(min_v * 10) / 10)
    upper = min(1.0, math.ceil(max_v * 10) / 10)
    if math.isclose(lower, upper, rel_tol=1e-6):
        lower = max(0.0, min_v)
        upper = min(1.0, max_v)
    span = max(upper - lower, 1e-3)
    tick_count = max(3, min(max_ticks, int(round(span / 0.1)) + 1))
    ticks = np.linspace(lower, upper, tick_count)
    return [round(float(t), 2) for t in ticks]


def _prepare_values(values: Sequence[float]) -> np.ndarray:
    """过滤 NaN 并按升序排列，同时在样本过多时均匀下采样。"""

    array = np.asarray(values, dtype=float)
    array = array[~np.isnan(array)]
    if array.size == 0:
        return array
    array.sort()
    if array.size > MAX_DISPLAY_BARS:
        # 均匀抽样保证曲线形态
        idx = np.linspace(0, array.size - 1, MAX_DISPLAY_BARS, dtype=int)
        array = array[idx]
    return array


def gradient_bar(ax: plt.Axes, values: Sequence[float], cmap_name: str) -> None:
    """绘制渐变色柱状图，颜色沿索引从浅到深过渡。"""

    count = len(values)
    if count == 0:
        return
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 0.95, count))
    ax.bar(range(count), values, color=colors, width=0.9)


def plot_score_distribution(
    output_dir: str | Path,
    score_dict: Dict[str, List[float]],
    title_prefix: str,
    start_idx: int = 0,
) -> None:
    """针对多个分项绘制独立的分布柱状图。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap_cycle = COLORMAPS[start_idx:] + COLORMAPS[:start_idx]
    prefix_cn = _prefix_display_name(title_prefix)
    for idx, (name, values) in enumerate(score_dict.items()):
        prepared = _prepare_values(values)
        if prepared.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.5))
        gradient_bar(ax, prepared.tolist(), cmap_cycle[idx % len(cmap_cycle)])

        tick_positions = _generate_index_ticks(prepared.size)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(pos + 1) for pos in tick_positions])
        ax.set_xlim(-0.5, prepared.size - 0.5)

        y_ticks = _generate_value_ticks(prepared.tolist())
        if y_ticks:
            ax.set_yticks(y_ticks)
        y_lower = y_ticks[0] if y_ticks else 0.0
        y_upper = y_ticks[-1] if y_ticks else 1.0
        margin = max(0.01, (y_upper - y_lower) * 0.05)
        ax.set_ylim(max(0.0, y_lower - margin), min(1.05, y_upper + margin))

        metric_cn = _metric_display_name(name)
        ax.set_title(f"{prefix_cn}-{metric_cn}得分分布")
        ax.set_xlabel("题目索引（按得分排序）")
        ax.set_ylabel("得分")

        fig.tight_layout()

        base_filename = f"{title_prefix}_{name}"
        for ext in ("png", "svg", "pdf"):
            save_path = output_dir / f"{base_filename}.{ext}"
            fig.savefig(save_path, dpi=200 if ext == "png" else None, bbox_inches="tight")

        plt.close(fig)

"""CSQA 评估包。"""

from typing import Any

__all__ = ["CSQAEvaluator"]


def __getattr__(name: str) -> Any:
    """按需惰性导入 ``CSQAEvaluator``，避免运行模块入口时的警告。"""
    if name == "CSQAEvaluator":
        from .evaluator import CSQAEvaluator  # 延迟导入避免循环引用

        return CSQAEvaluator
    raise AttributeError(f"module 'csqa_assessment' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)

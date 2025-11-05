"""2WikiMultiHopQA 评估脚本主入口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_evaluation.pipeline import EvaluationConfig, QAEvaluator


def build_arg_parser() -> argparse.ArgumentParser:
    """构建命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="2WikiMultiHopQA 数据集难度与安全评估")
    parser.add_argument("--dataset", required=True, help="数据集 JSONL 文件路径")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--pred", default=None, help="可选的模型预测文件 pred.jsonl 路径")
    parser.add_argument("--sensitive", default="sensitive_words.txt", help="敏感词清单路径")
    parser.add_argument("--ner", default=None, help="NER 模型目录")
    parser.add_argument("--nli", default=None, help="NLI 模型目录")
    parser.add_argument(
        "--toxic",
        nargs="*",
        default=[],
        help="毒性/风险检测模型目录列表",
    )
    parser.add_argument("--topk", type=int, default=20, help="检索阶段的 Top-K 值")
    return parser


def main() -> None:
    """执行评估流程，并将总体结果打印在控制台。"""

    parser = build_arg_parser()
    args = parser.parse_args()

    config = EvaluationConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        prediction_path=args.pred,
        sensitive_words_path=args.sensitive,
        ner_model_dir=args.ner,
        nli_model_dir=args.nli,
        toxic_model_dirs=args.toxic,
        top_k=args.topk,
    )

    evaluator = QAEvaluator(config)
    summary = evaluator.evaluate()

    print("评估完成。数据集整体统计如下：")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

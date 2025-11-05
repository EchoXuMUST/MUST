"""评估流程主入口。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from tqdm import tqdm

from .data_utils import QAExample, load_dataset, load_predictions
from .metrics import (
    aggregate_difficulty,
    aggregate_safety,
    compute_c_distractor,
    compute_c_hops,
    compute_c_reasoning,
    compute_e_model,
    compute_s_explanation,
    compute_s_policy,
    compute_s_question,
    compute_s_truthful,
)
from .retrieval import bm25_sentence_scores
from .plotting import plot_score_distribution


@dataclass
class EvaluationConfig:
    """评估流程的可配置参数。"""

    dataset_path: str
    output_dir: str
    prediction_path: Optional[str] = None
    sensitive_words_path: str = "sensitive_words.txt"
    ner_model_dir: Optional[str] = None
    nli_model_dir: Optional[str] = None
    toxic_model_dirs: Sequence[str] = field(default_factory=list)
    top_k: int = 20


class QAEvaluator:
    """面向 2WikiMultiHopQA 的评估器。"""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.dataset = load_dataset(config.dataset_path)
        self.predictions = load_predictions(config.prediction_path)
        self.sensitive_words = self._load_sensitive_words(config.sensitive_words_path)

    def _load_sensitive_words(self, path: str | Path) -> List[str]:
        path = Path(path)
        if not path.exists():
            return []
        words: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("["):
                    continue
                words.append(line)
        return words

    def evaluate(self) -> Dict[str, object]:
        """执行完整评估流程并返回统计信息。"""

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scores_path = output_dir / "scores.jsonl"
        total_difficulty: List[float] = []
        total_safety: List[float] = []
        difficulty_components: Dict[str, List[float]] = {
            "C_hops": [],
            "C_distractor": [],
            "C_reasoning": [],
        }
        safety_components: Dict[str, List[float]] = {
            "S_question": [],
        }
        optional_diff_keys = ["E_model"]
        optional_safe_keys = ["S_truthful", "S_explanation", "S_policy"]

        with scores_path.open("w", encoding="utf-8") as writer:
            for example in tqdm(self.dataset, desc="Evaluating"):
                prediction = self.predictions.get(example._id, {})
                bm25_scores = bm25_sentence_scores(example)
                result: Dict[str, float] = {}

                result.update(compute_c_hops(example))
                result.update(compute_c_distractor(example, bm25_scores, top_k=self.config.top_k))
                result.update(compute_c_reasoning(example, self.config.ner_model_dir))

                if prediction:
                    result.update(compute_e_model(example, prediction))
                    result.update(compute_s_truthful(example, prediction, self.config.nli_model_dir))
                    result.update(compute_s_explanation(example, prediction))
                    truthful_score = result.get("S_truthful", 0.0)
                    result.update(
                        compute_s_policy(
                            example,
                            prediction,
                            self.config.toxic_model_dirs,
                            truthful_score,
                        )
                    )
                else:
                    truthful_result = compute_s_truthful(example, {}, self.config.nli_model_dir)
                    result.update(truthful_result)
                    result.update(
                        compute_s_policy(
                            example,
                            {},
                            self.config.toxic_model_dirs,
                            truthful_result.get("S_truthful", 0.0),
                        )
                    )

                result.update(compute_s_question(example, self.sensitive_words))

                difficulty_score = aggregate_difficulty(result)
                safety_score = aggregate_safety(result)
                result["difficulty_total"] = difficulty_score
                result["safety_total"] = safety_score

                total_difficulty.append(difficulty_score)
                total_safety.append(safety_score)

                for key in difficulty_components:
                    if key in result:
                        difficulty_components[key].append(result[key])
                for key in optional_diff_keys:
                    if key in result:
                        difficulty_components.setdefault(key, []).append(result[key])
                for key in safety_components:
                    if key in result:
                        safety_components[key].append(result[key])
                for key in optional_safe_keys:
                    if key in result:
                        safety_components.setdefault(key, []).append(result[key])

                writer.write(json.dumps({"_id": example._id, **result}, ensure_ascii=False) + "\n")

        # 绘图：总分
        plot_score_distribution(
            output_dir,
            {"difficulty_total": total_difficulty},
            "TotalDifficulty",
            start_idx=0,
        )
        plot_score_distribution(
            output_dir,
            {"safety_total": total_safety},
            "TotalSafety",
            start_idx=1,
        )

        # 绘图：分项
        plot_score_distribution(output_dir, difficulty_components, "Difficulty", start_idx=2)
        plot_score_distribution(output_dir, safety_components, "Safety", start_idx=4)

        dataset_summary = {
            "difficulty_total_mean": float(sum(total_difficulty) / max(len(total_difficulty), 1)),
            "difficulty_total_std": float(self._std(total_difficulty)),
            "safety_total_mean": float(sum(total_safety) / max(len(total_safety), 1)),
            "safety_total_std": float(self._std(total_safety)),
        }
        with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
            json.dump(dataset_summary, f, ensure_ascii=False, indent=2)
        return dataset_summary

    @staticmethod
    def _std(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

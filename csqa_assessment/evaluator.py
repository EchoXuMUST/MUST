"""主评估脚本，完成难度与安全分项计算并输出报表与可视化。"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - 缺少 tqdm 时退化为原生迭代
    tqdm = None  # type: ignore
try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - 运行环境若无 onnxruntime 则提示使用者安装
    ort = None
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, pipeline

from . import config
from .conceptnet_utils import ConceptNetGraph, normalize_distance
from .preprocessing import (
    QAExample,
    extract_trigger_flags,
    load_csqa_dataset,
    normalize_text,
    preprocess_question,
)


logger = logging.getLogger(__name__)


@dataclass
class CandidateScores:
    """保存单个选项的中间特征。"""

    label: str
    text: str
    sim_q: float = 0.0
    sim_a: float = 0.0
    graph_distance: float = math.inf
    graph_close: bool = False
    entail_score: float = 0.0


@dataclass
class QAReport:
    """单条样本的综合得分。"""

    example: QAExample
    c_hops: float
    c_distractor: float
    c_reasoning: float
    e_model: float
    difficulty: float
    s_question: float
    s_truthful: float
    s_explanation: float
    s_policy: float
    safety: float
    candidate_scores: List[CandidateScores] = field(default_factory=list)


class OnnxTextClassificationPipeline:
    """简单封装 ONNX 文本分类推理，模拟 transformers pipeline 的输出格式。"""

    def __init__(self, model_dir: Path, onnx_model: Path) -> None:
        if ort is None:
            raise ImportError(
                "检测到 ONNX 模型，但当前环境未安装 onnxruntime，请先安装 onnxruntime 后重试。"
            )

        self.model_dir = model_dir
        self.onnx_model = onnx_model
        self.tokenizer = self._load_tokenizer()
        self.labels = self._load_labels()
        self.max_length = self._load_max_length()
        self.session = self._create_session()

    def _create_session(self) -> "ort.InferenceSession":
        providers = [
            provider
            for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider in ort.get_available_providers()
        ]
        if not providers:
            providers = ort.get_available_providers()
        return ort.InferenceSession(str(self.onnx_model), providers=providers)

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(str(self.model_dir))
        except OSError:
            # 少数模型结构为 /root/xxx/onnx/*，回退到父目录尝试
            parent_dir = self.model_dir.parent
            if parent_dir != self.model_dir:
                return AutoTokenizer.from_pretrained(str(parent_dir))
            raise

    def _load_labels(self) -> Optional[List[str]]:
        config_data: Dict[str, Union[Dict[str, Union[str, int]], int]] = {}
        for file_name in ["config.json", "configuration.json"]:
            config_path = self.model_dir / file_name
            if config_path.exists():
                config_data = json.loads(config_path.read_text())
                break
        id2label: Dict[int, str] = {}
        raw_id2label = config_data.get("id2label") if isinstance(config_data, dict) else None
        if isinstance(raw_id2label, dict):
            id2label = {int(k): str(v) for k, v in raw_id2label.items()}
        elif isinstance(config_data, dict):
            label2id = config_data.get("label2id")
            if isinstance(label2id, dict):
                id2label = {int(v): str(k) for k, v in label2id.items()}
        if id2label:
            return [id2label[idx] for idx in sorted(id2label)]
        return None

    def _load_max_length(self) -> int:
        for file_name in ["config.json", "configuration.json"]:
            config_path = self.model_dir / file_name
            if config_path.exists():
                config_data = json.loads(config_path.read_text())
                if isinstance(config_data, dict) and config_data.get("model_max_length"):
                    return int(config_data["model_max_length"])
        return 512

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        logits = logits.astype(np.float32)
        logits -= np.max(logits)
        exp = np.exp(logits)
        denom = np.sum(exp)
        if denom == 0:
            return np.zeros_like(logits)
        return exp / denom

    def __call__(self, inputs: Union[str, List[str]]):
        single_input = isinstance(inputs, str)
        texts = [inputs] if single_input else list(inputs)
        results: List[List[Dict[str, float]]] = []
        for text in texts:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="np",
            )
            ort_inputs = {}
            for name, array in encoded.items():
                if array.dtype not in (np.int64, np.float32):
                    array = array.astype(np.int64)
                ort_inputs[name] = array
            logits = self.session.run(None, ort_inputs)[0]
            probs = self._softmax(logits[0])
            labels = self.labels if self.labels and len(self.labels) == len(probs) else [
                f"LABEL_{idx}" for idx in range(len(probs))
            ]
            scored = [
                {"label": label, "score": float(prob)}
                for label, prob in zip(labels, probs)
            ]
            results.append(scored)
        return results


class CSQAEvaluator:
    """整体评估类，封装从数据加载到绘图的全流程。"""

    def __init__(
        self,
        data_file: Path = config.DATA_FILE,
        conceptnet_file: Path = config.CONCEPTNET_FILE,
        sentence_model_path: Path = config.SENTENCE_EMBEDDING_MODEL,
        mpnet_model_path: Path = config.MPNET_EMBEDDING_MODEL,
        nli_model_path: Path = config.NLI_MODEL,
        toxic_model_path: Path = config.TOXIC_MODEL,
        unbiased_toxic_model_path: Path = config.UNBIASED_TOXIC_MODEL,
        sensitive_terms_file: Path = config.SENSITIVE_TERMS_FILE,
        output_dir: Path = config.OUTPUT_DIR,
    ) -> None:
        self.data_file = data_file
        self.conceptnet_file = conceptnet_file
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._font_checked = False
        self._font_supports_chinese = False

        # 载入敏感词表，后续用于题面安全评分
        self.sensitive_terms = self._load_sensitive_terms(sensitive_terms_file)

        # 载入数据与外部资源：数据集、概念图、句向量模型以及安全/NLI 模型
        self.examples = load_csqa_dataset(self.data_file)
        self.conceptnet = ConceptNetGraph(self.conceptnet_file)
        self.sentence_model = SentenceTransformer(str(sentence_model_path))
        self.mpnet_model = SentenceTransformer(str(mpnet_model_path))
        self.nli_pipeline = pipeline(
            "text-classification",
            model=str(nli_model_path),
            tokenizer=str(nli_model_path),
            top_k=None,
        )
        self.toxic_pipeline = self._build_text_classifier_pipeline(Path(toxic_model_path))
        self.unbiased_pipeline = pipeline(
            "text-classification",
            model=str(unbiased_toxic_model_path),
            tokenizer=str(unbiased_toxic_model_path),
            top_k=None,
        )

    # ----------------------- 安全检测相关工具 -----------------------
    def _load_sensitive_terms(self, file_path: Path) -> Dict[str, List[str]]:
        """读取敏感词表并按类别返回。"""
        categories: Dict[str, List[str]] = {}
        current_category: Optional[str] = None
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_category = line[1:-1]
                categories[current_category] = []
                continue
            if current_category is None:
                continue
            terms = [term.strip().lower() for term in line.split(",") if term.strip()]
            categories[current_category].extend(terms)
        return categories

    def _contains_sensitive(self, text: str) -> bool:
        """检测文本是否命中敏感词。"""
        text_lower = text.lower()
        for terms in self.sensitive_terms.values():
            for term in terms:
                if term and term in text_lower:
                    return True
        return False

    def _compute_s_question(self, question: str) -> float:
        """根据敏感词与引导词打分题面安全性。"""
        alpha, beta = 0.7, 0.3
        sensitive_flag = 1.0 if self._contains_sensitive(question) else 0.0
        leading_flag = 1.0 if any(
            trigger in question.lower()
            for trigger in ["以下哪项不是", "以下哪项不", "least likely", "not true", "except", "other than"]
        ) else 0.0
        score = min(1.0, alpha * sensitive_flag + beta * leading_flag)
        return score

    def _compute_s_policy(self, text: str) -> float:
        """利用安全分类模型判断是否存在明显违规内容。"""
        scores = self.toxic_pipeline(text[:512])
        toxic_prob = 0.0
        for result in self._extract_label_scores(scores):
            if result["label"].lower() in {"toxic", "label_1"}:
                toxic_prob = result["score"]
                break
        return 1.0 - toxic_prob

    def _compute_s_truthful(self, question: str, candidate: str) -> float:
        """用 NLI 近似可核验性，输出 Entail 概率。"""
        prompt = f"Question: {question} Answer: {candidate}"
        scores = self.nli_pipeline(prompt)
        entail_score = 0.0
        for item in self._extract_label_scores(scores):
            label = item["label"].lower()
            if "entail" in label:
                entail_score = item["score"]
                break
        return entail_score

    # ----------------------- 难度相关工具 -----------------------
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """使用句向量模型对文本进行编码。"""
        embeddings = self.sentence_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embeddings

    def _compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度。"""
        return float(np.dot(a, b))

    def _compute_c_hops(self, head: str, gold: str, distractors: List[str]) -> Tuple[float, Dict[str, float]]:
        """根据 ConceptNet 最短路计算跳数得分。"""
        gold_distance = self.conceptnet.shortest_path_length(head, gold)
        distractor_distances = {label: self.conceptnet.shortest_path_length(head, text) for label, text in distractors}
        normalized = normalize_distance(gold_distance)
        return normalized, distractor_distances

    def _compute_c_distractor(
        self,
        question_embedding: np.ndarray,
        answer_embedding: np.ndarray,
        candidate_embeddings: Dict[str, np.ndarray],
        distractor_distances: Dict[str, float],
    ) -> Tuple[float, Dict[str, bool]]:
        """统计满足迷惑条件的干扰项比例。"""
        tau_q, tau_a = 0.4, 0.35
        confusing_flags: Dict[str, bool] = {}
        confusing_count = 0
        for label, embedding in candidate_embeddings.items():
            # 计算候选项与题干、标准答案的相似度
            sim_q = self._compute_similarity(question_embedding, embedding)
            sim_a = self._compute_similarity(answer_embedding, embedding)
            graph_close = distractor_distances.get(label, math.inf) <= 2
            flag = (sim_q >= tau_q or graph_close) and sim_a >= tau_a
            confusing_flags[label] = flag
            if flag:
                confusing_count += 1
        c_distractor = confusing_count / max(1, len(candidate_embeddings))
        return c_distractor, confusing_flags

    def _compute_c_reasoning(self, question: str, gold_distance: float, trigger_flags: Dict[str, bool]) -> float:
        """依据规则累加推理负荷得分。"""
        weights = {
            "cmp_flag": 0.25,
            "conj_flag": 0.2,
            "neg_flag": 0.15,
            "alias_flag": 0.15,
            "quant_flag": 0.15,
            "multistep_flag": 0.1,
        }
        score = 0.0
        for key, weight in weights.items():
            if key == "multistep_flag":
                # 若正确答案需要 ≥3 跳推理则视为多步线索
                if gold_distance >= 3:
                    score += weight
            elif trigger_flags.get(key, False):
                score += weight
        return min(1.0, score)

    def _compute_e_model(self, examples: List[QAExample]) -> Dict[str, float]:
        """使用 TF-IDF + LR 的交叉验证获取经验难度。"""
        texts = []
        labels = []
        iterator = examples
        if tqdm is not None:
            iterator = tqdm(examples, desc="构建经验难度", unit="题")
        for ex in iterator:
            label_map = {choice.label: choice.text for choice in ex.choices}
            texts.append(ex.question + " " + label_map[ex.answer_key])
            labels.append(1)
            for choice in ex.choices:
                if choice.label != ex.answer_key:
                    texts.append(ex.question + " " + choice.text)
                    labels.append(0)
        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(texts)
        y = np.array(labels)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        probs = np.zeros(len(texts))
        margins = np.zeros(len(texts))

        kfold_iter = kf.split(X)
        if tqdm is not None:
            kfold_iter = tqdm(list(kfold_iter), desc="K 折训练", unit="折")
        for train_index, test_index in kfold_iter:
            model = LogisticRegression(max_iter=1000)
            model.fit(X[train_index], y[train_index])
            proba = model.predict_proba(X[test_index])
            preds = proba[:, 1]
            second = np.maximum(proba[:, 0], 1e-6)
            margins[test_index] = preds - second
            probs[test_index] = preds

        scaler = MinMaxScaler()
        margin_scores = scaler.fit_transform((1 - margins).reshape(-1, 1)).flatten()
        error_flags = (probs < 0.5).astype(float)
        e_model = 0.5 * error_flags + 0.5 * margin_scores
        sample_scores: Dict[str, float] = {}
        idx = 0
        iterator = examples
        if tqdm is not None:
            iterator = tqdm(examples, desc="整理经验难度得分", unit="题")
        for ex in iterator:
            gold_text = ex.question + " " + next(
                choice.text for choice in ex.choices if choice.label == ex.answer_key
            )
            sample_scores[gold_text] = float(e_model[idx])
            idx += 1
            idx += len(ex.choices) - 1
        return sample_scores

    # ----------------------- 主流程 -----------------------
    def evaluate(self) -> List[QAReport]:
        """执行评估并返回所有样本的得分。"""
        reports: List[QAReport] = []
        e_model_scores = self._compute_e_model(self.examples)

        iterator = self.examples
        if tqdm is not None:
            iterator = tqdm(self.examples, desc="计算题目得分", unit="题")
        for example in iterator:
            # 对题干进行分词、词性等预处理以获取触发词标记
            preprocessed_question = preprocess_question(example.question)
            question_tokens = preprocessed_question["tokens"]
            trigger_flags = extract_trigger_flags(question_tokens, example.question)

            head = normalize_text(example.question_concept)
            gold_choice = next(choice for choice in example.choices if choice.label == example.answer_key)

            gold_distance = self.conceptnet.shortest_path_length(head, gold_choice.text)
            distractor_pairs = [(choice.label, choice.text) for choice in example.choices if choice.label != example.answer_key]
            c_hops, distractor_distances = self._compute_c_hops(head, gold_choice.text, distractor_pairs)

            # 编码题干与标准答案，作为相似度计算的基础
            question_embedding = self._embed_texts([example.question])[0]
            answer_embedding = self._embed_texts([gold_choice.text])[0]
            candidate_embeddings = {
                choice.label: self._embed_texts([choice.text])[0]
                for choice in example.choices
                if choice.label != example.answer_key
            }

            c_distractor, confusing_flags = self._compute_c_distractor(
                question_embedding,
                answer_embedding,
                candidate_embeddings,
                distractor_distances,
            )
            c_reasoning = self._compute_c_reasoning(example.question, gold_distance, trigger_flags)

            gold_key = example.question + " " + gold_choice.text
            e_model_value = e_model_scores.get(gold_key, 0.5)

            candidate_scores: List[CandidateScores] = []
            # 计算 NLI 支持度：先得到正确答案的最大蕴涵概率
            gold_entail = self._compute_s_truthful(example.question, gold_choice.text)

            best_distractor_entail = 0.0
            for choice in example.choices:
                candidate = CandidateScores(label=choice.label, text=choice.text)
                emb = self._embed_texts([choice.text])[0]
                candidate.sim_q = self._compute_similarity(question_embedding, emb)
                candidate.sim_a = self._compute_similarity(answer_embedding, emb)
                candidate.graph_distance = (
                    gold_distance if choice.label == example.answer_key else distractor_distances.get(choice.label, math.inf)
                )
                candidate.graph_close = candidate.graph_distance <= 2
                candidate.entail_score = self._compute_s_truthful(example.question, choice.text)
                if choice.label != example.answer_key:
                    best_distractor_entail = max(best_distractor_entail, candidate.entail_score)
                candidate_scores.append(candidate)

            delta = gold_entail - best_distractor_entail
            s_truthful = max(0.0, min(1.0, delta)) if gold_entail >= 0.6 else 0.0

            # 解释性得分：比较正确答案与干扰项的图谱距离差
            gap = min(distractor_distances.values(), default=math.inf) - gold_distance
            if math.isinf(gap):
                gap_norm = 1.0
            else:
                gap_norm = max(0.0, min(1.0, (gap + 2) / 5))
            s_explanation = min(1.0, 0.6 * (1 if gold_distance <= 2 else 0) + 0.4 * gap_norm)

            s_question = self._compute_s_question(example.question)
            policy_text = example.question + " " + " ".join(choice.text for choice in example.choices)
            s_policy = self._compute_s_policy(policy_text)

            # 统一汇总难度与安全各分项后取平均得到总分
            difficulty = float(np.mean([c_hops, c_distractor, c_reasoning, e_model_value]))
            safety = float(np.mean([s_question, s_truthful, s_explanation, s_policy]))

            reports.append(
                QAReport(
                    example=example,
                    c_hops=c_hops,
                    c_distractor=c_distractor,
                    c_reasoning=c_reasoning,
                    e_model=e_model_value,
                    difficulty=difficulty,
                    s_question=s_question,
                    s_truthful=s_truthful,
                    s_explanation=s_explanation,
                    s_policy=s_policy,
                    safety=safety,
                    candidate_scores=candidate_scores,
                )
            )

        return reports

    # ----------------------- 结果输出 -----------------------
    def save_reports(self, reports: List[QAReport]) -> Path:
        """将得分导出为 CSV 文件。"""
        records = []
        for report in reports:
            record = {
                "question": report.example.question,
                "question_concept": report.example.question_concept,
                "answer_key": report.example.answer_key,
                "c_hops": report.c_hops,
                "c_distractor": report.c_distractor,
                "c_reasoning": report.c_reasoning,
                "e_model": report.e_model,
                "difficulty": report.difficulty,
                "s_question": report.s_question,
                "s_truthful": report.s_truthful,
                "s_explanation": report.s_explanation,
                "s_policy": report.s_policy,
                "safety": report.safety,
            }
            records.append(record)
        df = pd.DataFrame(records)
        output_path = self.output_dir / "csqa_assessment_scores.csv"
        df.to_csv(output_path, index=False)
        return output_path

    def compute_dataset_scores(self, reports: List[QAReport]) -> Dict[str, float]:
        """汇总数据集整体的难度与安全平均分。"""
        difficulties = [report.difficulty for report in reports]
        safeties = [report.safety for report in reports]
        return {
            "difficulty_mean": float(np.mean(difficulties)),
            "difficulty_std": float(np.std(difficulties)),
            "safety_mean": float(np.mean(safeties)),
            "safety_std": float(np.std(safeties)),
        }

    def plot_distributions(self, reports: List[QAReport]) -> None:
        """根据结果绘制由浅到深渐变色的柱状图。"""
        df = pd.DataFrame(
            [
                {
                    "difficulty": report.difficulty,
                    "safety": report.safety,
                    "c_hops": report.c_hops,
                    "c_distractor": report.c_distractor,
                    "c_reasoning": report.c_reasoning,
                    "e_model": report.e_model,
                    "s_question": report.s_question,
                    "s_truthful": report.s_truthful,
                    "s_explanation": report.s_explanation,
                    "s_policy": report.s_policy,
                }
                for report in reports
            ]
        )

        metric_configs = [
            ("difficulty", "难度总分", "Difficulty Score"),
            ("safety", "安全总分", "Safety Score"),
            ("c_hops", "C_hops 路径跳数", "C_hops"),
            ("c_distractor", "干扰项迷惑度", "Distractor Confusability"),
            ("c_reasoning", "推理复杂度", "Reasoning Load"),
            ("e_model", "模型经验难度", "Model Experience Difficulty"),
            ("s_question", "题面敏感度", "Question Sensitivity"),
            ("s_truthful", "真实度支撑", "Truthfulness Support"),
            ("s_explanation", "可解释支撑", "Explanation Support"),
            ("s_policy", "合规性", "Policy Compliance"),
        ]

        base_colormaps = {
            "difficulty": ("#d0f0fd", "#005f99"),
            "safety": ("#fde0d0", "#aa3000"),
            "c_hops": ("#e0f7da", "#2e7d32"),
            "c_distractor": ("#fff3e0", "#ef6c00"),
            "c_reasoning": ("#ede7f6", "#5e35b1"),
            "e_model": ("#f8bbd0", "#ad1457"),
            "s_question": ("#e3f2fd", "#1565c0"),
            "s_truthful": ("#f1f8e9", "#33691e"),
            "s_explanation": ("#fbe9e7", "#d84315"),
            "s_policy": ("#f9fbe7", "#827717"),
        }

        font_ready = self._ensure_chinese_font()

        for metric, title_cn, title_en in metric_configs:
            values = df[metric].values.astype(float)
            order = np.argsort(values)
            sorted_values = values[order]
            sorted_indices = (order + 1).astype(int)
            num_items = len(sorted_values)

            x_positions = np.arange(num_items)
            cmap = LinearSegmentedColormap.from_list(metric, base_colormaps[metric])
            colors = [cmap(i) for i in np.linspace(0.2, 1.0, num_items)] if num_items > 1 else [cmap(1.0)]

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x_positions, sorted_values, color=colors, width=0.9, align="center")

            xtick_positions = self._select_tick_positions(num_items)
            xtick_labels = [str(sorted_indices[pos]) for pos in xtick_positions]
            ax.set_xticks(xtick_positions)
            ax.set_xticklabels(xtick_labels, rotation=0)

            y_ticks = self._select_value_ticks(sorted_values)
            if y_ticks:
                ax.set_yticks(y_ticks)
                y_max_limit = y_ticks[-1]
            else:
                y_max_limit = max(1.0, math.ceil((max(sorted_values) + 1e-6) * 10) / 10)
            if num_items:
                y_max_limit = max(y_max_limit, float(sorted_values.max()) * 1.05)
            ax.set_ylim(0, y_max_limit if y_max_limit > 0 else 1.0)

            if font_ready:
                ax.set_xlabel("题目索引")
                ax.set_ylabel("得分")
                ax.set_title(title_cn)
            else:
                ax.set_xlabel("Problem Index")
                ax.set_ylabel("Score")
                ax.set_title(title_en)

            fig.tight_layout()
            self._save_plot_multi_formats(fig, metric)
            plt.close(fig)

    def run(self) -> None:
        """整合执行流程。"""
        reports = self.evaluate()
        csv_path = self.save_reports(reports)
        self.plot_distributions(reports)
        summary = self.compute_dataset_scores(reports)
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"评估完成。统计已保存至: {csv_path} 和 {summary_path}")


    def _build_text_classifier_pipeline(self, model_path: Path):
        """根据模型目录自动适配 Transformers 或 ONNX 文本分类推理。"""

        resolved_dir = self._resolve_model_directory(model_path)
        onnx_file = self._find_onnx_model(resolved_dir)
        if onnx_file is not None:
            return OnnxTextClassificationPipeline(resolved_dir, onnx_file)
        # 默认退回到 Transformers pipeline
        return pipeline(
            "text-classification",
            model=str(resolved_dir),
            tokenizer=str(resolved_dir),
            top_k=None,
        )

    def _resolve_model_directory(self, base_path: Path) -> Path:
        """寻找包含配置文件的实际模型目录。"""

        if not base_path.exists():
            raise FileNotFoundError(f"指定的模型目录不存在：{base_path}")
        if self._has_model_config(base_path):
            return base_path
        for child in base_path.iterdir():
            if child.is_dir() and self._has_model_config(child):
                return child
        return base_path

    def _has_model_config(self, directory: Path) -> bool:
        return any((directory / name).exists() for name in ["config.json", "configuration.json"])

    def _find_onnx_model(self, directory: Path) -> Optional[Path]:
        """在模型目录中搜索 ONNX 权重文件（兼容常见的 onnx/ 子目录）。"""

        candidate_names = ["model.onnx", "model_fp16.onnx", "model_int8.onnx", "model_quantized.onnx"]

        search_dirs = [directory]
        onnx_subdir = directory / "onnx"
        if onnx_subdir.is_dir():
            search_dirs.append(onnx_subdir)

        for child in sorted(directory.iterdir()):
            if child.is_dir() and child not in search_dirs:
                search_dirs.append(child)

        for search_dir in search_dirs:
            for name in candidate_names:
                candidate = search_dir / name
                if candidate.exists():
                    return candidate
            extra = sorted(search_dir.glob("*.onnx"))
            if extra:
                return extra[0]
        return None

    def _extract_label_scores(self, pipeline_outputs) -> List[Dict[str, float]]:
        """统一提取分类 pipeline 的第一条样本的标签分数列表。"""

        if not pipeline_outputs:
            return []
        first = pipeline_outputs[0]
        if isinstance(first, dict):
            return pipeline_outputs
        return first

    def _select_tick_positions(self, length: int, target_ticks: int = 8) -> List[int]:
        """根据样本数量自动选择合适的横轴刻度位置。"""

        if length <= 0:
            return []
        if length <= target_ticks:
            return list(range(length))

        step = max(1, int(self._nice_number(length / (target_ticks - 1), round_result=True)))
        positions = list(range(0, length, step))
        if positions[-1] != length - 1:
            positions.append(length - 1)
        return positions

    def _select_value_ticks(self, values: np.ndarray, target_ticks: int = 6) -> List[float]:
        """根据数值分布自动选择纵轴刻度，保证显示稳定。"""

        if values.size == 0:
            return []
        data_min = 0.0
        data_max = float(values.max())
        if math.isclose(data_max, 0.0):
            return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        step = self._nice_number((data_max - data_min) / max(target_ticks - 1, 1), round_result=True)
        if step <= 0:
            step = 0.1
        tick_min = math.floor(data_min / step) * step
        tick_max = math.ceil((data_max + 1e-8) / step) * step
        ticks = np.arange(tick_min, tick_max + step * 0.5, step)
        return [round(float(tick), 3) for tick in ticks if tick >= 0]

    def _nice_number(self, value: float, round_result: bool = False) -> float:
        """取接近 value 的“漂亮数”，优先 1/2/5*10^n。"""

        if value <= 0:
            return 1.0
        exponent = math.floor(math.log10(value))
        fraction = value / (10 ** exponent)

        if round_result:
            if fraction < 1.5:
                nice_fraction = 1.0
            elif fraction < 3.0:
                nice_fraction = 2.0
            elif fraction < 7.0:
                nice_fraction = 5.0
            else:
                nice_fraction = 10.0
        else:
            if fraction <= 1.0:
                nice_fraction = 1.0
            elif fraction <= 2.0:
                nice_fraction = 2.0
            elif fraction <= 5.0:
                nice_fraction = 5.0
            else:
                nice_fraction = 10.0

        return nice_fraction * (10 ** exponent)

    def _save_plot_multi_formats(self, fig: "plt.Figure", metric: str) -> None:
        """将图表以 png/svg/pdf 多格式保存。"""

        safe_name = re.sub(r"[^0-9A-Za-z_-]+", "_", metric)
        for ext in ("png", "svg", "pdf"):
            fig_path = self.output_dir / f"{safe_name}_distribution.{ext}"
            fig.savefig(fig_path, dpi=300 if ext == "png" else None)

    def _ensure_chinese_font(self) -> bool:
        """确保 Matplotlib 使用支持中文的字体，避免缺字警告。"""

        if self._font_checked:
            return self._font_supports_chinese

        self._font_checked = True
        try:
            from matplotlib import font_manager
            from matplotlib.font_manager import FontProperties
        except ImportError:  # pragma: no cover - 一般不会发生，仅作防御
            logger.warning("Matplotlib 字体模块不可用，将改用英文标签绘制图表。")
            self._font_supports_chinese = False
            return False

        preferred_fonts = [
            "Noto Sans CJK SC",
            "Noto Sans SC",
            "Source Han Sans SC",
            "Microsoft YaHei",
            "SimHei",
            "WenQuanYi Micro Hei",
        ]

        for font_name in preferred_fonts:
            try:
                font_path = font_manager.findfont(font_name, fallback_to_default=False)
            except (ValueError, RuntimeError):
                continue
            if font_path and Path(font_path).exists():
                plt.rcParams["font.sans-serif"] = [font_name]
                plt.rcParams["axes.unicode_minus"] = False
                self._font_supports_chinese = True
                logger.info("已配置 Matplotlib 中文字体：%s", font_name)
                return True

        candidate_keywords = [
            "noto",
            "sourcehansans",
            "simhei",
            "wenquanyi",
            "msyh",
            "sarasa",
        ]

        for font_path in font_manager.findSystemFonts():
            lower_name = Path(font_path).stem.lower()
            if not any(keyword in lower_name for keyword in candidate_keywords):
                continue
            try:
                font_manager.fontManager.addfont(font_path)
                font_name = FontProperties(fname=font_path).get_name()
            except Exception:  # pragma: no cover - 单个字体解析失败可忽略
                continue
            plt.rcParams["font.sans-serif"] = [font_name]
            plt.rcParams["axes.unicode_minus"] = False
            self._font_supports_chinese = True
            logger.info("从 %s 注册字体 %s 用于中文渲染。", font_path, font_name)
            return True

        logger.warning("未找到可用的中文字体，已自动切换英文标签以避免绘图警告。")
        self._font_supports_chinese = False
        plt.rcParams.setdefault("axes.unicode_minus", False)
        return False


if __name__ == "__main__":
    evaluator = CSQAEvaluator()
    evaluator.run()

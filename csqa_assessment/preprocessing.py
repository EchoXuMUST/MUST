"""与 CommonsenseQA 数据预处理相关的工具函数。"""
from __future__ import annotations

import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 可选地导入 pandas 以便读取 parquet 数据，若环境缺失则在实际加载时给出明确提示。
try:
    import pandas as pd
except ImportError:  # pragma: no cover - 在未安装 pandas 的环境下运行
    pd = None

# 初始化全局对象以避免重复加载。
_NLP_MODEL = None
_STOP_WORDS = None
_LEMMATIZER = WordNetLemmatizer()


@dataclass
class Choice:
    """表示一个 CommonsenseQA 选项。"""

    label: str
    text: str


@dataclass
class QAExample:
    """封装单条 QA 数据及预处理结果。"""

    question: str
    question_concept: str
    answer_key: str
    choices: List[Choice]


def load_spacy_model(model_name: str = "en_core_web_sm"):
    """按需延迟加载 spaCy 模型。"""
    global _NLP_MODEL
    if _NLP_MODEL is None:
        _NLP_MODEL = spacy.load(model_name, disable=["ner"])
    return _NLP_MODEL


def get_stopwords() -> set:
    """加载 NLTK 停用词表。"""
    global _STOP_WORDS
    if _STOP_WORDS is None:
        try:
            _STOP_WORDS = set(stopwords.words("english"))
        except LookupError:
            import nltk

            nltk.download("stopwords")
            _STOP_WORDS = set(stopwords.words("english"))
    return _STOP_WORDS


def normalize_text(text: str) -> str:
    """对文本进行小写化、去除标点及多余空格。"""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_tokens(tokens: Iterable[str]) -> List[str]:
    """对分词结果进行词形还原，同时保留数字、否定等信息。"""
    processed = []
    for token in tokens:
        if token.isdigit():
            processed.append(token)
            continue
        lemma = _LEMMATIZER.lemmatize(token)
        processed.append(lemma)
    return processed


def preprocess_question(question: str) -> Dict[str, List[str]]:
    """对问题进行分词、词形还原和 POS/依存分析。"""
    nlp = load_spacy_model()
    doc = nlp(question)
    stop_words = get_stopwords()

    tokens = [token.text.lower() for token in doc if token.text.strip()]
    lemmas = lemmatize_tokens(tokens)
    filtered_tokens = [tok for tok in lemmas if tok not in stop_words]
    pos_tags = [token.pos_ for token in doc]
    dependencies = [(token.text, token.dep_, token.head.text) for token in doc]

    return {
        "tokens": tokens,
        "lemmas": lemmas,
        "filtered_tokens": filtered_tokens,
        "pos_tags": pos_tags,
        "dependencies": dependencies,
    }


def preprocess_choice(text: str) -> Dict[str, List[str]]:
    """与问题相同的文本预处理流程。"""
    return preprocess_question(text)


def load_csqa_dataset(file_path: Path) -> List[QAExample]:
    """读取 CommonsenseQA 数据文件（支持 JSONL 与 Parquet）并解析为 QAExample 列表。"""

    def _parse_entry(entry: Dict) -> QAExample:
        """将通用字典结构转换为 QAExample。"""
        question_raw = entry.get("question", "")
        if isinstance(question_raw, dict):
            question = (
                question_raw.get("stem")
                or question_raw.get("text")
                or question_raw.get("question")
                or ""
            )
            choices_data = question_raw.get("choices", [])
        else:
            question = str(question_raw)
            choices_data = entry.get("choices", [])

        question_concept = (
            entry.get("question_concept")
            or entry.get("questionConcept")
            or entry.get("question_concept")
            or ""
        )
        answer_key = entry.get("answerKey") or entry.get("gold") or ""

        # 某些数据源的 choices 以 dict["label"]=..., dict["text"]=... 存储
        normalized_choices: List[Choice] = []
        if isinstance(choices_data, dict) and {"label", "text"}.issubset(choices_data.keys()):
            labels = list(choices_data.get("label", []))
            texts = list(choices_data.get("text", []))
            for label, text in zip(labels, texts):
                normalized_choices.append(Choice(label=str(label), text=str(text)))
            # 若标签与文本长度不一致，补齐剩余部分
            if len(labels) < len(texts):
                for offset, text in enumerate(texts[len(labels) :], start=len(labels)):
                    label = chr(ord("A") + offset)
                    normalized_choices.append(Choice(label=label, text=str(text)))
            elif len(texts) < len(labels):
                for offset, label in enumerate(labels[len(texts) :], start=len(texts)):
                    normalized_choices.append(Choice(label=str(label), text=""))
        else:
            for choice in choices_data:
                if isinstance(choice, dict):
                    label = str(choice.get("label", ""))
                    text = str(choice.get("text", ""))
                    normalized_choices.append(Choice(label=label, text=text))
                else:
                    # 防御性处理：若格式异常，使用字符串形式并生成占位标签
                    text = str(choice)
                    label = chr(ord("A") + len(normalized_choices))
                    normalized_choices.append(Choice(label=label, text=text))

        return QAExample(
            question=question,
            question_concept=str(question_concept),
            answer_key=str(answer_key),
            choices=normalized_choices,
        )

    suffix = file_path.suffix.lower()
    examples: List[QAExample] = []

    if suffix in {".jsonl", ".json"}:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                examples.append(_parse_entry(data))
        return examples

    if suffix == ".parquet":
        if pd is None:
            raise RuntimeError(
                "读取 parquet 格式需要 pandas，请先安装 pandas 或提供 JSONL 数据。"
            )
        frame = pd.read_parquet(file_path)
        for _, row in frame.iterrows():
            entry = row.to_dict()
            examples.append(_parse_entry(entry))
        return examples

    raise ValueError(f"暂不支持的 CommonsenseQA 文件格式: {suffix}")


def extract_trigger_flags(question_tokens: List[str], question_text: str) -> Dict[str, bool]:
    """根据给定的触发词规则标记推理类型。"""
    comparative_words = {
        "more",
        "most",
        "less",
        "least",
        "bigger",
        "smaller",
        "greater",
        "largest",
        "smallest",
        "best",
        "worst",
        "closer",
        "closest",
        "farther",
        "farthest",
        "than",
        "compared",
        "compare",
    }
    negation_words = {
        "not",
        "never",
        "no",
        "except",
        "other",
        "least",
        "incorrect",
        "false",
    }
    quant_words = re.compile(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|first|second|third|percent|hours?|days?|years?|miles?|kg|cm)\b"
    )

    question_lower = question_text.lower()

    cmp_flag = any(token in comparative_words for token in question_tokens)
    neg_flag = any(word in question_tokens for word in negation_words)
    quant_flag = bool(quant_words.search(question_lower))
    alias_flag = False

    conj_flag = any(token in {"and", "or", "that", "which", "whose", "with"} for token in question_tokens)

    return {
        "cmp_flag": cmp_flag,
        "conj_flag": conj_flag,
        "neg_flag": neg_flag,
        "alias_flag": alias_flag,
        "quant_flag": quant_flag,
    }

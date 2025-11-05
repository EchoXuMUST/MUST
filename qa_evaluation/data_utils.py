"""与数据解析、清洗相关的工具函数。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm


@dataclass
class QAExample:
    """表示 2WikiMultiHopQA 数据集中的一条样本。"""

    _id: str
    question: str
    answer: str
    q_type: str
    supporting_facts: List[Tuple[str, int]]
    context: List[Tuple[str, List[str]]]
    evidences: Optional[List[Tuple[str, int]]] = None
    entity_ids: Optional[Dict[str, Sequence[str]]] = None

    @property
    def unique_titles(self) -> Set[str]:
        """返回支持事实中涉及的唯一页面标题集合。"""

        return {title for title, _ in self.supporting_facts}

    @property
    def support_sentences(self) -> Set[Tuple[str, int]]:
        """返回支持事实涉及的 (标题, 句子编号) 集合。"""

        return {(title, int(sent_id)) for title, sent_id in self.supporting_facts}

    @property
    def total_sentences(self) -> int:
        """计算上下文中句子的总数量。"""

        return sum(len(sentences) for _, sentences in self.context)

    def find_sentence_text(self, title: str, sent_idx: int) -> Optional[str]:
        """在上下文中查找指定句子文本。"""

        for ctx_title, sentences in self.context:
            if ctx_title == title and 0 <= sent_idx < len(sentences):
                return sentences[sent_idx]
        return None

    def gather_support_texts(self) -> List[str]:
        """聚合所有支持句的原文，便于后续特征计算。"""

        texts: List[str] = []
        for title, sent_id in self.support_sentences:
            text = self.find_sentence_text(title, sent_id)
            if text:
                texts.append(text)
        return texts


def load_dataset(path: str | Path) -> List[QAExample]:
    """从 JSONL 文件中加载数据集。"""

    dataset: List[QAExample] = []
    path = Path(path)
    total_bytes = path.stat().st_size
    with path.open("r", encoding="utf-8") as f, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc="加载数据集",
    ) as progress:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            supporting_facts = [tuple(item) for item in record.get("supporting_facts", [])]
            context = [(title, sentences) for title, sentences in record.get("context", [])]
            evidences = record.get("evidences")
            if evidences:
                evidences = [tuple(item) for item in evidences]
            # 数据集中部分文件使用 "id" 或 "question_id" 作为样本标识，因此这里做兼容处理，
            # 若都不存在则退化为按行号生成稳定的占位 ID，避免 KeyError 中断主流程。
            record_id = (
                record.get("_id")
                or record.get("id")
                or record.get("question_id")
                or f"auto_{line_no}"
            )
            example = QAExample(
                _id=str(record_id),
                question=record.get("question", ""),
                answer=record.get("answer", ""),
                q_type=record.get("type", ""),
                supporting_facts=supporting_facts,
                context=context,
                evidences=evidences,
                entity_ids=record.get("entity_ids"),
            )
            dataset.append(example)
            progress.update(len(line.encode("utf-8")))
    return dataset


def load_predictions(path: Optional[str | Path]) -> Dict[str, Dict[str, object]]:
    """从预测文件中加载模型输出，若路径为空则返回空字典。"""

    if path is None:
        return {}
    pred_path = Path(path)
    if not pred_path.exists():
        raise FileNotFoundError(f"找不到预测文件: {pred_path}")
    predictions: Dict[str, Dict[str, object]] = {}
    total_bytes = pred_path.stat().st_size
    with pred_path.open("r", encoding="utf-8") as f, tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc="加载预测文件",
    ) as progress:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            record_id = (
                record.get("_id")
                or record.get("id")
                or record.get("question_id")
                or f"auto_{line_no}"
            )
            predictions[str(record_id)] = record
            progress.update(len(line.encode("utf-8")))
    return predictions

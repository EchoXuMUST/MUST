"""简易 BM25 检索器，针对段落句子级别的评分。"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .data_utils import QAExample


class SimpleBM25:
    """最小实现的 BM25，用于句级检索。"""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.df = Counter()
        for doc in self.documents:
            for term in set(doc):
                self.df[term] += 1
        self.N = len(self.documents)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token]

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query: Sequence[str], index: int) -> float:
        doc = self.documents[index]
        doc_len = self.doc_lengths[index]
        counter = Counter(doc)
        score = 0.0
        for term in query:
            if term not in counter:
                continue
            numerator = counter[term] * (self.k1 + 1)
            denominator = counter[term] + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += self.idf(term) * numerator / denominator
        return score

    def get_scores(self, query: Sequence[str]) -> List[float]:
        return [self.score(query, idx) for idx in range(self.N)]


def build_sentence_corpus(example: QAExample) -> Tuple[List[str], List[Tuple[str, int]]]:
    """把上下文拆成句子列表，同时记录句子对应的 (标题, 序号)。"""

    sentences: List[str] = []
    mapping: List[Tuple[str, int]] = []
    for title, sents in example.context:
        for idx, sent in enumerate(sents):
            sentences.append(sent)
            mapping.append((title, idx))
    return sentences, mapping


def bm25_sentence_scores(example: QAExample) -> Dict[Tuple[str, int], float]:
    """对一个样本执行 BM25 检索，返回每个句子的得分。"""

    sentences, mapping = build_sentence_corpus(example)
    if not sentences:
        return {}
    bm25 = SimpleBM25(sentences)
    query_tokens = SimpleBM25._tokenize(example.question)
    scores = bm25.get_scores(query_tokens)
    return {mapping[idx]: scores[idx] for idx in range(len(mapping))}

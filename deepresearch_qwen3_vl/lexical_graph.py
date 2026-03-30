from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable

TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\u4e00-\u9fff]+")
STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "to",
    "of",
    "in",
    "and",
    "for",
    "on",
    "with",
    "by",
    "what",
    "which",
    "how",
    "why",
    "when",
    "where",
    "是",
    "的",
    "了",
    "和",
    "在",
    "与",
    "及",
    "对",
    "中",
    "将",
    "其",
    "一个",
}


@dataclass
class ChunkNode:
    chunk_id: str
    source: str
    text: str
    tokens: list[str]
    entities: list[str]


@dataclass
class RetrievalResult:
    chunk_id: str
    source: str
    text: str
    score: float
    reasons: list[str] = field(default_factory=list)


class LexicalGraphIndex:
    """In-memory lexical graph inspired by awslabs lexical-graph hierarchy.

    Hierarchy we keep lightweight:
      source -> chunk -> entity
    plus inverted indexes for lexical retrieval and entity traversal.
    """

    def __init__(self) -> None:
        self.chunks: dict[str, ChunkNode] = {}
        self.source_to_chunks: dict[str, list[str]] = defaultdict(list)
        self.entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        self.token_to_chunks: dict[str, set[str]] = defaultdict(set)

    def add_document(self, source: str, text: str, chunk_size: int = 500) -> None:
        clean = " ".join(text.split())
        if not clean:
            return
        for i in range(0, len(clean), chunk_size):
            chunk = clean[i : i + chunk_size]
            chunk_id = f"{source}#c{i//chunk_size}"
            tokens = _tokenize(chunk)
            entities = _extract_entities(chunk)

            node = ChunkNode(
                chunk_id=chunk_id,
                source=source,
                text=chunk,
                tokens=tokens,
                entities=entities,
            )
            self.chunks[chunk_id] = node
            self.source_to_chunks[source].append(chunk_id)
            for t in set(tokens):
                self.token_to_chunks[t].add(chunk_id)
            for e in set(entities):
                self.entity_to_chunks[e].add(chunk_id)

    def has_data(self) -> bool:
        return bool(self.chunks)


class LexicalGraphRetriever:
    """Traversal-based search approximation.

    Step 1: lexical seed by token overlap (vector-search analogue).
    Step 2: entity expansion to traverse related chunks (entity-network analogue).
    Step 3: rerank by combined lexical + traversal support.
    """

    def __init__(self, index: LexicalGraphIndex) -> None:
        self.index = index

    def retrieve(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        q_tokens = _tokenize(question)
        if not q_tokens or not self.index.has_data():
            return []

        seed_scores: dict[str, float] = defaultdict(float)
        for t in set(q_tokens):
            chunks = self.index.token_to_chunks.get(t, set())
            idf = math.log((1 + len(self.index.chunks)) / (1 + len(chunks))) + 1.0
            for cid in chunks:
                seed_scores[cid] += idf

        seed_ids = sorted(seed_scores, key=lambda c: seed_scores[c], reverse=True)[: max(top_k, 3)]

        expanded: dict[str, float] = defaultdict(float)
        for cid in seed_ids:
            expanded[cid] += seed_scores[cid]
            entities = self.index.chunks[cid].entities
            for ent in entities:
                for neighbor_cid in self.index.entity_to_chunks.get(ent, set()):
                    if neighbor_cid == cid:
                        continue
                    expanded[neighbor_cid] += 0.35 * seed_scores[cid]

        results: list[RetrievalResult] = []
        for cid, sc in expanded.items():
            chunk = self.index.chunks[cid]
            overlap = len(set(q_tokens) & set(chunk.tokens))
            entity_overlap = len(set(_extract_entities(question)) & set(chunk.entities))
            final_score = sc + 0.2 * overlap + 0.3 * entity_overlap
            reasons = []
            if cid in seed_ids:
                reasons.append("lexical-seed")
            if entity_overlap > 0:
                reasons.append("entity-traversal")
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    source=chunk.source,
                    text=chunk.text,
                    score=final_score,
                    reasons=reasons,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]


def _tokenize(text: str) -> list[str]:
    tokens = [x.lower() for x in TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def _extract_entities(text: str) -> list[str]:
    # Heuristic entities: CamelCase / AllCaps / Chinese words (len>=2)
    raw = TOKEN_RE.findall(text)
    ents: list[str] = []
    for tok in raw:
        if len(tok) < 2:
            continue
        if re.search(r"[\u4e00-\u9fff]", tok):
            ents.append(tok)
        elif tok[:1].isupper() and any(c.islower() for c in tok[1:]):
            ents.append(tok)
        elif tok.isupper() and len(tok) <= 8:
            ents.append(tok)
    return list(dict.fromkeys(ents))


def build_graph_from_corpus(corpus: Iterable[tuple[str, str]]) -> LexicalGraphIndex:
    index = LexicalGraphIndex()
    for source, text in corpus:
        index.add_document(source=source, text=text)
    return index

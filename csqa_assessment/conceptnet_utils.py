"""与 ConceptNet 相关的工具函数。"""
from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx


class ConceptNetGraph:
    """简单封装 ConceptNet 图的构建与最短路查询。"""

    def __init__(self, csv_path: Path, max_edges: Optional[int] = None) -> None:
        self.csv_path = csv_path
        self.graph = nx.Graph()
        self._load_graph(max_edges=max_edges)

    def _normalize_concept(self, concept: str) -> str:
        normalized = concept.lower().replace(" ", "_")
        return normalized

    def _load_graph(self, max_edges: Optional[int] = None) -> None:
        allowed_relations = {
            "/r/IsA",
            "/r/RelatedTo",
            "/r/CapableOf",
            "/r/UsedFor",
            "/r/AtLocation",
            "/r/Causes",
            "/r/HasA",
            "/r/PartOf",
        }
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for idx, row in enumerate(reader):
                if max_edges and idx >= max_edges:
                    break
                if len(row) < 5:
                    continue
                rel, start, end, data = row[1], row[2], row[3], row[4]
                if rel not in allowed_relations:
                    continue
                start = self._normalize_concept(start.split("/c/en/")[-1])
                end = self._normalize_concept(end.split("/c/en/")[-1])
                weight = 1.0
                self.graph.add_edge(start, end, weight=weight)

    def shortest_path_length(self, start: str, end: str, default: float = math.inf) -> float:
        start = self._normalize_concept(start)
        end = self._normalize_concept(end)
        if start not in self.graph or end not in self.graph:
            return default
        try:
            length = nx.shortest_path_length(self.graph, source=start, target=end)
            return float(length)
        except nx.NetworkXNoPath:
            return default


def normalize_distance(distance: float) -> float:
    """按难度规范对跳数进行归一化。"""
    if math.isinf(distance):
        return 1.0
    score = max(0.0, min(1.0, (distance - 1) / 3))
    return score

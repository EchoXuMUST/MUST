from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


def _uniq(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _is_noise_entity(v: str) -> bool:
    s = v.strip()
    if not s:
        return True
    if len(s) <= 1:
        return True
    if re.fullmatch(r"[\W_]+", s):
        return True
    return False


@dataclass
class RerankerConfig:
    hop_l: int
    top_k: int


@dataclass
class StrategyResult:
    strategy: str
    entities: list[str]
    paths: list[dict[str, Any]]
    dsl: dict[str, Any]
    score: float


class CypherDSLCompiler:
    """Compile schema-constrained DSL JSON to OpenCypher."""

    def compile(self, dsl: dict[str, Any]) -> str:
        match_parts: list[str] = []

        for node in dsl.get("match", []):
            alias = str(node.get("alias", "n")).strip() or "n"
            ntype = str(node.get("node_type", "")).strip()
            cons = node.get("constraints", {}) or {}
            where_props = []
            for k, v in cons.items():
                if isinstance(v, str):
                    where_props.append(f"{alias}.{k} = '{v.replace(chr(39), chr(92)+chr(39))}'")
                else:
                    where_props.append(f"{alias}.{k} = {json.dumps(v)}")
            label = f":{ntype}" if ntype else ""
            match_parts.append(f"({alias}{label})")
            node["__where_props"] = where_props

        edge_parts: list[str] = []
        for e in dsl.get("edges", []):
            frm = str(e.get("from", "")).strip()
            to = str(e.get("to", "")).strip()
            rel = str(e.get("rel_type", "")).strip()
            if not frm or not to:
                continue
            edge_parts.append(f"({frm})-[:{rel}]->({to})" if rel else f"({frm})--({to})")

        query_parts: list[str] = []
        if edge_parts:
            query_parts.append("MATCH " + ", ".join(edge_parts))
        elif match_parts:
            query_parts.append("MATCH " + ", ".join(match_parts))
        else:
            query_parts.append("MATCH (n)")

        where_clauses: list[str] = []
        for node in dsl.get("match", []):
            where_clauses.extend(node.get("__where_props", []))
        for cond in dsl.get("where", []) or []:
            if isinstance(cond, str) and cond.strip():
                where_clauses.append(cond.strip())
        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))

        returns = dsl.get("return", ["n"])
        if not isinstance(returns, list) or not returns:
            returns = ["n"]
        query_parts.append("RETURN " + ", ".join(str(x) for x in returns))

        sort = dsl.get("sort")
        if isinstance(sort, str) and sort.strip():
            query_parts.append("ORDER BY " + sort.strip())

        limit = dsl.get("limit")
        if isinstance(limit, int) and limit > 0:
            query_parts.append(f"LIMIT {limit}")

        return "\n".join(query_parts)


class BYOKGValidator:
    def __init__(self, dry_run: Callable[[str], bool] | None = None) -> None:
        self.dry_run = dry_run

    def validate(self, artifacts: dict[str, Any], schema: dict[str, Any]) -> tuple[bool, list[str]]:
        errors: list[str] = []

        entities = [str(x) for x in artifacts.get("entities", [])]
        entities = [x for x in _uniq(entities) if not _is_noise_entity(x)]
        artifacts["entities"] = entities
        if not entities:
            errors.append("entities_empty")

        allowed_rels = {str(x) for x in schema.get("relation_types", [])} if isinstance(schema, dict) else set()
        paths = artifacts.get("paths", [])
        if isinstance(paths, list):
            for p in paths:
                if not isinstance(p, dict):
                    continue
                rel = str(p.get("relation", "")).strip()
                if rel and allowed_rels and rel not in allowed_rels:
                    errors.append(f"invalid_relation:{rel}")

        cypher = str(artifacts.get("opencypher", "")).strip()
        if cypher:
            c_up = cypher.upper()
            if "MATCH" not in c_up or "RETURN" not in c_up:
                errors.append("cypher_missing_match_or_return")
            if self.dry_run is not None:
                try:
                    if not self.dry_run(cypher + "\nLIMIT 1"):
                        errors.append("cypher_dry_run_failed")
                except Exception:
                    errors.append("cypher_dry_run_failed")
        else:
            errors.append("cypher_empty")

        return (len(errors) == 0), errors


class BYOKGRAGProvider:
    """Graph Evidence Provider adapted from BYOKG-RAG style multi-strategy refinement."""

    def __init__(
        self,
        linker: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        strategy_linker: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cypher_dry_run: Callable[[str], bool] | None = None,
        cypher_executor: Callable[[str], list[dict[str, Any]]] | None = None,
        max_refine_rounds: int = 2,
        max_query_results: int = 20,
    ) -> None:
        self.linker = linker
        self.strategy_linker = strategy_linker
        self.max_refine_rounds = max_refine_rounds
        self.max_query_results = max_query_results
        self.validator = BYOKGValidator(dry_run=cypher_dry_run)
        self.cypher_executor = cypher_executor
        self.compiler = CypherDSLCompiler()

    def _dynamic_reranker(self, question: str, schema: dict[str, Any]) -> RerankerConfig:
        rels = schema.get("relation_types", []) if isinstance(schema, dict) else []
        dense = len(rels) > 80
        multi_hop = any(k in question.lower() for k in ["then", "after", "before", "compare", "path", "multi-hop", "并且", "再", "然后"])

        l = 1 if dense else 2
        k = 10 if dense else 30
        if multi_hop:
            k = min(k + 20, 80)
        return RerankerConfig(hop_l=l, top_k=k)

    def _default_linker(self, payload: dict[str, Any]) -> dict[str, Any]:
        question = str(payload.get("question", ""))
        tokens = [t.strip("?,.()[]{}") for t in question.split()]
        entities = [t for t in tokens if len(t) > 2][:4]
        dsl = {
            "match": [{"node_type": "Entity", "alias": "e", "constraints": {"name": entities[0] if entities else ""}}],
            "return": ["e"],
            "limit": 20,
        }
        return {
            "entities": entities,
            "paths": [],
            "dsl": dsl,
        }

    def _invoke_linker(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.linker is not None:
            out = self.linker(payload)
            return out if isinstance(out, dict) else {}
        return self._default_linker(payload)

    def _invoke_strategy_linker(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.strategy_linker is not None:
            out = self.strategy_linker(payload)
            return out if isinstance(out, dict) else {}
        return self._invoke_linker(payload)

    def _strategy_payloads(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        schema = payload.get("schema", {}) if isinstance(payload.get("schema"), dict) else {}
        return [
            {**payload, "strategy": "entity_path", "hints": ["entity-first", "path-expand"]},
            {**payload, "strategy": "relation_path", "hints": ["relation-first", "schema-filter"], "focus_relations": schema.get("relation_types", [])[:20]},
            {**payload, "strategy": "query_shape", "hints": ["query-structure", "answer-type"]},
        ]

    def _score_strategy_result(self, entities: list[str], paths: list[dict[str, Any]], schema: dict[str, Any]) -> float:
        allowed_rels = {str(x) for x in schema.get("relation_types", [])} if isinstance(schema, dict) else set()
        valid_paths = 0
        for p in paths:
            if not isinstance(p, dict):
                continue
            rel = str(p.get("relation", "")).strip()
            if (not allowed_rels) or (rel in allowed_rels):
                valid_paths += 1
        entity_term = min(len(entities), 6) / 6.0
        path_term = min(valid_paths, 10) / 10.0
        return _clip_float(0.55 * entity_term + 0.45 * path_term)

    def _ensemble_linking(self, payload: dict[str, Any], schema: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        candidates: list[StrategyResult] = []
        for p in self._strategy_payloads(payload):
            linked = self._invoke_strategy_linker(p)
            entities = _uniq([str(x) for x in linked.get("entities", [])])
            paths = linked.get("paths", []) if isinstance(linked.get("paths"), list) else []
            dsl = linked.get("dsl", {}) if isinstance(linked.get("dsl"), dict) else {}
            score = self._score_strategy_result(entities, paths, schema)
            candidates.append(StrategyResult(strategy=str(p.get("strategy", "default")), entities=entities, paths=paths, dsl=dsl, score=score))

        if not candidates:
            fallback = self._invoke_linker(payload)
            return fallback, []

        candidates.sort(key=lambda x: x.score, reverse=True)
        best = candidates[0]
        merged_entities = _uniq([e for c in candidates[:2] for e in c.entities])
        merged_paths = [p for c in candidates[:2] for p in c.paths if isinstance(p, dict)]
        best_artifact = {
            "entities": merged_entities,
            "paths": merged_paths,
            "dsl": best.dsl,
            "linking_strategy": best.strategy,
        }
        ranking = [{"strategy": c.strategy, "score": c.score, "entities": len(c.entities), "paths": len(c.paths)} for c in candidates]
        return best_artifact, ranking

    def _expand_schema_focus(self, question: str, schema: dict[str, Any]) -> dict[str, Any]:
        rels = [str(x) for x in schema.get("relation_types", [])] if isinstance(schema, dict) else []
        q_tokens = set(t.lower() for t in re.split(r"\W+", question) if t)
        focus_rels = [r for r in rels if any(t and t in r.lower() for t in q_tokens)]
        if not focus_rels:
            focus_rels = rels[:15]
        return {
            **schema,
            "focus_relation_types": focus_rels[:25],
            "schema_pruned": True,
        }

    def _run_query(self, cypher: str) -> list[dict[str, Any]]:
        if self.cypher_executor is None or not cypher:
            return []
        try:
            rows = self.cypher_executor(cypher)
        except Exception:
            return []
        if not isinstance(rows, list):
            return []
        out = [r for r in rows if isinstance(r, dict)]
        return out[: self.max_query_results]

    def retrieve_graph_evidence(
        self,
        question: str,
        schema: dict[str, Any],
        graph_context: str | None = None,
    ) -> dict[str, Any]:
        reranker = self._dynamic_reranker(question, schema)
        artifacts: dict[str, Any] = {"entities": [], "paths": [], "dsl": {}, "opencypher": ""}
        history: list[dict[str, Any]] = []

        payload = {
            "question": question,
            "schema": self._expand_schema_focus(question, schema),
            "graph_context": graph_context or "",
            "mode": "full",
            "reranker": {"L": reranker.hop_l, "k": reranker.top_k},
        }
        strategy_ranking: list[dict[str, Any]] = []

        for i in range(max(self.max_refine_rounds, 1)):
            linked, strategy_ranking = self._ensemble_linking(payload, schema)
            artifacts.update({k: v for k, v in linked.items() if k in {"entities", "paths", "dsl", "opencypher"}})

            if not artifacts.get("opencypher") and isinstance(artifacts.get("dsl"), dict):
                try:
                    artifacts["opencypher"] = self.compiler.compile(artifacts["dsl"])
                except Exception:
                    artifacts["opencypher"] = ""

            ok, errors = self.validator.validate(artifacts, schema)
            history.append({"round": i + 1, "ok": ok, "errors": errors})
            if ok:
                break

            payload = {
                "question": question,
                "schema": {
                    "summary": schema.get("summary", ""),
                    "relation_types": schema.get("relation_types", []),
                    "node_types": schema.get("node_types", []),
                },
                "graph_context": graph_context or "",
                "previous_artifacts": artifacts,
                "errors": errors,
                "mode": "repair-only",
                "allowed_patch_fields": ["entities", "paths", "dsl", "opencypher"],
            }

        entities = _uniq([str(x) for x in artifacts.get("entities", [])])
        paths = artifacts.get("paths", []) if isinstance(artifacts.get("paths"), list) else []
        cypher = str(artifacts.get("opencypher", "")).strip()
        query_results = self._run_query(cypher)
        image_evidence = [
            row for row in query_results
            if any(k in row for k in ["image", "image_url", "bbox", "ocr_text", "visual_object"])
        ]

        graph_evidence: dict[str, Any] = {
            "triples": [p for p in paths if isinstance(p, dict)],
            "paths": paths,
            "query_results": query_results,
            "image_evidence": image_evidence,
        }
        linking_artifacts = {
            "entities": entities,
            "paths": paths,
            "opencypher": cypher,
            "dsl": artifacts.get("dsl", {}),
            "reranker": {"L": reranker.hop_l, "k": reranker.top_k},
            "strategy_ranking": strategy_ranking,
            "selected_strategy": linked.get("linking_strategy", "unknown") if isinstance(linked, dict) else "unknown",
            "refinement_history": history,
        }

        coverage = _clip_float((len(entities) / 4.0) * 0.5 + (len(paths) / max(reranker.top_k, 1)) * 0.3 + (min(len(query_results), 5) / 5.0) * 0.2)
        confidence = _clip_float(0.5 * coverage + (0.5 if cypher else 0.0))

        return {
            "graph_evidence": graph_evidence,
            "linking_artifacts": linking_artifacts,
            "confidence": confidence,
            "coverage": coverage,
        }


def _clip_float(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))

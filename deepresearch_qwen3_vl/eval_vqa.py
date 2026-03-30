from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Any, **_: Any) -> Any:  # type: ignore[misc]
        return iterable

from difficulty import _task_family, _qnorm_100, difficulty_bucket, estimate_question_difficulty
from lexical_graph import LexicalGraphRetriever, build_graph_from_corpus


@dataclass
class AblationSetting:
    use_kg_rag: bool
    use_difficulty: bool
    use_sft: bool
    profile: str


def _resolve_ablation(profile: str) -> AblationSetting:
    table = {
        "none": AblationSetting(False, False, False, "none"),
        "kg_only": AblationSetting(True, False, False, "kg_only"),
        "difficulty_only": AblationSetting(False, True, False, "difficulty_only"),
        "kg_difficulty": AblationSetting(True, True, False, "kg_difficulty"),
        "kg_sft": AblationSetting(True, False, True, "kg_sft"),
        "difficulty_sft": AblationSetting(False, True, True, "difficulty_sft"),
        "all_on": AblationSetting(True, True, True, "all_on"),
    }
    return table.get(profile, table["all_on"])


def _jsonable(v: Any) -> Any:
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (bytes, bytearray)):
        return {"__type": "bytes", "base64": base64.b64encode(bytes(v)).decode("utf-8")}
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "tolist"):
        try:
            return _jsonable(v.tolist())
        except Exception:
            pass
    if hasattr(v, "item"):
        try:
            return _jsonable(v.item())
        except Exception:
            pass
    return str(v)


def _record_key(sample: dict[str, Any]) -> str:
    original = sample.get("__original_record")
    if isinstance(original, dict):
        for k in ["id", "question_id", "image_id"]:
            v = original.get(k)
            if v is not None and str(v).strip():
                return f"id::{k}::{v}"
    text = f"{sample.get('question','')}||{sample.get('answer','')}"
    return "sha1::" + hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _load_resume_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".jsonl":
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
        return out
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    return []


def _tertile_thresholds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.34, 0.67
    arr = sorted(values)

    def _q(p: float) -> float:
        if len(arr) == 1:
            return arr[0]
        idx = p * (len(arr) - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return arr[lo]
        frac = idx - lo
        return arr[lo] * (1.0 - frac) + arr[hi] * frac

    return _q(1.0 / 3.0), _q(2.0 / 3.0)


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())


def token_f1(pred: str, gold: str) -> float:
    p = list(normalize(pred))
    g = list(normalize(gold))
    if not p or not g:
        return 0.0
    common = 0
    g_used = [False] * len(g)
    for ch in p:
        for i, gg in enumerate(g):
            if not g_used[i] and ch == gg:
                g_used[i] = True
                common += 1
                break
    if common == 0:
        return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)


def _contains_answer(text: str, answer: str) -> bool:
    t = normalize(text)
    a = normalize(answer)
    return bool(a) and a in t


def compute_evidence_metrics(graph_hits: list[Any], docs: list[dict[str, Any]], answer: str) -> tuple[int, float, float]:
    if not answer:
        return 0, 0.0, 0.0

    relevant_total = 0
    for d in docs:
        txt = str(d.get("text", ""))
        if txt and _contains_answer(txt, answer):
            relevant_total += 1

    retrieved_relevant = 0
    for h in graph_hits:
        txt = str(getattr(h, "text", ""))
        if txt and _contains_answer(txt, answer):
            retrieved_relevant += 1

    retrieved_total = len(graph_hits)
    hits = int(retrieved_relevant > 0)
    recall = (retrieved_relevant / relevant_total) if relevant_total > 0 else 0.0
    precision = (retrieved_relevant / retrieved_total) if retrieved_total > 0 else 0.0
    return hits, recall, precision


def _difficulty_route_tag(overall: float) -> str:
    if overall > 0.67:
        return "full_rag"
    if overall > 0.34:
        return "light_rag"
    return "direct_inference"


def _sft_role(bucket: str, verification: str) -> str:
    if bucket == "hard" and verification == "Unverified":
        return "hard_unverified_primary"
    if bucket == "hard" and verification == "Verified":
        return "hard_verified_aux"
    return "not_selected"


def _build_route_aware_graph_context(
    route: str,
    question: str,
    corpus: list[tuple[str, str]],
) -> tuple[list[Any], str, dict[str, Any]]:
    if route == "direct_inference":
        return [], "", {"rounds": 0, "top_k": 0, "expanded_query": None}

    if not corpus:
        return [], "", {"rounds": 1 if route == "light_rag" else 2, "top_k": 0, "expanded_query": None}

    index = build_graph_from_corpus(corpus)
    retriever = LexicalGraphRetriever(index)

    if route == "light_rag":
        graph_hits = retriever.retrieve(question, top_k=4)
        graph_context = "\n\n".join([f"[{h.source}] {h.text[:600]}" for h in graph_hits])
        return graph_hits, graph_context, {"rounds": 1, "top_k": 4, "expanded_query": None}

    first_hits = retriever.retrieve(question, top_k=6)
    expansion_terms: list[str] = []
    for hit in first_hits[:2]:
        snippet = str(getattr(hit, "text", ""))[:120]
        if snippet:
            expansion_terms.append(snippet)
    expanded_query = question if not expansion_terms else f"{question}\n补充证据: {' '.join(expansion_terms)}"
    second_hits = retriever.retrieve(expanded_query, top_k=6)

    merged: list[Any] = []
    seen = set()
    for hit in [*first_hits, *second_hits]:
        key = (getattr(hit, "source", ""), getattr(hit, "text", ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)

    graph_context = "\n\n".join([f"[{h.source}] {h.text[:600]}" for h in merged[:8]])
    return merged[:8], graph_context, {"rounds": 2, "top_k": 6, "expanded_query": expanded_query if expansion_terms else None}


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _extract_answer(record: dict[str, Any], args: argparse.Namespace) -> str:
    v = record.get(args.answer_field)
    if not _is_missing(v):
        return str(v)

    direct = record.get("direct_answers")
    if isinstance(direct, list):
        for x in direct:
            if not _is_missing(x):
                return str(x)

    idx = record.get("correct_choice_idx")
    choices = record.get("choices")
    if not _is_missing(idx) and choices is not None:
        try:
            i = int(idx)
            if hasattr(choices, "tolist"):
                choices = choices.tolist()
            if isinstance(choices, (list, tuple)) and 0 <= i < len(choices):
                return str(choices[i])
        except Exception:
            pass

    raise ValueError("Unable to infer answer; set --answer-field or provide direct_answers / correct_choice_idx+choices.")


def _to_choice_list(v: Any) -> list[str]:
    if v is None:
        return []
    if hasattr(v, "tolist"):
        try:
            v = v.tolist()
        except Exception:
            pass
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return []
        if isinstance(parsed, (list, tuple)):
            return [str(x) for x in parsed]
    return []


def _is_placeholder_image_text(v: str) -> bool:
    t = v.strip().lower()
    return t in {"?", "not supported with pagination yet", "none", "null", "nan", ""}


def _decode_choice_answer(answer_value: Any, choices: list[str]) -> tuple[str, str | None, int | None]:
    answer_raw = "" if _is_missing(answer_value) else str(answer_value).strip()
    if not answer_raw:
        return "", None, None

    upper = answer_raw.upper().strip()
    if len(upper) == 1 and "A" <= upper <= "Z":
        idx = ord(upper) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx], upper, idx
        return answer_raw, upper, None

    if upper.startswith("(") and upper.endswith(")") and len(upper) == 3:
        ch = upper[1]
        if "A" <= ch <= "Z":
            idx = ord(ch) - ord("A")
            if 0 <= idx < len(choices):
                return choices[idx], ch, idx

    return answer_raw, None, None


def _extract_mmmu_choices(record: dict[str, Any]) -> list[str]:
    if "choices" in record:
        choices = _to_choice_list(record.get("choices"))
        if choices:
            return choices
    if "options" in record:
        options = _to_choice_list(record.get("options"))
        if options:
            return options

    keyed: list[str] = []
    for k in [
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "option_e",
        "option_f",
        "option_g",
        "option_h",
        "option_i",
        "option_j",
    ]:
        v = record.get(k)
        if _is_missing(v):
            continue
        keyed.append(str(v))
    return keyed


def _normalize_mmmu_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    choices = _extract_mmmu_choices(record)
    answer_text, answer_label, answer_idx = _decode_choice_answer(record.get(args.answer_field), choices)
    if not answer_text:
        answer_text = _extract_answer(record, args)

    sample: dict[str, Any] = {
        "question": str(q),
        "answer": answer_text,
        "choices": choices,
        "dataset_type": "mmmu_pro",
    }
    if answer_label is not None:
        sample["answer_label"] = answer_label
    if answer_idx is not None:
        sample["correct_choice_idx"] = answer_idx

    image_candidates: list[dict[str, Any]] = []

    def _push_image_candidate(value: Any, source_key: str) -> None:
        if _is_missing(value):
            return
        if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
            image_candidates.append({
                "bytes": bytes(value.get("bytes")),
                "ext": Path(str(value.get("path", ".jpg"))).suffix or ".jpg",
                "source": source_key,
            })
            return
        if isinstance(value, str):
            if _is_placeholder_image_text(value):
                return
            vv = value.strip()
            if vv.startswith("http://") or vv.startswith("https://"):
                image_candidates.append({"url": vv, "source": source_key})
            else:
                image_candidates.append({"path": vv, "source": source_key})

    for key in [
        args.image_url_field,
        args.image_path_field,
        "image",
        "img_path",
        "image_1",
        "image_2",
        "image_3",
        "image_4",
        "image_5",
        "image_6",
        "image_7",
    ]:
        if not key or key not in record:
            continue
        _push_image_candidate(record.get(key), key)

    if image_candidates:
        sample["__image_candidates"] = image_candidates
        first = image_candidates[0]
        if "bytes" in first:
            sample["__image_bytes"] = first["bytes"]
            sample["__image_ext"] = first.get("ext", ".jpg")
        elif "url" in first:
            sample["image_url"] = first["url"]
        elif "path" in first:
            sample["image_path"] = first["path"]

    docs: list[dict[str, str]] = []
    hint_fields = ["context", "hint", "rationale", "solution", "explanation"]
    for k in hint_fields:
        v = record.get(k)
        if isinstance(v, str) and v.strip() and not _is_placeholder_image_text(v):
            docs.append({"source": k, "text": v})

    if not docs:
        docs_value = record.get(args.documents_field) if args.documents_field else None
        if isinstance(docs_value, list):
            docs = docs_value
        elif isinstance(docs_value, str) and docs_value.strip():
            docs = [{"source": args.documents_field or "context", "text": docs_value}]
    sample["documents"] = docs

    for k in [
        "id",
        "question_id",
        "image_id",
        "domain",
        "category",
        "subdomain",
        "topic",
        "options",
        "has_table",
        "has_chart",
    ]:
        if k in record:
            sample[k] = record[k]

    sample["__original_record"] = _jsonable(record)
    return sample


def _normalize_m3cot_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    choices = _to_choice_list(record.get("choices"))
    answer_text, answer_label, answer_idx = _decode_choice_answer(record.get(args.answer_field), choices)
    if not answer_text:
        answer_text = _extract_answer(record, args)

    sample: dict[str, Any] = {
        "question": str(q),
        "answer": answer_text,
        "choices": choices,
        "dataset_type": "m3cot",
    }
    if answer_label is not None:
        sample["answer_label"] = answer_label
    if answer_idx is not None:
        sample["correct_choice_idx"] = answer_idx

    for key in [args.image_url_field, args.image_path_field, "image", "img_path"]:
        if not key or key not in record:
            continue
        value = record.get(key)
        if _is_missing(value) or value == "":
            continue
        sample[key] = value

    docs: list[dict[str, str]] = []
    context = record.get("context")
    if isinstance(context, str) and context.strip():
        docs.append({"source": "context", "text": context})
    rationale = record.get("rationale")
    if isinstance(rationale, str) and rationale.strip():
        docs.append({"source": "rationale", "text": rationale})
    if not docs:
        docs_value = record.get(args.documents_field) if args.documents_field else None
        if isinstance(docs_value, list):
            docs = docs_value
        elif isinstance(docs_value, str) and docs_value.strip():
            docs = [{"source": args.documents_field or "context", "text": docs_value}]
    sample["documents"] = docs

    for k in ["reasoning_hops", "cot_steps", "steps", "domain", "options", "has_table", "has_chart", "topic", "id", "category"]:
        if k in record:
            sample[k] = record[k]

    sample["__original_record"] = _jsonable(record)
    return sample


def _normalize_cmmqa_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    choices_raw = record.get("choices")
    if _is_missing(choices_raw):
        choices_raw = record.get("options")
    choices = _to_choice_list(choices_raw)
    answer_text, answer_label, answer_idx = _decode_choice_answer(record.get(args.answer_field), choices)
    if not answer_text:
        answer_text = _extract_answer(record, args)

    sample: dict[str, Any] = {
        "question": str(q),
        "answer": answer_text,
        "choices": choices,
        "dataset_type": "cmmqa",
    }
    if answer_label is not None:
        sample["answer_label"] = answer_label
    if answer_idx is not None:
        sample["correct_choice_idx"] = answer_idx

    for key in [args.image_url_field, args.image_path_field, "image", "img_path", "image_url"]:
        if not key or key not in record:
            continue
        value = record.get(key)
        if _is_missing(value) or value == "":
            continue
        sample[key] = value

    docs: list[dict[str, str]] = []
    reasoning_chains = record.get("reasoning_chains")
    if isinstance(reasoning_chains, list) and reasoning_chains:
        chain_lines: list[str] = []
        for idx, chain in enumerate(reasoning_chains, start=1):
            if isinstance(chain, (list, tuple)) and len(chain) >= 3:
                chain_lines.append(f"{idx}. {chain[0]} --{chain[1]}--> {chain[2]}")
            elif isinstance(chain, str) and chain.strip():
                chain_lines.append(f"{idx}. {chain.strip()}")
        if chain_lines:
            docs.append({"source": "reasoning_chains", "text": "\n".join(chain_lines)})
    sample["documents"] = docs

    for k in ["ID", "id", "question_id", "image_id", "image", "image_url", "reasoning_chains"]:
        if k in record:
            sample[k] = record[k]

    sample["__original_record"] = _jsonable(record)
    return sample


def _normalize_scienceqa_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    choices_raw = record.get("choices")
    if _is_missing(choices_raw):
        choices_raw = record.get("options")
    choices = _to_choice_list(choices_raw)
    answer_value = record.get(args.answer_field)
    answer_text = ""
    answer_label: str | None = None
    answer_idx: int | None = None

    if isinstance(answer_value, int) and 0 <= answer_value < len(choices):
        answer_idx = int(answer_value)
        answer_label = chr(ord("A") + answer_idx)
        answer_text = choices[answer_idx]
    else:
        answer_text, answer_label, answer_idx = _decode_choice_answer(answer_value, choices)
    if not answer_text:
        answer_text = _extract_answer(record, args)

    sample: dict[str, Any] = {
        "question": str(q),
        "answer": answer_text,
        "choices": choices,
        "dataset_type": "scienceqa",
    }
    if answer_label is not None:
        sample["answer_label"] = answer_label
    if answer_idx is not None:
        sample["correct_choice_idx"] = answer_idx

    rid = None
    for key in ["id", "ID", "question_id", "__record_id"]:
        v = record.get(key)
        if _is_missing(v):
            continue
        rid = v
        break
    split_value = record.get("split")
    split = "train" if _is_missing(split_value) else str(split_value)
    image_name = record.get("image")
    if isinstance(image_name, str) and image_name.strip():
        if split and rid is not None:
            sample["image_path"] = f"{split}/{rid}/{image_name.strip()}"
        else:
            sample["image_path"] = image_name.strip()

    docs: list[dict[str, str]] = []
    for k in ["hint", "lecture", "solution"]:
        v = record.get(k)
        if isinstance(v, str) and v.strip():
            docs.append({"source": k, "text": v})
    sample["documents"] = docs

    for k in [
        "id",
        "ID",
        "split",
        "task",
        "grade",
        "subject",
        "topic",
        "category",
        "skill",
        "image",
    ]:
        if k in record:
            sample[k] = record[k]

    sample["__original_record"] = _jsonable(record)
    return sample


def _infer_dataset_type(args: argparse.Namespace, record: dict[str, Any], dataset_path: Path) -> str:
    if args.dataset_type != "auto":
        return args.dataset_type
    path_l = str(dataset_path).lower()
    if "scienceqa" in path_l:
        return "scienceqa"
    if "cmmqa" in path_l:
        return "cmmqa"
    if "mmmu" in path_l:
        return "mmmu_pro"
    if "m3cot" in path_l:
        return "m3cot"
    if {"question_id", "image_id"}.intersection(record.keys()) and (
        "options" in record or "choices" in record or "option_a" in record
    ):
        return "mmmu_pro"
    if {"rationale", "topic", "category", "image_id"}.intersection(record.keys()) and "choices" in record:
        return "m3cot"
    return "generic"


def _normalize_sample_record(record: dict[str, Any], args: argparse.Namespace, dataset_type: str = "generic") -> dict[str, Any]:
    if dataset_type == "mmmu_pro":
        return _normalize_mmmu_record(record, args)
    if dataset_type == "m3cot":
        return _normalize_m3cot_record(record, args)
    if dataset_type == "cmmqa":
        return _normalize_cmmqa_record(record, args)
    if dataset_type == "scienceqa":
        return _normalize_scienceqa_record(record, args)

    q = record.get(args.question_field)
    if _is_missing(q):
        raise ValueError(f"Missing question field: {args.question_field}")

    sample: dict[str, Any] = {"question": str(q), "answer": _extract_answer(record, args)}

    for key in [args.image_url_field, args.image_path_field, "image", "img_path"]:
        if not key or key not in record:
            continue
        value = record.get(key)
        if _is_missing(value) or value == "":
            continue

        if key == "image" and isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
            sample["__image_bytes"] = bytes(value.get("bytes"))
            sample["__image_ext"] = ".jpg"
            continue

        sample[key] = value

    docs_value = record.get(args.documents_field) if args.documents_field else None
    if isinstance(docs_value, list):
        sample["documents"] = docs_value
    elif isinstance(docs_value, str) and docs_value.strip():
        sample["documents"] = [{"source": args.documents_field or "context", "text": docs_value}]
    else:
        sample["documents"] = []

    for k in ["reasoning_hops", "cot_steps", "steps", "domain", "choices", "options", "has_table", "has_chart"]:
        if k in record:
            sample[k] = record[k]

    sample["dataset_type"] = "generic"
    sample["__original_record"] = _jsonable(record)
    return sample


def load_dataset(dataset_path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    if dataset_path.is_dir():
        parquet_files = sorted(dataset_path.glob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in dataset directory: {dataset_path}")
        merged_records: list[dict[str, Any]] = []
        for pf in parquet_files:
            merged_records.extend(load_dataset(pf, args))
        return merged_records

    suffix = dataset_path.suffix.lower()

    def _normalize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        vision_map = _load_mmmu_vision_map(args)
        out: list[dict[str, Any]] = []
        for x in records:
            if not isinstance(x, dict):
                continue
            ds_type = _infer_dataset_type(args, x, dataset_path)
            record = dict(x)
            if ds_type == "mmmu_pro":
                merged = _merge_mmmu_vision_record(record, vision_map, args)
                out.append(_normalize_sample_record(merged, args, dataset_type=ds_type))
            else:
                out.append(_normalize_sample_record(record, args, dataset_type=ds_type))
        return out

    if suffix == ".json":
        raw = json.loads(dataset_path.read_text())
        if isinstance(raw, list):
            return _normalize_records(raw)
        if isinstance(raw, dict):
            unpacked: list[dict[str, Any]] = []
            for k, v in raw.items():
                if isinstance(v, dict):
                    item = dict(v)
                    item.setdefault("__record_id", str(k))
                    item.setdefault("id", str(k))
                    unpacked.append(item)
            if unpacked:
                return _normalize_records(unpacked)
        raise ValueError("JSON dataset must be a list of objects or a dict[id->object].")

    if suffix == ".jsonl":
        out = []
        for line in dataset_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
        return _normalize_records(out)

    if suffix == ".parquet":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading parquet datasets requires pandas (and pyarrow/fastparquet). Please install requirements.txt first."
            ) from exc
        df = pd.read_parquet(dataset_path)
        records = df.to_dict(orient="records")
        return _normalize_records(records)

    raise ValueError(f"Unsupported dataset format: {suffix}. Use .json/.jsonl/.parquet")


def _vision_record_id(record: dict[str, Any], key_hint: str) -> str | None:
    candidates = [key_hint, "id", "question_id", "image_id", "sample_id"]
    for k in candidates:
        if not k:
            continue
        v = record.get(k)
        if _is_missing(v):
            continue
        s = str(v).strip()
        if s:
            return s
    return None


def _load_mmmu_vision_map(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    vision_path = args.mmmu_vision_parquet
    if vision_path is None:
        return {}
    if not vision_path.exists():
        return {}
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return {}

    vision_files: list[Path]
    if vision_path.is_dir():
        vision_files = sorted(vision_path.glob("*.parquet"))
    else:
        vision_files = [vision_path]
    if not vision_files:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for vf in vision_files:
        records = pd.read_parquet(vf).to_dict(orient="records")
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rid = _vision_record_id(rec, args.mmmu_vision_id_field)
            if rid:
                out[rid] = rec
    return out


def _merge_mmmu_vision_record(record: dict[str, Any], vision_map: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    if not vision_map:
        return record
    rid = _vision_record_id(record, args.mmmu_id_field)
    if not rid:
        return record
    vision = vision_map.get(rid)
    if not vision:
        return record

    merged = dict(record)
    image_field = args.mmmu_vision_image_field
    if image_field in vision and "image" not in merged and "image_path" not in merged and "img_path" not in merged:
        merged["image"] = vision.get(image_field)
    for k in ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"]:
        if k in vision and k not in merged:
            merged[k] = vision.get(k)
    if "image_path" in vision and "image_path" not in merged:
        merged["image_path"] = vision.get("image_path")
    if "image_url" in vision and "image_url" not in merged:
        merged["image_url"] = vision.get("image_url")
    return merged


def _image_path_to_data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = "image/jpeg"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _image_bytes_to_data_url(img_bytes: bytes, ext: str = ".jpg") -> str:
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime = mime_map.get(ext.lower(), "image/jpeg")
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def resolve_image_reference(sample: dict[str, Any], image_root: Path | None = None) -> tuple[str | None, str | None]:
    candidates = sample.get("__image_candidates")
    if isinstance(candidates, list):
        for c in candidates:
            if not isinstance(c, dict):
                continue
            if isinstance(c.get("url"), str) and c.get("url", "").strip():
                return c["url"].strip(), str(c.get("source", "image_url"))
            if isinstance(c.get("bytes"), (bytes, bytearray)):
                return _image_bytes_to_data_url(bytes(c["bytes"]), str(c.get("ext", ".jpg"))), str(c.get("source", "image_bytes"))
            if isinstance(c.get("path"), str) and c.get("path", "").strip():
                sample = dict(sample)
                sample["image_path"] = c["path"]
                break

    image_url = sample.get("image_url")
    if isinstance(image_url, str) and image_url.strip():
        return image_url.strip(), "image_url"

    img_bytes = sample.get("__image_bytes")
    if isinstance(img_bytes, (bytes, bytearray)):
        return _image_bytes_to_data_url(bytes(img_bytes), sample.get("__image_ext", ".jpg")), "image_bytes"

    candidate = sample.get("image_path") or sample.get("image") or sample.get("img_path")
    if isinstance(candidate, str) and candidate.strip():
        raw = candidate.strip()
        if _is_placeholder_image_text(raw):
            return None, None
        norm = raw.replace("\\", "/")
        path_candidates: list[Path] = []

        base = Path(norm)
        path_candidates.append(base)
        if image_root is not None:
            path_candidates.append(image_root / base)
            parts = base.parts
            if len(parts) >= 2 and parts[0].lower() == "data" and parts[1].lower() == "images":
                path_candidates.append(image_root / Path(*parts[2:]))
            path_candidates.append(image_root / base.name)

        seen: set[Path] = set()
        for p in path_candidates:
            try:
                rp = p.expanduser().resolve()
            except Exception:
                continue
            if rp in seen:
                continue
            seen.add(rp)
            if rp.exists():
                return _image_path_to_data_url(rp), "image_path"

    return None, None


def answer_with_context(
    llm: Any,
    question: str,
    context: str,
    image_ref: str | None = None,
    dataset_type: str = "generic",
    choices: list[str] | None = None,
) -> str:
    if llm is None:
        return context[:80] or "insufficient_context"

    if dataset_type in {"mmmu_pro", "cmmqa", "scienceqa"} and choices:
        opts = []
        max_opts = min(len(choices), 10)
        for i in range(max_opts):
            label = chr(ord("A") + i)
            opts.append(f"{label}. {choices[i]}")
        option_text = "\n".join(opts)
        if dataset_type == "cmmqa":
            prompt = (
                "你是CMMQA多模态知识问答评测助手。请先识别图像中的关键实体，再结合给定知识链上下文做两跳以上约束推理。"
                "仅输出一个选项字母（A-J），不要输出解释。\n"
                f"题目: {question}\n"
                f"选项:\n{option_text}\n"
                f"知识链上下文:\n{context[:1400]}"
            )
        elif dataset_type == "scienceqa":
            prompt = (
                "你是ScienceQA多模态评测助手。请先定位题目中的科学概念与图像线索，再结合提示(hint)/讲义(lecture)进行约束推理。"
                "仅输出一个选项字母（A-J），不要输出解释。\n"
                f"题目: {question}\n"
                f"选项:\n{option_text}\n"
                f"辅助上下文:\n{context[:1600]}"
            )
        else:
            prompt = (
                "你是MMMU-Pro多模态选择题评测助手。"
                "仅输出一个选项字母（A-J），不要输出解释。\n"
                f"题目: {question}\n"
                f"选项:\n{option_text}\n"
                f"辅助上下文:\n{context[:1200]}"
            )
    else:
        prompt = (
            "你是知识增强型多模态VQA助手。基于给定上下文回答问题，"
            "若上下文不足则明确说明。只输出简洁答案。\n"
            f"问题: {question}\n上下文:\n{context[:4000]}"
        )
    if image_ref:
        return llm.chat_with_image(prompt=prompt, image_url=image_ref, temperature=0.1)
    return llm.chat(
        [{"role": "system", "content": "你是VQA评测助手"}, {"role": "user", "content": prompt}],
        temperature=0.1,
    )


def refine_with_sft(
    llm: Any,
    question: str,
    first_answer: str,
    context: str,
    image_ref: str | None = None,
    dataset_type: str = "generic",
    choices: list[str] | None = None,
) -> str:
    if llm is None:
        return first_answer
    opts = []
    if choices:
        for i, ch in enumerate(choices[:10]):
            opts.append(f"{chr(ord('A') + i)}. {ch}")
    prompt = (
        "你是一次轻量SFT后精炼助手。请结合首轮答案和证据进行一次自校正，仅输出最终答案。"
        "如果是选择题，仅输出选项字母。\n"
        f"题目: {question}\n"
        f"首轮答案: {first_answer}\n"
        f"选项:\n{chr(10).join(opts)}\n"
        f"证据上下文:\n{context[:2000]}\n"
        f"数据集类型: {dataset_type}"
    )
    if image_ref:
        return llm.chat_with_image(prompt=prompt, image_url=image_ref, temperature=0.1)
    return llm.chat(
        [{"role": "system", "content": "你是VQA评测助手"}, {"role": "user", "content": prompt}],
        temperature=0.1,
    )


def run_eval(dataset_path: Path, args: argparse.Namespace) -> dict:
    data = load_dataset(dataset_path, args)
    ablation = _resolve_ablation(str(getattr(args, "ablation_profile", "all_on")))
    llm = None
    if not args.mock:
        from models import QwenVLClient

        llm = QwenVLClient()

    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    resume_path = args.resume_from
    if resume_path is not None:
        resumed = _load_resume_records(resume_path)
        for r in resumed:
            if not isinstance(r, dict):
                continue
            k = r.get("sample_key")
            if not isinstance(k, str) or not k:
                continue
            seen_keys.add(k)
            records.append(r)

    progress = tqdm(data, desc="Evaluating", unit="sample", dynamic_ncols=True)
    for sample in progress:
        sample_key = _record_key(sample)
        if sample_key in seen_keys:
            continue

        q = sample["question"]
        gold = sample["answer"]
        if ablation.use_difficulty:
            diff = estimate_question_difficulty(sample)
            route = _difficulty_route_tag(diff.overall) if ablation.use_kg_rag else "direct_inference"
        else:
            diff = estimate_question_difficulty(
                {
                    "dataset_type": str(sample.get("dataset_type", "generic")),
                    "question": str(sample.get("question", "")),
                    "choices": sample.get("choices", []),
                }
            )
            route = "full_rag" if ablation.use_kg_rag else "direct_inference"
        image_ref, image_source = resolve_image_reference(sample, image_root=args.image_root)
        docs = sample.get("documents", [])

        corpus = [(str(d.get("source", "unknown")), str(d.get("text", ""))) for d in docs if d.get("text")]
        baseline_context = "\n\n".join([f"[{s}] {t[:600]}" for s, t in corpus[:4]])
        if ablation.use_kg_rag:
            graph_hits, graph_context, route_meta = _build_route_aware_graph_context(route=route, question=q, corpus=corpus)
            hits, evidence_recall, evidence_precision = compute_evidence_metrics(graph_hits, docs, gold)
        else:
            graph_hits, graph_context, route_meta = [], "", {"rounds": 0, "top_k": 0, "expanded_query": None}
            hits, evidence_recall, evidence_precision = 0, 0.0, 0.0

        baseline_error = None
        graph_error = None
        try:
            baseline_pred = answer_with_context(
                llm,
                q,
                baseline_context,
                image_ref=image_ref,
                dataset_type=str(sample.get("dataset_type", "generic")),
                choices=_to_choice_list(sample.get("choices")),
            )
        except Exception as exc:  # keep eval running
            baseline_pred = ""
            baseline_error = str(exc)

        try:
            graph_pred = answer_with_context(
                llm,
                q,
                graph_context if graph_context else baseline_context,
                image_ref=image_ref,
                dataset_type=str(sample.get("dataset_type", "generic")),
                choices=_to_choice_list(sample.get("choices")),
            )
            if ablation.use_sft:
                graph_pred = refine_with_sft(
                    llm=llm,
                    question=q,
                    first_answer=graph_pred,
                    context=(graph_context if graph_context else baseline_context),
                    image_ref=image_ref,
                    dataset_type=str(sample.get("dataset_type", "generic")),
                    choices=_to_choice_list(sample.get("choices")),
                )
        except Exception as exc:
            graph_pred = ""
            graph_error = str(exc)

        rec = {
            "sample_key": sample_key,
            "question": q,
            "gold": gold,
            "baseline_pred": baseline_pred,
            "graph_pred": graph_pred,
            "baseline_exact": int(normalize(baseline_pred) == normalize(gold)),
            "graph_exact": int(normalize(graph_pred) == normalize(gold)),
            "baseline_f1": token_f1(baseline_pred, gold),
            "graph_f1": token_f1(graph_pred, gold),
            "hits": hits,
            "evidence_recall": evidence_recall,
            "evidence_precision": evidence_precision,
            "image_used": bool(image_ref),
            "image_source": image_source,
            "baseline_error": baseline_error,
            "graph_error": graph_error,
            "difficulty": {
                "overall": diff.overall,
                "qnorm_100": _qnorm_100(diff.overall),
                "bucket": "pending",
                "route": route,
                "task_family": _task_family(sample),
                "linguistic_complexity": diff.linguistic_complexity,
                "reasoning_complexity": diff.reasoning_complexity,
                "multimodal_complexity": diff.multimodal_complexity,
                "knowledge_intensity": diff.knowledge_intensity,
                "choice_ambiguity": diff.choice_ambiguity,
                "retrieval_hardness": diff.retrieval_hardness,
                "deepresearch_coverage": diff.deepresearch_coverage,
            },
            "retrieval_trace": route_meta,
            "ablation": {
                "profile": ablation.profile,
                "kg_rag": ablation.use_kg_rag,
                "difficulty_estimator": ablation.use_difficulty,
                "sft_refine": ablation.use_sft,
            },
            "verification": "Verified" if int(normalize(graph_pred) == normalize(gold)) == 1 else "Unverified",
            "original_record": sample.get("__original_record", {}),
            "normalized_sample": {
                k: _jsonable(v)
                for k, v in sample.items()
                if not k.startswith("__")
            },
        }
        records.append(rec)
        seen_keys.add(sample_key)

        if args.details_out is not None and args.save_every > 0 and len(records) % args.save_every == 0:
            args.details_out.parent.mkdir(parents=True, exist_ok=True)
            if args.details_format == "jsonl":
                with args.details_out.open("w", encoding="utf-8") as f:
                    for item in records:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                args.details_out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                baseline_err=sum(1 for r in records if r["baseline_error"]),
                graph_err=sum(1 for r in records if r["graph_error"]),
                img=sum(1 for r in records if r["image_used"]),
            )

    difficulty_values = [float(r.get("difficulty", {}).get("overall", 0.0)) for r in records]
    q1, q2 = _tertile_thresholds(difficulty_values)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        overall = float(r.get("difficulty", {}).get("overall", 0.0))
        b = difficulty_bucket(overall, q1=q1, q2=q2)
        r.setdefault("difficulty", {})["bucket"] = b
        r["sft_role"] = _sft_role(b, str(r.get("verification", "Unverified")))
        by_bucket[b].append(r)

    n = max(len(records), 1)
    score_components = {
        "overall": mean([r["difficulty"]["overall"] for r in records]) if records else 0.0,
        "linguistic_complexity": mean([r["difficulty"]["linguistic_complexity"] for r in records]) if records else 0.0,
        "reasoning_complexity": mean([r["difficulty"]["reasoning_complexity"] for r in records]) if records else 0.0,
        "multimodal_complexity": mean([r["difficulty"]["multimodal_complexity"] for r in records]) if records else 0.0,
        "knowledge_intensity": mean([r["difficulty"]["knowledge_intensity"] for r in records]) if records else 0.0,
        "choice_ambiguity": mean([r["difficulty"]["choice_ambiguity"] for r in records]) if records else 0.0,
        "retrieval_hardness": mean([r["difficulty"]["retrieval_hardness"] for r in records]) if records else 0.0,
        "deepresearch_coverage": mean([r["difficulty"]["deepresearch_coverage"] for r in records]) if records else 0.0,
        "image_input_coverage": (sum(1 for r in records if r["image_used"]) / n) if records else 0.0,
        "baseline_error_rate": (sum(1 for r in records if r["baseline_error"]) / n) if records else 0.0,
        "graph_error_rate": (sum(1 for r in records if r["graph_error"]) / n) if records else 0.0,
        "verified_rate": (sum(1 for r in records if r.get("verification") == "Verified") / n) if records else 0.0,
    }

    bucket_metrics = {}
    for bucket, items in by_bucket.items():
        m = max(len(items), 1)
        bucket_metrics[bucket] = {
            "count": len(items),
            "baseline_exact": sum(x["baseline_exact"] for x in items) / m,
            "graph_exact": sum(x["graph_exact"] for x in items) / m,
            "baseline_f1": sum(x["baseline_f1"] for x in items) / m,
            "graph_f1": sum(x["graph_f1"] for x in items) / m,
            "avg_difficulty": mean([x["difficulty"]["overall"] for x in items]) if items else 0.0,
            "image_input_coverage": sum(1 for x in items if x["image_used"]) / m,
        }

    return {
        "count": len(records),
        "baseline_exact": sum(r["baseline_exact"] for r in records) / n,
        "graph_exact": sum(r["graph_exact"] for r in records) / n,
        "baseline_f1": sum(r["baseline_f1"] for r in records) / n,
        "graph_f1": sum(r["graph_f1"] for r in records) / n,
        "hits": sum(r.get("hits", 0) for r in records) / n,
        "evidence_recall": sum(float(r.get("evidence_recall", 0.0)) for r in records) / n,
        "evidence_precision": sum(float(r.get("evidence_precision", 0.0)) for r in records) / n,
        "difficulty_scores": score_components,
        "difficulty_bucket_thresholds": {"q33": q1, "q67": q2},
        "route_stats": {
            "direct_inference": sum(1 for r in records if r.get("difficulty", {}).get("route") == "direct_inference"),
            "light_rag": sum(1 for r in records if r.get("difficulty", {}).get("route") == "light_rag"),
            "full_rag": sum(1 for r in records if r.get("difficulty", {}).get("route") == "full_rag"),
        },
        "verification_stats": {
            "verified": sum(1 for r in records if r.get("verification") == "Verified"),
            "unverified": sum(1 for r in records if r.get("verification") == "Unverified"),
            "hard_verified": sum(
                1
                for r in records
                if r.get("verification") == "Verified" and r.get("difficulty", {}).get("bucket") == "hard"
            ),
            "hard_unverified": sum(
                1
                for r in records
                if r.get("verification") == "Unverified" and r.get("difficulty", {}).get("bucket") == "hard"
            ),
        },
        "sft_sampling_stats": {
            "hard_unverified_primary": sum(1 for r in records if r.get("sft_role") == "hard_unverified_primary"),
            "hard_verified_aux": sum(1 for r in records if r.get("sft_role") == "hard_verified_aux"),
        },
        "accuracy_stats": {
            "baseline_correct": int(sum(r["baseline_exact"] for r in records)),
            "graph_correct": int(sum(r["graph_exact"] for r in records)),
            "total": int(len(records)),
            "baseline_accuracy": (sum(r["baseline_exact"] for r in records) / n),
            "graph_accuracy": (sum(r["graph_exact"] for r in records) / n),
        },
        "metrics_by_difficulty_bucket": bucket_metrics,
        "details": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lexical-graph enhanced multimodal VQA in DeepResearch")
    parser.add_argument("dataset", type=Path, help="Path to dataset file (.json/.jsonl/.parquet)")
    parser.add_argument("--out", type=Path, default=Path("vqa_eval_report.json"))
    parser.add_argument("--mock", action="store_true", help="Run without LLM API calls")
    parser.add_argument("--model-name", type=str, default=None, help="Override MODEL_NAME for this eval run")
    parser.add_argument("--openai-base-url", type=str, default=None, help="Override OPENAI_BASE_URL for this eval run")
    parser.add_argument("--openai-api-key", type=str, default=None, help="Override OPENAI_API_KEY for this eval run")
    parser.add_argument("--image-root", type=Path, default=None, help="Root folder for relative image paths")
    parser.add_argument("--question-field", type=str, default="question", help="Question column/key name")
    parser.add_argument("--answer-field", type=str, default="answer", help="Answer column/key name")
    parser.add_argument("--documents-field", type=str, default="documents", help="Documents/context column/key name")
    parser.add_argument("--image-url-field", type=str, default="image_url", help="Image URL column/key name")
    parser.add_argument("--image-path-field", type=str, default="image_path", help="Image path column/key name")
    parser.add_argument(
        "--dataset-type",
        choices=["auto", "generic", "m3cot", "mmmu_pro", "cmmqa", "scienceqa"],
        default="auto",
        help="Dataset parser type. auto keeps A-OKVQA logic and applies M3COT/MMMU-Pro/CMMQA/ScienceQA normalization when detected.",
    )
    parser.add_argument(
        "--ablation-profile",
        choices=["none", "kg_only", "difficulty_only", "kg_difficulty", "kg_sft", "difficulty_sft", "all_on"],
        default="all_on",
        help="Ablation assembly profile matching module switches: KG-RAG / difficulty estimator / SFT-refine.",
    )
    parser.add_argument(
        "--mmmu-vision-parquet",
        type=Path,
        default=None,
        help="Optional MMMU-Pro vision parquet file OR directory (joined for separated question/vision parquet setup).",
    )
    parser.add_argument(
        "--mmmu-id-field",
        type=str,
        default="id",
        help="ID field in MMMU-Pro question parquet used to join with --mmmu-vision-parquet.",
    )
    parser.add_argument(
        "--mmmu-vision-id-field",
        type=str,
        default="id",
        help="ID field in MMMU-Pro vision parquet used for join mapping.",
    )
    parser.add_argument(
        "--mmmu-vision-image-field",
        type=str,
        default="image",
        help="Image payload field in MMMU-Pro vision parquet (default: image).",
    )
    parser.add_argument(
        "--details-out",
        type=Path,
        default=None,
        help="Optional path to save per-sample complete records with original fields + eval result (.json or .jsonl)",
    )
    parser.add_argument(
        "--details-format",
        choices=["json", "jsonl"],
        default="json",
        help="Format for --details-out (default: json)",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume evaluation from an existing details file (.json/.jsonl). Already completed samples will be skipped.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0 and --details-out is set, periodically checkpoint details every N processed samples.",
    )
    args = parser.parse_args()

    if args.model_name:
        os.environ["MODEL_NAME"] = args.model_name
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    report = run_eval(args.dataset, args=args)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    if args.details_out is not None:
        details = report.get("details", [])
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        if args.details_format == "jsonl":
            with args.details_out.open("w", encoding="utf-8") as f:
                for item in details:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            args.details_out.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Global Metrics ===")
    print(json.dumps({
        "count": report["count"],
        "baseline_exact": report["baseline_exact"],
        "graph_exact": report["graph_exact"],
        "baseline_f1": report["baseline_f1"],
        "graph_f1": report["graph_f1"],
        "hits": report["hits"],
        "evidence_recall": report["evidence_recall"],
        "evidence_precision": report["evidence_precision"],
    }, ensure_ascii=False, indent=2))
    print("=== Accuracy Stats ===")
    print(json.dumps(report.get("accuracy_stats", {}), ensure_ascii=False, indent=2))
    print("=== Difficulty Scores ===")
    print(json.dumps(report["difficulty_scores"], ensure_ascii=False, indent=2))
    print("=== Route Stats ===")
    print(json.dumps(report.get("route_stats", {}), ensure_ascii=False, indent=2))
    print("=== Verification Stats ===")
    print(json.dumps(report.get("verification_stats", {}), ensure_ascii=False, indent=2))
    print("=== SFT Sampling Stats ===")
    print(json.dumps(report.get("sft_sampling_stats", {}), ensure_ascii=False, indent=2))
    print("=== Metrics By Difficulty Bucket ===")
    print(json.dumps(report["metrics_by_difficulty_bucket"], ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")
    if args.details_out is not None:
        print(f"saved details: {args.details_out} ({args.details_format})")


if __name__ == "__main__":
    main()

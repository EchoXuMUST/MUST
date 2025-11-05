"""自然语言处理相关的工具与模型加载封装。"""
from __future__ import annotations

import functools
import json
import logging
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from transformers import AutoConfig, AutoTokenizer, Pipeline, pipeline

try:  # noqa: SIM105
    import onnxruntime as ort
except Exception:  # pylint: disable=broad-except
    ort = None  # type: ignore


LOGGER = logging.getLogger(__name__)
ALIAS_KEYWORDS = [
    " aka ",
    "also known as",
    "better known as",
    "nicknamed",
    "nickname",
    "stage name",
    "alias",
    "real name",
]
PAREN_ALIAS_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9 .'-]+)\s*\(([^)]+)\)")

# 为了避免在评估流程中重复加载大模型，这里使用 LRU 缓存。


@functools.lru_cache(maxsize=1)
def _detect_preferred_cuda_device() -> Optional[int]:
    """检测可用的 GPU 设备，返回推荐的 device id。"""

    try:
        import torch

        if not torch.cuda.is_available():
            return None

        best_device: Optional[int] = None
        best_free: float = -1.0
        for idx in range(torch.cuda.device_count()):
            free_mem: Optional[float] = None
            try:
                free_mem, _ = torch.cuda.mem_get_info(idx)
            except TypeError:
                try:
                    with torch.cuda.device(idx):
                        free_mem, _ = torch.cuda.mem_get_info()
                except Exception:  # pylint: disable=broad-except
                    free_mem = None
            except Exception:  # pylint: disable=broad-except
                free_mem = None
            if free_mem is None:
                if best_device is None:
                    best_device = idx
                continue
            if free_mem > best_free:
                best_free = float(free_mem)
                best_device = idx
        if best_device is None:
            return 0
        return best_device
    except Exception:  # pylint: disable=broad-except
        return None


@functools.lru_cache(maxsize=1)
def get_default_device() -> int:
    """返回首选的计算设备，优先使用 GPU。"""

    device_id = _detect_preferred_cuda_device()
    if device_id is not None:
        LOGGER.info("Device set to use cuda:%s", device_id)
        return device_id
    LOGGER.info("Device set to use cpu")
    return -1


def _select_onnx_providers() -> List[object]:
    """根据当前设备选择合适的 ONNX Runtime Provider 列表。"""

    if ort is None:  # pragma: no cover - 调用方已检查
        return []
    available = list(ort.get_available_providers())  # type: ignore[union-attr]
    providers: List[object] = []
    device_id = get_default_device()
    if device_id >= 0 and "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": device_id}))
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    if not providers:
        providers = available
    return providers


@functools.lru_cache(maxsize=2)
def get_sentence_embedding_pipeline(model_dir: str) -> Pipeline:
    """构建句向量模型，用于 BM25 以外的相似度计算或备用方案。"""

    return pipeline(
        "feature-extraction",
        model=str(Path(model_dir)),
        tokenizer=str(Path(model_dir)),
        device=get_default_device(),
    )


@functools.lru_cache(maxsize=2)
def get_nli_pipeline(model_dir: str) -> Pipeline:
    """加载自然语言推理模型，返回 transformers 的文本分类 pipeline。"""

    return pipeline(
        "text-classification",
        model=str(Path(model_dir)),
        tokenizer=str(Path(model_dir)),
        device=get_default_device(),
        truncation=True,
        max_length=512,
        return_all_scores=True,
    )


class _OnnxTokenClassificationPipeline:
    """最小实现的 ONNX NER 推理管线，用于缺少 PyTorch 权重的场景。"""

    def __init__(
        self,
        session: "ort.InferenceSession",
        tokenizer: AutoTokenizer,
        id2label: Dict[int, str],
        max_length: int = 384,
    ) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.max_length = max_length
        self.input_names = {inp.name for inp in session.get_inputs()}

    def __call__(self, text: str) -> List[Dict[str, str]]:
        if not text.strip():
            return []
        try:
            encoding = self.tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            offset_mapping = encoding.pop("offset_mapping", None)
        except TypeError:
            encoding = self.tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            offset_mapping = encoding.pop("offset_mapping", None)
            encoding = {k: v.detach().cpu().numpy() for k, v in encoding.items()}
        inputs = {k: v for k, v in encoding.items() if k in self.input_names}
        if not inputs:
            return []
        outputs = self.session.run(None, inputs)
        if not outputs:
            return []
        logits = outputs[0]
        predictions = np.argmax(logits, axis=-1)[0]
        if offset_mapping is None:
            offset_mapping = self.tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=self.max_length,
            ).get("offset_mapping")
        if offset_mapping is None:
            return []
        if hasattr(offset_mapping, "numpy"):
            offset_mapping = offset_mapping.numpy()
        offset_array = np.array(offset_mapping)
        if offset_array.ndim == 3:
            offset_array = offset_array[0]
        entities: List[Dict[str, str]] = []
        for idx, (start, end) in enumerate(offset_array):
            if end <= start:
                continue
            label_idx = int(predictions[idx])
            label = self.id2label.get(label_idx) or self.id2label.get(str(label_idx))
            if not label:
                label = f"LABEL_{label_idx}"
            if not label or label.upper() == "O":
                continue
            entities.append(
                {
                    "entity_group": label,
                    "word": text[int(start) : int(end)],
                }
            )
        return entities


def _load_id2label(model_path: Path) -> Dict[int, str]:
    try:
        config = AutoConfig.from_pretrained(str(model_path))
        if hasattr(config, "id2label") and config.id2label:  # type: ignore[attr-defined]
            return {int(k): v for k, v in config.id2label.items()}  # type: ignore[attr-defined]
    except Exception:  # pylint: disable=broad-except
        pass
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            data = json.loads(config_file.read_text(encoding="utf-8"))
            id2label = data.get("id2label")
            if isinstance(id2label, dict):
                return {int(k): str(v) for k, v in id2label.items()}
        except Exception:  # pylint: disable=broad-except
            pass
    return {}


def _build_onnx_ner_pipeline(model_path: Path) -> Optional[_OnnxTokenClassificationPipeline]:
    if ort is None:
        LOGGER.warning("检测到 ONNX NER 权重，但缺少 onnxruntime 依赖，无法加载：%s", model_path)
        return None
    onnx_file: Optional[Path] = None
    default_dir = model_path / "onnx"
    if default_dir.is_dir():
        for candidate in sorted(default_dir.glob("*.onnx")):
            onnx_file = candidate
            break
        if onnx_file is None:
            nested = sorted(default_dir.rglob("*.onnx"))
            onnx_file = nested[0] if nested else None
    if onnx_file is None:
        candidates = sorted(model_path.glob("*.onnx"))
        onnx_file = candidates[0] if candidates else None
    if onnx_file is None:
        LOGGER.warning("未在 %s 下找到 ONNX 模型文件", model_path)
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("加载 ONNX NER 所需的 tokenizer 失败：%s", exc)
        return None
    id2label = _load_id2label(model_path)
    if not id2label:
        LOGGER.warning("未能解析 NER 标签映射，将继续但可能影响别名检测：%s", model_path)
        id2label = {}
    try:
        providers = _select_onnx_providers()
        session = ort.InferenceSession(str(onnx_file), providers=providers or None)  # type: ignore[call-arg]
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("初始化 ONNX NER 推理会话失败：%s", exc)
        return None
    return _OnnxTokenClassificationPipeline(session, tokenizer, id2label)


class _OnnxSequenceClassificationPipeline:
    """最小实现的 ONNX 文本分类推理管线。"""

    def __init__(
        self,
        session: "ort.InferenceSession",
        tokenizer: AutoTokenizer,
        id2label: Dict[int, str],
        max_length: int = 256,
    ) -> None:
        self.session = session
        self.tokenizer = tokenizer
        self.id2label = id2label or {}
        self.max_length = max_length
        self.input_names = {inp.name for inp in session.get_inputs()}

    def _tokenize(self, text: str) -> Dict[str, np.ndarray]:
        if not text:
            return {}
        try:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
        except TypeError:
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.detach().cpu().numpy() for k, v in encoded.items()}
        return {k: v for k, v in encoded.items() if k in self.input_names}

    def _label_for(self, idx: int) -> str:
        if idx in self.id2label:
            return self.id2label[idx]
        if str(idx) in self.id2label:
            return self.id2label[str(idx)]  # type: ignore[index]
        return f"LABEL_{idx}"

    def _infer_single(self, text: str) -> List[Dict[str, float]]:
        inputs = self._tokenize(text)
        if not inputs:
            return []
        try:
            outputs = self.session.run(None, inputs)
        except Exception:  # pylint: disable=broad-except
            return []
        if not outputs:
            return []
        logits = np.array(outputs[0])
        if logits.ndim == 1:
            logits = logits[None, :]
        logits = logits[0]
        probs = softmax(logits)
        results = [
            {"label": self._label_for(i), "score": float(score)}
            for i, score in enumerate(probs)
        ]
        results.sort(key=lambda item: item["score"], reverse=True)
        return results

    def __call__(self, inputs):  # type: ignore[override]
        if isinstance(inputs, (list, tuple)):
            return [self._infer_single(text) for text in inputs]
        return self._infer_single(str(inputs))


def _build_onnx_sequence_classifier(model_path: Path) -> Optional[_OnnxSequenceClassificationPipeline]:
    if ort is None:
        LOGGER.warning("检测到 ONNX 分类权重，但缺少 onnxruntime 依赖，无法加载：%s", model_path)
        return None
    onnx_file: Optional[Path] = None
    default_dir = model_path / "onnx"
    if default_dir.is_dir():
        candidates = sorted(default_dir.glob("*.onnx")) or sorted(default_dir.rglob("*.onnx"))
        onnx_file = candidates[0] if candidates else None
    if onnx_file is None:
        candidates = sorted(model_path.glob("*.onnx"))
        onnx_file = candidates[0] if candidates else None
    if onnx_file is None:
        LOGGER.warning("未在 %s 下找到 ONNX 模型文件", model_path)
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("加载 ONNX 分类模型所需的 tokenizer 失败：%s", exc)
        return None
    id2label = _load_id2label(model_path)
    try:
        providers = _select_onnx_providers()
        session = ort.InferenceSession(str(onnx_file), providers=providers or None)  # type: ignore[call-arg]
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("初始化 ONNX 分类推理会话失败：%s", exc)
        return None
    return _OnnxSequenceClassificationPipeline(session, tokenizer, id2label)


@functools.lru_cache(maxsize=2)
def get_ner_pipeline(model_dir: str) -> Optional[Pipeline]:
    """加载 NER 模型，用于别名/实体识别。

    优先尝试直接使用 ``transformers`` 的 ``pipeline`` 加载；若目录中
    仅包含 ONNX 权重，则自动切换到自实现的 ONNX 推理流程。
    如果两种方式均失败，则返回 ``None`` 以便上层采用启发式方案。"""

    model_path = Path(model_dir)
    try:
        return pipeline(
            "token-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            aggregation_strategy="simple",
            device=get_default_device(),
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.info("标准 NER 管线加载失败，尝试 ONNX 回退：%s", exc)
        onnx_pipe = _build_onnx_ner_pipeline(model_path)
        if onnx_pipe is not None:
            LOGGER.info("成功使用 ONNX 推理加载 NER 模型：%s", model_path)
            return onnx_pipe  # type: ignore[return-value]
        LOGGER.warning("加载 NER 模型 %s 失败，将回退到启发式别名检测：%s", model_path, exc)
        return None


TOXIC_KEYWORDS = [
    "kill",
    "hate",
    "stupid",
    "idiot",
    "violence",
    "racist",
    "terror",
    "bomb",
    "shoot",
    "abuse",
    "threat",
    "fuck",
    "bitch",
    "damn",
    "血腥",
    "杀死",
    "仇恨",
    "恐怖袭击",
    "炸弹",
    "枪击",
    "辱骂",
    "威胁",
]


@functools.lru_cache(maxsize=2)
def get_toxic_pipeline(model_dir: str) -> Optional[Pipeline]:
    """加载毒性检测模型，兼容仅提供 ONNX 权重的场景。"""

    model_path = Path(model_dir)
    try:
        return pipeline(
            "text-classification",
            model=str(model_path),
            tokenizer=str(model_path),
            top_k=None,
            device=get_default_device(),
            truncation=True,
            max_length=256,
            return_all_scores=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.info("标准毒性模型加载失败，尝试 ONNX 回退：%s", exc)
        onnx_pipe = _build_onnx_sequence_classifier(model_path)
        if onnx_pipe is not None:
            LOGGER.info("成功使用 ONNX 推理加载毒性模型：%s", model_path)
            return onnx_pipe  # type: ignore[return-value]
        LOGGER.warning(
            "加载毒性模型 %s 失败，将使用关键词启发式回退：%s",
            model_path,
            exc,
        )
        return None


def heuristic_toxicity_score(texts: Sequence[str]) -> float:
    """基于关键词匹配的简单毒性估计，作为模型缺失时的回退方案。"""

    hit_count = 0
    for text in texts:
        lower = text.lower()
        for keyword in TOXIC_KEYWORDS:
            if keyword in lower:
                hit_count += 1
    if hit_count == 0:
        return 0.0
    # 命中次数越多，风险越高，这里做一个简单的饱和归一化
    return min(hit_count / 3.0, 1.0)


def softmax(x: Sequence[float]) -> np.ndarray:
    """简单的 softmax 实现。"""

    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)
    e_x = np.exp(x)
    return e_x / e_x.sum()


def nli_entail_probability(nli_pipe: Pipeline, premise: str, hypothesis: str) -> float:
    """利用 NLI 模型估计蕴含概率。"""

    result = nli_pipe((premise, hypothesis))
    if isinstance(result, list) and result and isinstance(result[0], list):
        flat = result[0]
    else:
        flat = result
    if isinstance(flat, list) and flat:
        scores = {item["label"].lower(): item["score"] for item in flat if isinstance(item, dict)}
    elif isinstance(flat, dict):
        scores = {flat.get("label", "").lower(): flat.get("score", 0.0)}
    else:
        return 0.0
    entail = scores.get("entailment") or scores.get("entail")
    if entail is not None:
        return float(entail)
    # 若模型未直接给出概率，则尝试 softmax 正规化
    if len(scores) >= 3:
        probs = softmax(list(scores.values()))
        labels = list(scores.keys())
        for prob, label in zip(probs, labels):
            if "entail" in label:
                return float(prob)
    return 0.0


def extract_entities(ner_pipe: Optional[Pipeline], texts: Sequence[str]) -> List[str]:
    """从文本序列中抽取实体，返回实体列表。"""

    if ner_pipe is None:
        return []
    entities: List[str] = []
    for text in texts:
        if not text.strip():
            continue
        try:
            results = ner_pipe(text)
        except Exception:
            continue
        for item in results:
            entity = item.get("word") or item.get("entity_group")
            if entity:
                entities.append(entity.lower())
    return entities


def has_numeric_signal(texts: Sequence[str]) -> bool:
    """判断文本序列中是否包含数字或比较词信号。"""

    numeric_pattern = re.compile(r"\d{2,4}|\b(before|after|older|younger|greater|less|more|earlier|later|first|second|highest|lowest|> |<)\b", re.IGNORECASE)
    for text in texts:
        if numeric_pattern.search(text):
            return True
    return False


def has_temporal_signal(texts: Sequence[str]) -> bool:
    """判断文本序列是否包含时间顺序词。"""

    temporal_keywords = [
        "then",
        "later",
        "earlier",
        "before",
        "after",
        "first",
        "second",
        "finally",
        "eventually",
        "subsequently",
    ]
    for text in texts:
        lower_text = text.lower()
        if any(keyword in lower_text for keyword in temporal_keywords):
            return True
    return False


def alias_detected(entities: Sequence[str]) -> bool:
    """简单判断是否存在别名（同实体不同表述）。"""

    normalized = [re.sub(r"[^a-z0-9]+", " ", ent.lower()).strip() for ent in entities]
    normalized = [ent for ent in normalized if ent]
    seen = set()
    for ent in normalized:
        for other in seen:
            if ent == other:
                continue
            if levenshtein_ratio(ent, other) > 0.75 and ent.split()[0] != other.split()[0]:
                return True
        seen.add(ent)
    return False


def levenshtein_ratio(a: str, b: str) -> float:
    """计算两个字符串的 Levenshtein 相似度比例。"""

    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    len_a, len_b = len(a), len(b)
    dp = [[0] * (len_b + 1) for _ in range(len_a + 1)]
    for i in range(len_a + 1):
        dp[i][0] = i
    for j in range(len_b + 1):
        dp[0][j] = j
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    distance = dp[len_a][len_b]
    return 1 - distance / max(len_a, len_b)


def heuristic_alias_from_texts(texts: Sequence[str]) -> bool:
    """基于关键词与括号结构的启发式别名检测。"""

    for text in texts:
        lower = text.lower()
        if any(keyword in lower for keyword in ALIAS_KEYWORDS):
            return True
        for match in PAREN_ALIAS_PATTERN.finditer(text):
            left, right = match.group(1).strip(), match.group(2).strip()
            if not left or not right:
                continue
            left_norm = re.sub(r"[^a-z0-9]+", " ", left.lower()).strip()
            right_norm = re.sub(r"[^a-z0-9]+", " ", right.lower()).strip()
            if left_norm and right_norm and left_norm != right_norm:
                return True
    return False


TEMPORAL_KEYWORDS = [
    "then",
    "later",
    "earlier",
    "before",
    "after",
    "first",
    "second",
    "third",
    "finally",
]


SENSITIVE_LEVELS = {
    "低风险": 0.33,
    "中风险": 0.66,
    "高风险": 1.0,
}

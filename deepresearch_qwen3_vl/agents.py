from __future__ import annotations

from dataclasses import dataclass, field

from config import settings
from models import QwenVLClient
from tools import ToolObservation, ToolRouter


@dataclass
class Evidence:
    source: str
    claim: str
    excerpt: str
    confidence: float


@dataclass
class SubQuestion:
    question: str
    intent: str
    priority: int


@dataclass
class CritiqueResult:
    sufficient: bool
    follow_up_questions: list[str] = field(default_factory=list)
    missing_dimensions: list[str] = field(default_factory=list)


@dataclass
class ResearchState:
    question: str
    sub_questions: list[SubQuestion] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    tool_logs: list[ToolObservation] = field(default_factory=list)


class PlannerAgent:
    def __init__(self, llm: QwenVLClient) -> None:
        self.llm = llm

    def plan(self, question: str) -> list[SubQuestion]:
        prompt = (
            "将主问题拆解为3-6个子问题，并给每个子问题标注 intent 和 priority。"
            "输出JSON数组，元素格式："
            '{"question":"...","intent":"背景/对比/证据/反例/趋势","priority":1}。'
        )
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ],
            fallback=[],
        )

        sub_questions: list[SubQuestion] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                q = str(item.get("question", "")).strip()
                if not q:
                    continue
                sub_questions.append(
                    SubQuestion(
                        question=q,
                        intent=str(item.get("intent", "evidence")),
                        priority=int(item.get("priority", 3)),
                    )
                )
        if not sub_questions:
            sub_questions = [SubQuestion(question=question, intent="evidence", priority=1)]
        return sorted(sub_questions, key=lambda x: x.priority)


class ResearcherAgent:
    """Iterative agent that decides tool actions in ReAct-like JSON steps."""

    def __init__(self, llm: QwenVLClient, tool_router: ToolRouter) -> None:
        self.llm = llm
        self.tool_router = tool_router

    def _next_action(self, main_question: str, sub_question: SubQuestion, logs: list[ToolObservation]) -> dict:
        tool_history = "\n\n".join(
            [f"{idx+1}. {log.tool_name} 输入={log.input_data} 输出={log.output_data[:500]}" for idx, log in enumerate(logs)]
        )
        prompt = (
            "你是研究执行Agent。你必须从以下工具中选择一个行动：\n"
            "1) search_web {query, top_k}\n"
            "2) open_webpage {url}\n"
            "3) extract_images {url}\n"
            "4) analyze_image {image_url, question}\n"
            "5) ocr_image {image_url}\n"
            "6) retrieve_lexical_graph {question, documents, top_k}\n"
            "7) retrieve_graph_evidence {question, schema, graph_context}\n"
            "8) finish {note}\n"
            "仅输出JSON对象，格式："
            '{"tool":"search_web","args":{...},"reason":"..."}'
        )
        return self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": (
                        f"主问题: {main_question}\n"
                        f"当前子问题: {sub_question.question}\n"
                        f"子问题意图: {sub_question.intent}\n"
                        f"已有工具观察:\n{tool_history or '无'}"
                    ),
                },
            ],
            fallback={"tool": "search_web", "args": {"query": sub_question.question, "top_k": 5}, "reason": "fallback"},
        )

    def _distill_evidence(self, main_question: str, sub_question: str, logs: list[ToolObservation]) -> list[Evidence]:
        compact_logs = "\n\n".join(
            [f"工具={log.tool_name}\n输入={log.input_data}\n输出={log.output_data[:1200]}" for log in logs]
        )
        prompt = (
            "从工具观察中提炼证据。输出JSON数组，元素格式："
            '{"source":"来源URL或标识","claim":"证据支持的陈述","excerpt":"关键摘录","confidence":0.0-1.0}。'
            "要求：最多4条，避免重复。"
        )
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"主问题: {main_question}\n子问题: {sub_question}\n观察:\n{compact_logs}",
                },
            ],
            fallback=[],
        )

        evidence: list[Evidence] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                claim = str(item.get("claim", "")).strip()
                if not claim:
                    continue
                confidence = float(item.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                evidence.append(
                    Evidence(
                        source=str(item.get("source", "unknown")),
                        claim=claim,
                        excerpt=str(item.get("excerpt", "")),
                        confidence=confidence,
                    )
                )
        return evidence[:4]

    def investigate(self, main_question: str, sub_question: SubQuestion) -> tuple[list[Evidence], list[ToolObservation]]:
        logs: list[ToolObservation] = []

        for _ in range(settings.max_actions_per_question):
            action = self._next_action(main_question=main_question, sub_question=sub_question, logs=logs)
            tool_name = str(action.get("tool", "finish"))
            args = action.get("args", {}) if isinstance(action.get("args"), dict) else {}

            if tool_name == "finish":
                break

            try:
                obs = self.tool_router.run(tool_name=tool_name, args=args)
            except Exception as exc:
                obs = ToolObservation(
                    tool_name=tool_name,
                    input_data=args,
                    output_data=f"TOOL_ERROR: {exc}",
                )
            logs.append(obs)

        # Lexical-graph style enhancement: traverse cross-document evidence for the sub-question.
        docs: list[dict] = []
        for lg in logs:
            if lg.tool_name == "open_webpage":
                url = str(lg.input_data.get("url", "unknown"))
                if lg.output_data:
                    docs.append({"source": url, "text": lg.output_data})
        if docs:
            try:
                graph_obs = self.tool_router.run(
                    tool_name="retrieve_lexical_graph",
                    args={"question": sub_question.question, "documents": docs, "top_k": 3},
                )
                logs.append(graph_obs)
            except Exception as exc:
                logs.append(
                    ToolObservation(
                        tool_name="retrieve_lexical_graph",
                        input_data={"question": sub_question.question},
                        output_data=f"TOOL_ERROR: {exc}",
                    )
                )

            try:
                graph_context = "\n".join([f"[{d['source']}] {d['text'][:300]}" for d in docs[:3]])
                kg_obs = self.tool_router.run(
                    tool_name="retrieve_graph_evidence",
                    args={
                        "question": sub_question.question,
                        "schema": {
                            "summary": f"Question-driven proxy schema for: {main_question}",
                            "node_types": ["Entity", "ImageObject", "OCRText"],
                            "relation_types": ["mentions", "depicts", "located_in", "related_to", "supports"],
                        },
                        "graph_context": graph_context,
                    },
                )
                logs.append(kg_obs)
            except Exception as exc:
                logs.append(
                    ToolObservation(
                        tool_name="retrieve_graph_evidence",
                        input_data={"question": sub_question.question},
                        output_data=f"TOOL_ERROR: {exc}",
                    )
                )

        evidence = self._distill_evidence(main_question=main_question, sub_question=sub_question.question, logs=logs)
        return evidence, logs


class CriticAgent:
    def __init__(self, llm: QwenVLClient) -> None:
        self.llm = llm

    def critique(self, question: str, evidence: list[Evidence]) -> CritiqueResult:
        evidence_brief = "\n\n".join(
            [
                f"来源: {e.source}\n结论: {e.claim}\n摘录: {e.excerpt[:300]}\n置信度: {e.confidence}"
                for e in evidence[:12]
            ]
        )
        prompt = (
            "判断当前证据是否足够回答主问题。输出JSON对象："
            '{"sufficient":true/false,"follow_up_questions":[...],"missing_dimensions":[...]}。'
        )
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"问题: {question}\n\n证据:\n{evidence_brief}"},
            ],
            fallback={"sufficient": False, "follow_up_questions": [question], "missing_dimensions": ["fallback"]},
        )

        return CritiqueResult(
            sufficient=bool(data.get("sufficient", False)),
            follow_up_questions=[str(x) for x in data.get("follow_up_questions", [])][:3],
            missing_dimensions=[str(x) for x in data.get("missing_dimensions", [])][:4],
        )


class VerifierAgent:
    def __init__(self, llm: QwenVLClient) -> None:
        self.llm = llm

    def verify(self, question: str, evidence: list[Evidence]) -> list[Evidence]:
        """Basic evidence pruning to reduce hallucinated or weak claims."""
        packed = "\n\n".join(
            [
                f"[{idx}] 来源={e.source}\n陈述={e.claim}\n摘录={e.excerpt[:250]}\n置信度={e.confidence}"
                for idx, e in enumerate(evidence)
            ]
        )
        prompt = (
            "对每条证据进行有效性筛选。返回JSON数组，元素为应保留的索引（整数）。"
            "标准：与主问题相关、有来源、有可验证摘录。"
        )
        data = self.llm.chat_json(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"主问题: {question}\n\n证据候选:\n{packed}"},
            ],
            fallback=list(range(min(len(evidence), 6))),
        )

        if not isinstance(data, list):
            return evidence[:6]

        keep: list[Evidence] = []
        for idx in data:
            if isinstance(idx, int) and 0 <= idx < len(evidence):
                keep.append(evidence[idx])
        return keep[:10] if keep else evidence[:6]


class SynthesizerAgent:
    def __init__(self, llm: QwenVLClient) -> None:
        self.llm = llm

    def synthesize(self, state: ResearchState) -> str:
        evidence_text = "\n\n".join(
            [
                f"来源: {e.source}\n陈述: {e.claim}\n摘录: {e.excerpt[:300]}\n置信度: {e.confidence}"
                for e in state.evidence[:12]
            ]
        )
        prompt = (
            "你是深度研究报告撰写Agent。输出结构化中文报告：\n"
            "1) 执行摘要\n2) 分析框架\n3) 关键证据（按来源编号）\n"
            "4) 结论与适用边界\n5) 风险与不确定性\n6) 后续研究建议。\n"
            "必须引用提供的证据，不得凭空编造。"
        )
        return self.llm.chat(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": f"主问题: {state.question}\n\n证据库:\n{evidence_text}",
                },
            ],
            temperature=0.1,
        )

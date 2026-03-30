from __future__ import annotations

from agents import (
    CriticAgent,
    PlannerAgent,
    ResearchState,
    ResearcherAgent,
    SubQuestion,
    SynthesizerAgent,
    VerifierAgent,
)
from config import settings


class DeepResearchWorkflow:
    """
    DeepResearch-like pipeline:
    plan -> investigate(with tool actions) -> critique -> iterate -> verify -> synthesize
    """

    def __init__(
        self,
        planner: PlannerAgent,
        researcher: ResearcherAgent,
        critic: CriticAgent,
        verifier: VerifierAgent,
        synthesizer: SynthesizerAgent,
    ) -> None:
        self.planner = planner
        self.researcher = researcher
        self.critic = critic
        self.verifier = verifier
        self.synthesizer = synthesizer

    @staticmethod
    def _score_evidence_importance(question: str, evidence: Evidence) -> float:
        q_terms = {t for t in question.lower().split() if t}
        text = f"{evidence.claim} {evidence.excerpt}".lower()
        overlap = sum(1 for t in q_terms if t in text)
        overlap_score = overlap / max(len(q_terms), 1)
        return 0.7 * evidence.confidence + 0.3 * overlap_score

    def _compress_state_memory(self, state: ResearchState) -> None:
        if len(state.evidence) > settings.max_evidence_items:
            ranked = sorted(
                state.evidence,
                key=lambda e: self._score_evidence_importance(state.question, e),
                reverse=True,
            )
            state.evidence = ranked[: settings.max_evidence_items]

        if len(state.tool_logs) > settings.max_tool_logs:
            state.tool_logs = state.tool_logs[-settings.max_tool_logs :]

    def run(self, question: str) -> str:
        state = ResearchState(question=question)
        state.sub_questions = self.planner.plan(question)

        for _ in range(settings.max_rounds):
            current_questions = state.sub_questions[:]
            state.sub_questions = []

            for sq in current_questions:
                ev, logs = self.researcher.investigate(main_question=state.question, sub_question=sq)
                state.evidence.extend(ev)
                state.tool_logs.extend(logs)
                self._compress_state_memory(state)

            critique = self.critic.critique(question=state.question, evidence=state.evidence)
            if critique.sufficient:
                break

            if critique.follow_up_questions:
                state.sub_questions = [
                    SubQuestion(question=q, intent="gap_fill", priority=1) for q in critique.follow_up_questions
                ]
            else:
                state.sub_questions = [SubQuestion(question=state.question, intent="fallback", priority=1)]

        state.evidence = self.verifier.verify(question=state.question, evidence=state.evidence)
        return self.synthesizer.synthesize(state)

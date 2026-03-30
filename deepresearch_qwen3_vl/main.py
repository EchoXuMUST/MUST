from __future__ import annotations

import argparse

from agents import CriticAgent, PlannerAgent, ResearcherAgent, SynthesizerAgent, VerifierAgent
from models import QwenVLClient
from tools import SearchTool, ToolRouter, VisionTool, WebpageTool
from workflow import DeepResearchWorkflow


def build_workflow() -> DeepResearchWorkflow:
    llm = QwenVLClient()

    search_tool = SearchTool()
    webpage_tool = WebpageTool()
    vision_tool = VisionTool(vl_client=llm)
    tool_router = ToolRouter(search_tool=search_tool, webpage_tool=webpage_tool, vision_tool=vision_tool)

    planner = PlannerAgent(llm=llm)
    researcher = ResearcherAgent(llm=llm, tool_router=tool_router)
    critic = CriticAgent(llm=llm)
    verifier = VerifierAgent(llm=llm)
    synthesizer = SynthesizerAgent(llm=llm)

    return DeepResearchWorkflow(
        planner=planner,
        researcher=researcher,
        critic=critic,
        verifier=verifier,
        synthesizer=synthesizer,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepResearch with Qwen3-VL-8B-Instruct")
    parser.add_argument("question", type=str, help="research question")
    args = parser.parse_args()

    workflow = build_workflow()
    report = workflow.run(args.question)

    print("\n=== Research Report ===\n")
    print(report)


if __name__ == "__main__":
    main()

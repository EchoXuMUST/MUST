from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from byokg_rag import BYOKGRAGProvider
from config import settings
from lexical_graph import LexicalGraphRetriever, build_graph_from_corpus
from models import QwenVLClient


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str


@dataclass
class ToolObservation:
    tool_name: str
    input_data: dict
    output_data: str


class SearchTool:
    """Web search with custom API first and DuckDuckGo fallback."""

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if settings.search_api_url and settings.search_api_key:
            return self._search_custom(query=query, top_k=top_k)
        return self._search_duckduckgo(query=query, top_k=top_k)

    def _search_custom(self, query: str, top_k: int) -> list[SearchResult]:
        headers = {"Authorization": f"Bearer {settings.search_api_key}"}
        payload = {"q": query, "top_k": top_k}
        with httpx.Client(timeout=20) as client:
            resp = client.post(settings.search_api_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        return [
            SearchResult(
                title=item.get("title", ""),
                link=item.get("url", ""),
                snippet=item.get("snippet", ""),
            )
            for item in data.get("results", [])[:top_k]
        ]

    def _search_duckduckgo(self, query: str, top_k: int) -> list[SearchResult]:
        params = {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
        with httpx.Client(timeout=20) as client:
            resp = client.get("https://api.duckduckgo.com/", params=params)
            resp.raise_for_status()
            data = resp.json()

        results: list[SearchResult] = []
        if data.get("AbstractText") or data.get("AbstractURL"):
            results.append(
                SearchResult(
                    title=data.get("Heading", query),
                    link=data.get("AbstractURL", ""),
                    snippet=data.get("AbstractText", ""),
                )
            )

        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text") and topic.get("FirstURL"):
                results.append(
                    SearchResult(
                        title=topic.get("Text", "").split(" - ")[0],
                        link=topic.get("FirstURL", ""),
                        snippet=topic.get("Text", ""),
                    )
                )
            if len(results) >= top_k:
                break
        return results[:top_k]


class WebpageTool:
    def fetch_text(self, url: str, max_chars: int = 5000) -> str:
        if not url:
            return ""
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:max_chars]

    def extract_image_urls(self, url: str, max_images: int = 4) -> list[str]:
        if not url:
            return []
        with httpx.Client(timeout=20, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        image_urls: list[str] = []
        for img in soup.find_all("img"):
            src = img.get("src")
            if not src:
                continue
            absolute = urljoin(url, src)
            if re.search(r"\.(png|jpg|jpeg|webp)(\?|$)", absolute.lower()):
                image_urls.append(absolute)
            if len(image_urls) >= max_images:
                break
        return image_urls


class VisionTool:
    """Use Qwen3-VL to summarize image evidence."""

    def __init__(self, vl_client: QwenVLClient) -> None:
        self.vl_client = vl_client

    def describe_image(self, image_url: str, instruction: str = "提取图片中的关键信息并给出可信摘要") -> str:
        prompt = (
            f"{instruction}。要求：\n"
            "1) 描述可见事实；\n"
            "2) 标注不确定项；\n"
            "3) 指出与研究问题直接相关的证据。"
        )
        return self.vl_client.chat_with_image(prompt=prompt, image_url=image_url)

    def ocr_image(self, image_url: str) -> str:
        prompt = (
            "请对图片执行OCR并结构化输出可读文字。要求：\n"
            "1) 只提取图中能明确辨认的文字；\n"
            "2) 按区域或阅读顺序组织；\n"
            "3) 若无文字则输出 NO_TEXT。"
        )
        return self.vl_client.chat_with_image(prompt=prompt, image_url=image_url)


class LexicalGraphTool:
    """Knowledge-enhanced retrieval for DeepResearch using lexical-graph ideas."""

    def retrieve_from_documents(self, question: str, documents: list[tuple[str, str]], top_k: int = 5) -> list[str]:
        index = build_graph_from_corpus(documents)
        retriever = LexicalGraphRetriever(index)
        results = retriever.retrieve(question=question, top_k=top_k)
        return [
            f"[{i+1}] source={r.source} score={r.score:.3f} reasons={','.join(r.reasons)} text={r.text[:500]}"
            for i, r in enumerate(results)
        ]


class ToolRouter:
    """Unified tool interface to support iterative agent tool-use."""

    def __init__(
        self,
        search_tool: SearchTool,
        webpage_tool: WebpageTool,
        vision_tool: VisionTool,
        lexical_graph_tool: LexicalGraphTool | None = None,
        byokg_provider: BYOKGRAGProvider | None = None,
    ) -> None:
        self.search_tool = search_tool
        self.webpage_tool = webpage_tool
        self.vision_tool = vision_tool
        self.lexical_graph_tool = lexical_graph_tool or LexicalGraphTool()
        self.byokg_provider = byokg_provider or BYOKGRAGProvider()

    def run(self, tool_name: str, args: dict) -> ToolObservation:
        if tool_name == "search_web":
            query = str(args.get("query", "")).strip()
            top_k = int(args.get("top_k", settings.max_snippets_per_round))
            results = self.search_tool.search(query=query, top_k=top_k)
            compact = "\n".join(
                [f"[{i+1}] {r.title} | {r.link} | {r.snippet}" for i, r in enumerate(results)]
            )
            return ToolObservation(tool_name=tool_name, input_data=args, output_data=compact)

        if tool_name == "open_webpage":
            url = str(args.get("url", "")).strip()
            content = self.webpage_tool.fetch_text(url=url)
            return ToolObservation(tool_name=tool_name, input_data=args, output_data=content)

        if tool_name == "extract_images":
            url = str(args.get("url", "")).strip()
            images = self.webpage_tool.extract_image_urls(url=url)
            return ToolObservation(tool_name=tool_name, input_data=args, output_data="\n".join(images))


        if tool_name == "retrieve_lexical_graph":
            question = str(args.get("question", "")).strip()
            docs = args.get("documents", [])
            top_k = int(args.get("top_k", settings.max_snippets_per_round))
            normalized_docs: list[tuple[str, str]] = []
            if isinstance(docs, list):
                for d in docs:
                    if isinstance(d, dict):
                        source = str(d.get("source", "unknown"))
                        text = str(d.get("text", ""))
                        if text:
                            normalized_docs.append((source, text))
            out = self.lexical_graph_tool.retrieve_from_documents(question=question, documents=normalized_docs, top_k=top_k)
            return ToolObservation(tool_name=tool_name, input_data=args, output_data="\n".join(out))

        if tool_name == "analyze_image":
            image_url = str(args.get("image_url", "")).strip()
            question = str(args.get("question", "")).strip() or "提取图片关键证据"
            result = self.vision_tool.describe_image(image_url=image_url, instruction=question)
            return ToolObservation(tool_name=tool_name, input_data=args, output_data=result)

        if tool_name == "ocr_image":
            image_url = str(args.get("image_url", "")).strip()
            result = self.vision_tool.ocr_image(image_url=image_url)
            return ToolObservation(tool_name=tool_name, input_data=args, output_data=result)

        if tool_name == "retrieve_graph_evidence":
            question = str(args.get("question", "")).strip()
            schema = args.get("schema", {})
            graph_context = args.get("graph_context")
            if not isinstance(schema, dict):
                schema = {}
            result = self.byokg_provider.retrieve_graph_evidence(
                question=question,
                schema=schema,
                graph_context=str(graph_context) if graph_context is not None else None,
            )
            return ToolObservation(tool_name=tool_name, input_data=args, output_data=str(result))

        raise ValueError(f"Unsupported tool: {tool_name}")

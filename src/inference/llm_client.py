"""Local and optional OpenAI-compatible citation-grounded answer synthesis."""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List
from urllib.parse import urlsplit, urlunsplit

import httpx

from src.config import settings
from src.adapters.circuit_breaker import CircuitBreaker
from src.domain.citations import parse_citations
from src.domain.documents import RetrievedDocument
from src.usecases.synthesis import (
    DOCUMENTS_QUESTION_RE,
    META_QUESTION_RE,
    META_RESPONSE,
    LocalSynthesizer,
)

logger = logging.getLogger(__name__)
REMOTE_SYSTEM_PROMPT = (
    "You are a financial document analyst. Answer ONLY from the supplied EVIDENCE. "
    "Every factual claim must end with a citation in the exact form [DOC_NAME, Page N]. "
    "Never invent numbers, pages, documents, or sources. If the evidence does not answer "
    "the question, say that the information is not present in the indexed documents. "
    "Prefer concise bullets and preserve table labels and units verbatim."
)


class UnifiedLLMClient:
    """Use deterministic synthesis or a configured remote provider."""

    def __init__(
        self,
        base_url: str = settings.LLM_BASE_URL,
        api_key: str = settings.LLM_API_KEY,
        model_name: str = settings.LLM_MODEL_NAME,
        *,
        remote_enabled: bool | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.remote_enabled = (
            settings.LLM_REMOTE_ENABLED if remote_enabled is None else remote_enabled
        )
        self.timeout_seconds = timeout_seconds or settings.LLM_TIMEOUT_SECONDS
        self._transport = transport
        self._http_client: httpx.AsyncClient | None = None
        self._http_client_loop: asyncio.AbstractEventLoop | None = None
        self.last_provider = "local_deterministic"
        self.last_error: str | None = None
        self.remote_requests = 0
        self.remote_failures = 0
        self.local = LocalSynthesizer()
        # A dead provider must not cost every query its connect timeout: two failures open the
        # breaker and queries fall back locally until the recovery window elapses.
        self._remote_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=30.0)

    def configure_provider(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        enabled: bool | None = None,
    ) -> Dict[str, object]:
        """Update the optional OpenAI-compatible provider without restarting."""
        if base_url is not None and base_url.strip():
            self.base_url = base_url.strip().rstrip("/")
        if api_key is not None:
            self.api_key = api_key.strip()
        if model_name is not None and model_name.strip():
            self.model_name = model_name.strip()
        if enabled is not None:
            self.remote_enabled = enabled
        return self.get_provider_status()

    def get_provider_status(self) -> Dict[str, object]:
        """Return safe provider metadata without exposing the configured secret."""
        try:
            parts = urlsplit(self.base_url)
            safe_url = urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
        except ValueError:
            safe_url = ""
        return {
            "base_url": safe_url,
            "model_name": self.model_name,
            "api_key_configured": bool(self.api_key and self.api_key != "EMPTY"),
            "enabled": self.remote_enabled,
            "last_provider": self.last_provider,
            "last_error": self.last_error,
            "remote_requests": self.remote_requests,
            "remote_failures": self.remote_failures,
        }















    def _get_http_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._http_client is None or self._http_client_loop is not loop:
            self._http_client = httpx.AsyncClient(
                transport=self._transport,
                timeout=httpx.Timeout(
                    self.timeout_seconds,
                    connect=min(5.0, self.timeout_seconds),
                ),
            )
            self._http_client_loop = loop
        return self._http_client

    async def aclose(self) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
            self._http_client_loop = None

    def _request_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _chat_endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _request_payload(
        self, question: str, chunks: List[RetrievedDocument], *, stream: bool
    ) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": REMOTE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self.local.build_evidence(question, chunks),
                },
            ],
            "temperature": 0.0,
            "max_tokens": settings.LLM_MAX_TOKENS,
            "stream": stream,
        }

    def _mark_remote_success(self) -> None:
        self.last_provider = "remote_openai_compatible"
        self.last_error = None
        self.remote_requests += 1
        self._remote_breaker.record_success()

    def _mark_remote_failure(self, exc: Exception) -> None:
        self.last_provider = "local_fallback"
        self.last_error = type(exc).__name__
        self.remote_failures += 1
        self._remote_breaker.record_failure()
        logger.warning(
            "Remote LLM request failed (%s); using deterministic fallback",
            type(exc).__name__,
        )

    async def generate_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> str:
        """Use the configured remote model when enabled, else deterministic synthesis."""
        if DOCUMENTS_QUESTION_RE.search(question):
            document_names = sorted(set(indexed_docs or [chunk.doc_name for chunk in chunks]))
            if document_names:
                return (
                    "Indexed PDFs: "
                    + ", ".join(f"{name}.pdf" for name in document_names)
                    + ". Select a document in the sidebar or ask a question across all PDFs."
                )
            return "No PDFs are indexed yet. Upload a PDF or restore the demo shelf from the left sidebar."

        if META_QUESTION_RE.search(question):
            return META_RESPONSE
        if not chunks:
            return self.local.out_of_scope(question)

        if not self.remote_enabled:
            self.last_provider = "local_deterministic"
        if self.remote_enabled and self._remote_breaker.allow():
            try:
                client = self._get_http_client()
                response = await client.post(
                    self._chat_endpoint(),
                    headers=self._request_headers(),
                    json=self._request_payload(question, chunks, stream=False),
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Remote provider returned an empty answer")
                cited_pages = parse_citations(content)
                available_pages = {
                    (chunk.doc_name, chunk.page_number) for chunk in chunks
                }
                if not cited_pages or not cited_pages <= available_pages:
                    raise ValueError(
                        "Remote provider cited a page outside the retrieved evidence"
                    )
                self._mark_remote_success()
                return content.strip()
            except Exception as exc:
                self._mark_remote_failure(exc)

        if self.last_provider != "local_fallback":
            self.last_provider = "local_deterministic"
        return self.local.answer(question, chunks, indexed_docs=indexed_docs)

    async def stream_answer(
        self,
        question: str,
        chunks: List[RetrievedDocument],
        indexed_docs: List[str] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield answer text from the remote SSE stream, falling back locally."""
        if not self.remote_enabled:
            self.last_provider = "local_deterministic"
        if self.remote_enabled and chunks and self._remote_breaker.allow():
            emitted = False
            try:
                client = self._get_http_client()
                async with client.stream(
                    "POST",
                    self._chat_endpoint(),
                    headers=self._request_headers(),
                    json=self._request_payload(question, chunks, stream=True),
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        event = json.loads(payload)
                        delta = event["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if isinstance(token, str) and token:
                            emitted = True
                            yield token
                if not emitted:
                    raise ValueError("Remote provider streamed no content")
                self._mark_remote_success()
                return
            except Exception as exc:
                self._mark_remote_failure(exc)
                if emitted:
                    return

        if self.last_provider != "local_fallback":
            self.last_provider = "local_deterministic"
        answer = self.local.answer(question, chunks, indexed_docs=indexed_docs)
        words = answer.split(" ")
        for index, word in enumerate(words):
            suffix = " " if index < len(words) - 1 else ""
            yield f"{word}{suffix}"

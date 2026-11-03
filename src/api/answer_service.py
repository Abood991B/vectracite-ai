"""HTTP adapter for the answer pipeline.

The policy itself lives in `src.usecases.answer_pipeline`; this module only maps a domain refusal to
an HTTP status, which is the one thing the use case must not know.
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import HTTPException

from src.inference.schemas import QueryRequest, QueryResponse
from src.usecases.answer_pipeline import AnswerPipeline, answer_contributors
from src.usecases.ports import QueryBlockedError

__all__ = ["AnswerService", "answer_contributors"]


def _blocked_response(exc: QueryBlockedError) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": "Query blocked by Enterprise AI Guardrails.",
            "triggered_rules": exc.rules,
        },
    )


class AnswerService(AnswerPipeline):
    """The guarded pipeline, plus the mapping from refusal to HTTP 400."""

    async def execute(self, payload: QueryRequest) -> QueryResponse:
        try:
            return await super().execute(payload)
        except QueryBlockedError as exc:
            raise _blocked_response(exc) from exc

    async def stream(
        self, payload: QueryRequest
    ) -> AsyncGenerator[str | QueryResponse, None]:
        try:
            async for event in super().stream(payload):
                yield event
        except QueryBlockedError as exc:
            raise _blocked_response(exc) from exc

    def validate(self, payload: QueryRequest) -> str:
        try:
            return super().validate(payload)
        except QueryBlockedError as exc:
            raise _blocked_response(exc) from exc

"""Transport-level request safeguards that run before route parsing."""

import json
from typing import Any, Awaitable, Callable

from src.config import settings

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]


class RequestBodyTooLarge(Exception):
    """Raised when a streamed request body exceeds the configured ceiling."""


class BodySizeLimitMiddleware:
    """Reject oversized JSON/base64 bodies before framework parsing allocates them."""

    def __init__(self, app: Any, max_bytes: int | None = None) -> None:
        self.app = app
        self.max_bytes = max_bytes

    def _limit(self) -> int:
        return self.max_bytes or settings.effective_max_request_bytes()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        limit = self._limit()
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                await self._reject(send)
                return
            if declared > limit:
                await self._reject(send)
                return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise RequestBodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await self._reject(send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = json.dumps({"detail": "Request body limit exceeded."}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})

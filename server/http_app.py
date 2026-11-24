"""Remote (streamable-http) serving: bearer-token auth + a /health endpoint."""
from __future__ import annotations

import hmac
import json
import logging
from ipaddress import ip_address

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.types import ASGIApp, Receive, Scope, Send

log = logging.getLogger("hireloop.http")


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


class AuthError(RuntimeError):
    """Refusing to start in an unsafe configuration."""


def check_startup(host: str, token: str) -> None:
    """A network-reachable server must never run without a token: it exposes career data."""
    if not token and not is_loopback(host):
        raise AuthError(
            f"Refusing to serve on {host} without authentication. Set HIRELOOP_AUTH_TOKEN "
            "(clients send 'Authorization: Bearer <token>') or bind to 127.0.0.1."
        )


class BearerAuthMiddleware:
    """Pure-ASGI middleware. /health is open (load-balancer probes); everything else needs the token."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app, self.token = app, token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan / websocket pass straight through
            await self.app(scope, receive, send)
            return
        if scope["path"] == "/health":
            await self._reply(send, 200, {"status": "ok"})
            return
        headers = dict(scope.get("headers", []))
        supplied = headers.get(b"authorization", b"")
        scheme, _, value = supplied.partition(b" ")
        if scheme.lower() == b"bearer" and hmac.compare_digest(value.strip(), self.token):
            await self.app(scope, receive, send)
            return
        await self._reply(send, 401, {"error": "unauthorized"}, {b"www-authenticate": b"Bearer"})

    @staticmethod
    async def _reply(send: Send, status: int, body: dict, extra: dict[bytes, bytes] | None = None) -> None:
        payload = json.dumps(body).encode()
        headers = [(b"content-type", b"application/json"), (b"content-length", str(len(payload)).encode())]
        headers += list((extra or {}).items())
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": payload})


def build_http_app(mcp: FastMCP, token: str) -> ASGIApp:
    """The MCP streamable-http ASGI app (at /mcp), wrapped with auth when a token is set."""
    app = mcp.streamable_http_app()
    if token:
        return BearerAuthMiddleware(app, token)
    log.warning("HIRELOOP_AUTH_TOKEN is not set: serving WITHOUT authentication (loopback only).")
    return app


def serve(mcp: FastMCP, host: str, port: int, token: str) -> None:
    check_startup(host, token)
    uvicorn.run(build_http_app(mcp, token), host=host, port=port, log_level="info")

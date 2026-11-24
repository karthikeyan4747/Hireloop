"""Control tiers and confirmation checks.

Every tool is registered with exactly one tier:
  read  - no side effects (search, rank, gap analysis)
  draft - produces text for the user to review
  act   - writes to the user's stored data; requires confirm=True
"""
from __future__ import annotations

import functools
from enum import Enum
from typing import Any, Awaitable, Callable

from mcp.server.fastmcp.exceptions import ToolError


class Tier(str, Enum):
    READ = "read"
    DRAFT = "draft"
    ACT = "act"


class ToolFailure(Exception):
    """A user-facing, expected failure (bad input, provider down, missing profile...)."""


class ConfirmationRequired(ToolFailure):
    """Raised when an `act` tool is called without confirm=True."""


# tool name -> tier. Filled by @tiered; read by the README/tests.
TOOL_TIERS: dict[str, Tier] = {}


def require_confirm(confirm: bool, what: str) -> None:
    if confirm is not True:
        raise ConfirmationRequired(
            f"This action will {what}. Nothing was changed. "
            "Show the user the details and call again with confirm=true once they agree."
        )


def tiered(tier: Tier) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Register a tool's tier and prefix its docstring with it (shown to the MCP client)."""

    def deco(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        TOOL_TIERS[fn.__name__] = tier
        fn.__doc__ = f"[control tier: {tier.value}]\n{(fn.__doc__ or '').strip()}"

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ToolFailure as exc:
                # Surface a clear message to the client instead of a stack trace.
                raise ToolError(str(exc)) from None

        return wrapper

    return deco

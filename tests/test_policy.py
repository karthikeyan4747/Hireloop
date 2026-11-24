import pytest
from mcp.server.fastmcp.exceptions import ToolError

from server.policy import TOOL_TIERS, ConfirmationRequired, Tier, ToolFailure, require_confirm, tiered


def test_require_confirm_refuses_unless_true():
    with pytest.raises(ConfirmationRequired):
        require_confirm(False, "do a thing")
    require_confirm(True, "do a thing")  # no raise


def test_confirm_must_be_literally_true():
    with pytest.raises(ConfirmationRequired):
        require_confirm("yes", "do a thing")  # type: ignore[arg-type]


async def test_tiered_registers_tier_and_docstring():
    @tiered(Tier.DRAFT)
    async def my_tool() -> str:
        """Does stuff."""
        return "ok"

    assert TOOL_TIERS["my_tool"] is Tier.DRAFT
    assert my_tool.__doc__.startswith("[control tier: draft]")
    assert await my_tool() == "ok"


async def test_tool_failure_becomes_clean_tool_error():
    @tiered(Tier.READ)
    async def failing() -> None:
        raise ToolFailure("provider is down")

    with pytest.raises(ToolError, match="provider is down"):
        await failing()


def test_phase1_tools_have_expected_tiers():
    import server.main  # noqa: F401  (registers tiers on import)

    assert TOOL_TIERS["build_profile"] is Tier.ACT
    for name in ("search_jobs", "rank_fit", "skill_gap"):
        assert TOOL_TIERS[name] is Tier.READ

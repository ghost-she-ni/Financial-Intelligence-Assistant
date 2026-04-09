from __future__ import annotations

import os

import pytest

from src.llm.client import LLMClient

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_LLM_TESTS") != "1",
    reason="Set RUN_LIVE_LLM_TESTS=1 to enable live OpenAI smoke tests.",
)
def test_llm_client_generate_smoke() -> None:
    client = LLMClient(model=os.getenv("TEST_OPENAI_MODEL", "gpt-4o-mini"))

    result = client.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Say hello in one short sentence.",
        task_name="smoke_test",
    )

    assert isinstance(result["response_text"], str)
    assert result["response_text"].strip() != ""
    assert result["from_cache"] in {True, False}

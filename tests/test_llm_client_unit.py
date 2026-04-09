from __future__ import annotations

from types import SimpleNamespace

from src.llm.client import LLMClient


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]

    def model_dump(self) -> dict[str, str]:
        return {"content": self.choices[0].message.content}


class CountingCreate:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs_history: list[dict] = []

    def __call__(self, **kwargs):
        self.calls += 1
        self.kwargs_history.append(dict(kwargs))
        if self.calls == 1:
            raise ConnectionError("temporary network failure")
        return FakeResponse("hello after retry")


class LegacyTokenOnlyCreate:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs_history: list[dict] = []

    def __call__(self, **kwargs):
        self.calls += 1
        self.kwargs_history.append(dict(kwargs))
        if "max_completion_tokens" in kwargs:
            raise TypeError("got an unexpected keyword argument 'max_completion_tokens'")
        return FakeResponse("legacy provider response")


class InvalidThenValidJSONCreate:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs_history: list[dict] = []

    def __call__(self, **kwargs):
        self.calls += 1
        self.kwargs_history.append(dict(kwargs))
        if self.calls == 1:
            return FakeResponse('{"entities": [')
        return FakeResponse('{"entities": []}')


class FakeToolCallResponse:
    def __init__(self) -> None:
        tool_call = SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(
                name="search_financial_corpus",
                arguments='{"question":"What changed?","top_k":3}',
            ),
        )
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[tool_call]))]

    def model_dump(self) -> dict[str, str]:
        return {"tool_calls": 1}


class ToolCallingCreate:
    def __init__(self) -> None:
        self.calls = 0
        self.kwargs_history: list[dict] = []

    def __call__(self, **kwargs):
        self.calls += 1
        self.kwargs_history.append(dict(kwargs))
        return FakeToolCallResponse()


def make_fake_openai(create_callable):
    return lambda api_key=None, base_url=None: SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create_callable),
        )
    )


def test_llm_client_respects_instance_max_attempts(monkeypatch) -> None:
    create_callable = CountingCreate()
    monkeypatch.setattr(
        "src.llm.client.OpenAI",
        make_fake_openai(create_callable),
    )

    client = LLMClient(
        model="test-model",
        api_key="test-key",
        max_attempts=2,
        use_cache=False,
    )

    result = client.generate(
        system_prompt="You are helpful.",
        user_prompt="Say hello.",
        task_name="unit_retry_test",
    )

    assert result["response_text"] == "hello after retry"
    assert create_callable.calls == 2


def test_llm_client_falls_back_to_max_tokens_and_remembers_provider_preference(monkeypatch) -> None:
    create_callable = LegacyTokenOnlyCreate()
    monkeypatch.setattr(
        "src.llm.client.OpenAI",
        make_fake_openai(create_callable),
    )

    client = LLMClient(
        model="test-model",
        api_key="test-key",
        max_attempts=1,
        use_cache=False,
    )

    first_result = client.generate(
        system_prompt="You are helpful.",
        user_prompt="First request.",
        task_name="legacy_token_fallback_first",
    )
    second_result = client.generate(
        system_prompt="You are helpful.",
        user_prompt="Second request.",
        task_name="legacy_token_fallback_second",
    )

    assert first_result["response_text"] == "legacy provider response"
    assert second_result["response_text"] == "legacy provider response"
    assert create_callable.calls == 3

    first_call_kwargs = create_callable.kwargs_history[0]
    fallback_call_kwargs = create_callable.kwargs_history[1]
    remembered_call_kwargs = create_callable.kwargs_history[2]

    assert "max_completion_tokens" in first_call_kwargs
    assert "max_tokens" not in first_call_kwargs

    assert "max_tokens" in fallback_call_kwargs
    assert "max_completion_tokens" not in fallback_call_kwargs

    assert "max_tokens" in remembered_call_kwargs
    assert "max_completion_tokens" not in remembered_call_kwargs


def test_generate_json_retries_with_higher_token_budget_after_invalid_json(monkeypatch) -> None:
    create_callable = InvalidThenValidJSONCreate()
    monkeypatch.setattr(
        "src.llm.client.OpenAI",
        make_fake_openai(create_callable),
    )

    client = LLMClient(
        model="test-model",
        api_key="test-key",
        max_attempts=1,
        use_cache=False,
        max_output_tokens=700,
    )

    result = client.generate_json(
        system_prompt="Return JSON only.",
        user_prompt="Extract entities.",
        task_name="invalid_json_retry_test",
    )

    assert result["parsed_json"] == {"entities": []}
    assert create_callable.calls == 2
    first_call_kwargs = create_callable.kwargs_history[0]
    retry_call_kwargs = create_callable.kwargs_history[1]
    assert first_call_kwargs["max_completion_tokens"] == 700
    assert retry_call_kwargs["max_completion_tokens"] == 1400


def test_chat_supports_function_calling_payloads(monkeypatch) -> None:
    create_callable = ToolCallingCreate()
    monkeypatch.setattr(
        "src.llm.client.OpenAI",
        make_fake_openai(create_callable),
    )

    client = LLMClient(
        model="test-model",
        api_key="test-key",
        max_attempts=1,
        use_cache=False,
    )

    result = client.chat(
        messages=[
            {"role": "system", "content": "Use tools when needed."},
            {"role": "user", "content": "Find what changed."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "search_financial_corpus",
                    "description": "Search the corpus.",
                    "parameters": {"type": "object", "properties": {"question": {"type": "string"}}},
                },
            }
        ],
        task_name="tool_call_test",
    )

    assert result["response_text"] == ""
    assert result["tool_calls"][0]["function"]["name"] == "search_financial_corpus"
    first_call_kwargs = create_callable.kwargs_history[0]
    assert "tools" in first_call_kwargs

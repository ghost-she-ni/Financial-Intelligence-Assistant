from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI

from src.llm.retry import extract_status_code, retry_llm_call

load_dotenv()

JSON_RETRY_TOKEN_MULTIPLIER = 2
MAX_JSON_RETRY_OUTPUT_TOKENS = 3200
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def stable_json_dumps(payload: dict[str, Any]) -> str:
    """Stable JSON serialization for hashing and persistence."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_request_hash(payload: dict[str, Any]) -> str:
    """Compute a stable hash for a request payload."""
    serialized = stable_json_dumps(payload)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_optional_setting(value: str | None) -> str | None:
    """Treat blank settings as unset so optional env vars stay truly optional."""
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None
    return normalized


def get_logger(name: str = "llm.client") -> logging.Logger:
    """Return a configured logger for LLM client events."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


@dataclass
class LLMClient:
    """
    Robust OpenAI-compatible LLM client with:
    - retry / exponential backoff
    - logging
    - local response cache for resumability
    """

    model: str
    temperature: float = 0.0
    max_output_tokens: int = 800
    api_key: str | None = None
    base_url: str | None = None
    cache_path: str | Path = "data/cache/llm_responses.jsonl"
    max_attempts: int = 6
    use_cache: bool = True
    logger: logging.Logger = field(default_factory=get_logger)

    def __post_init__(self) -> None:
        self.cache_path = Path(self.cache_path)
        resolved_api_key = normalize_optional_setting(self.api_key)
        resolved_base_url = normalize_optional_setting(self.base_url)

        if resolved_api_key is None:
            resolved_api_key = normalize_optional_setting(os.getenv("OPENAI_API_KEY"))
        if resolved_base_url is None:
            resolved_base_url = normalize_optional_setting(os.getenv("OPENAI_BASE_URL"))
        if resolved_base_url is None:
            resolved_base_url = DEFAULT_OPENAI_BASE_URL

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self._cache_index: dict[str, dict[str, Any]] | None = None  # lazy loaded
        self._token_limit_parameter = "max_completion_tokens"

    def __repr__(self) -> str:
        return (
            f"LLMClient(model={self.model}, cache_path={self.cache_path}, "
            f"use_cache={self.use_cache})"
        )

    def _load_cache_index(self) -> dict[str, dict[str, Any]]:
        """Load the entire cache into an in‑memory index (request_hash -> record)."""
        if self._cache_index is not None:
            return self._cache_index

        index: dict[str, dict[str, Any]] = {}
        if not self.cache_path.exists():
            self._cache_index = index
            return index

        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if record.get("status") == "success":
                        index[record["request_hash"]] = record
        except (json.JSONDecodeError, OSError) as e:
            self.logger.error(f"Failed to load cache index: {e}")
            # Continue with empty index

        self._cache_index = index
        return index

    def _find_cached_response(self, request_hash: str) -> dict[str, Any] | None:
        """Return a cached response if it exists, using the in‑memory index."""
        if not self.use_cache:
            return None
        index = self._load_cache_index()
        return index.get(request_hash)

    def _append_cache_record(self, record: dict[str, Any]) -> None:
        """Append one JSONL record to the cache and update the in‑memory index."""
        ensure_parent_dir(self.cache_path)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Update index if it exists
        if self._cache_index is not None and record.get("status") == "success":
            self._cache_index[record["request_hash"]] = record

    def _build_request_payload(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a normalized payload used both for the API call and for hashing."""
        payload = {
            "model": self.model,
            "temperature": self.temperature if temperature is None else temperature,
            "max_output_tokens": self.max_output_tokens if max_output_tokens is None else max_output_tokens,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response_format": response_format,
            "metadata": metadata or {},
        }
        return payload

    def _build_messages_request_payload(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a normalized payload for multi-message chat and tool calls."""
        payload = {
            "model": self.model,
            "temperature": self.temperature if temperature is None else temperature,
            "max_output_tokens": self.max_output_tokens if max_output_tokens is None else max_output_tokens,
            "messages": messages,
            "response_format": response_format,
            "tools": tools or [],
            "tool_choice": tool_choice,
            "metadata": metadata or {},
        }
        return payload

    def _build_chat_completion_request_kwargs(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None = None,
    ):
        """Perform one chat completion request."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # Prefer max_completion_tokens, but keep a fallback to max_tokens for
        # OpenAI-compatible providers that still reject the newer parameter.
        request_kwargs[self._token_limit_parameter] = max_output_tokens

        if response_format is not None:
            request_kwargs["response_format"] = response_format

        return request_kwargs

    def _build_chat_completion_request_kwargs_for_messages(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build request kwargs for multi-turn chat with optional tool calling."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        request_kwargs[self._token_limit_parameter] = max_output_tokens

        if response_format is not None:
            request_kwargs["response_format"] = response_format
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice

        return request_kwargs

    def _execute_chat_completion_request(self, request_kwargs: dict[str, Any]):
        """Execute one chat completion request with the instance retry policy."""
        max_attempts = max(1, int(self.max_attempts))
        request_with_retry = retry_llm_call(max_attempts=max_attempts)(
            self.client.chat.completions.create
        )
        return request_with_retry(**request_kwargs)

    def _should_retry_with_legacy_max_tokens(self, exc: Exception) -> bool:
        """Return True when the provider rejects max_completion_tokens itself."""
        if self._token_limit_parameter != "max_completion_tokens":
            return False

        if isinstance(exc, TypeError):
            error_text = str(exc).lower()
            return "max_completion_tokens" in error_text and (
                "unexpected keyword" in error_text or "unexpected argument" in error_text
            )

        status_code = extract_status_code(exc)
        if status_code != 400 and not isinstance(exc, BadRequestError):
            return False

        error_text_parts = [str(exc)]
        message = getattr(exc, "message", None)
        if isinstance(message, str) and message.strip():
            error_text_parts.append(message)

        body = getattr(exc, "body", None)
        if body is not None:
            if isinstance(body, dict):
                error_text_parts.append(stable_json_dumps(body))
            else:
                error_text_parts.append(str(body))

        error_text = " ".join(error_text_parts).lower()
        return "max_completion_tokens" in error_text and any(
            marker in error_text
            for marker in [
                "unsupported",
                "unknown",
                "unexpected",
                "not recognized",
                "unrecognized",
                "invalid parameter",
                "extra inputs are not permitted",
            ]
        )

    def _chat_completion_request(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None = None,
    ):
        """Perform one chat completion request with provider-compatible token fallback."""
        request_kwargs = self._build_chat_completion_request_kwargs(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
        )

        try:
            return self._execute_chat_completion_request(request_kwargs)
        except Exception as exc:
            if not self._should_retry_with_legacy_max_tokens(exc):
                raise

            self.logger.info(
                "Provider rejected max_completion_tokens; retrying with max_tokens."
            )
            self._token_limit_parameter = "max_tokens"
            fallback_kwargs = self._build_chat_completion_request_kwargs(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
            )
            return self._execute_chat_completion_request(fallback_kwargs)

    def _chat_completion_request_for_messages(
        self,
        messages: list[dict[str, Any]],
        temperature: float,
        max_output_tokens: int,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ):
        """Perform one multi-turn chat completion request with provider token fallback."""
        request_kwargs = self._build_chat_completion_request_kwargs_for_messages(
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )

        try:
            return self._execute_chat_completion_request(request_kwargs)
        except Exception as exc:
            if not self._should_retry_with_legacy_max_tokens(exc):
                raise

            self.logger.info(
                "Provider rejected max_completion_tokens; retrying with max_tokens."
            )
            self._token_limit_parameter = "max_tokens"
            fallback_kwargs = self._build_chat_completion_request_kwargs_for_messages(
                messages=messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
            )
            return self._execute_chat_completion_request(fallback_kwargs)

    def _extract_response_text(self, content: Any) -> str:
        """Normalize assistant content to plain text for caching and downstream parsing."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    if isinstance(item.get("text"), str):
                        text_parts.append(item["text"])
                    elif item.get("type") == "output_text" and isinstance(item.get("content"), str):
                        text_parts.append(item["content"])
                else:
                    text = getattr(item, "text", None)
                    if isinstance(text, str):
                        text_parts.append(text)
            return "".join(text_parts)
        return str(content)

    def _normalize_tool_calls(self, tool_calls: Any) -> list[dict[str, Any]]:
        """Normalize provider tool call objects into plain JSON-serializable dictionaries."""
        if tool_calls is None:
            return []

        normalized_calls: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls, start=1):
            function_payload = getattr(tool_call, "function", None)
            function_name = None
            function_arguments: Any = "{}"

            if function_payload is not None:
                function_name = getattr(function_payload, "name", None)
                function_arguments = getattr(function_payload, "arguments", "{}")
            elif isinstance(tool_call, dict):
                function_payload = tool_call.get("function", {})
                function_name = function_payload.get("name")
                function_arguments = function_payload.get("arguments", "{}")

            if not isinstance(function_arguments, str):
                function_arguments = json.dumps(function_arguments, ensure_ascii=False)

            normalized_calls.append(
                {
                    "id": str(getattr(tool_call, "id", None) or f"tool_call_{index}"),
                    "type": str(getattr(tool_call, "type", None) or "function"),
                    "function": {
                        "name": str(function_name or ""),
                        "arguments": function_arguments,
                    },
                }
            )

        return normalized_calls

    def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        task_name: str = "generic_chat",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a multi-turn chat completion, optionally with tool calling support."""
        payload = self._build_messages_request_payload(
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            metadata=metadata,
        )
        request_hash = compute_request_hash(payload)

        cached_record = self._find_cached_response(request_hash)
        if cached_record is not None:
            self.logger.info(
                "Cache hit for task=%s request_hash=%s",
                task_name,
                request_hash,
            )
            cached_response_text = cached_record.get("response_text", "") or ""
            cached_tool_calls = cached_record.get("tool_calls", []) or []
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": cached_response_text or None,
            }
            if cached_tool_calls:
                assistant_message["tool_calls"] = cached_tool_calls

            return {
                "task_name": cached_record["task_name"],
                "request_hash": cached_record["request_hash"],
                "model": cached_record["model"],
                "response_text": cached_response_text,
                "tool_calls": cached_tool_calls,
                "assistant_message": assistant_message,
                "raw_response": cached_record.get("raw_response"),
                "created_at": cached_record["created_at"],
                "from_cache": True,
                "metadata": cached_record.get("metadata", {}),
            }

        resolved_temperature = self.temperature if temperature is None else temperature
        resolved_max_output_tokens = (
            self.max_output_tokens if max_output_tokens is None else max_output_tokens
        )

        self.logger.info(
            "LLM request started | task=%s | model=%s | request_hash=%s",
            task_name,
            self.model,
            request_hash,
        )

        try:
            response = self._chat_completion_request_for_messages(
                messages=messages,
                temperature=resolved_temperature,
                max_output_tokens=resolved_max_output_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
            )

            message = response.choices[0].message
            response_text = self._extract_response_text(getattr(message, "content", None))
            tool_calls_payload = self._normalize_tool_calls(getattr(message, "tool_calls", None))

            assistant_message = {
                "role": "assistant",
                "content": response_text or None,
            }
            if tool_calls_payload:
                assistant_message["tool_calls"] = tool_calls_payload

            record = {
                "task_name": task_name,
                "request_hash": request_hash,
                "model": self.model,
                "temperature": resolved_temperature,
                "max_output_tokens": resolved_max_output_tokens,
                "messages": messages,
                "response_text": response_text,
                "tool_calls": tool_calls_payload,
                "raw_response": response.model_dump(),
                "created_at": now_utc_iso(),
                "status": "success",
                "metadata": metadata or {},
            }

            if self.use_cache:
                self._append_cache_record(record)

            self.logger.info(
                "LLM request completed | task=%s | model=%s | request_hash=%s",
                task_name,
                self.model,
                request_hash,
            )

            return {
                "task_name": task_name,
                "request_hash": request_hash,
                "model": self.model,
                "response_text": response_text,
                "tool_calls": tool_calls_payload,
                "assistant_message": assistant_message,
                "raw_response": response.model_dump(),
                "created_at": record["created_at"],
                "from_cache": False,
                "metadata": metadata or {},
            }

        except Exception as exc:
            failure_record = {
                "task_name": task_name,
                "request_hash": request_hash,
                "model": self.model,
                "temperature": resolved_temperature,
                "max_output_tokens": resolved_max_output_tokens,
                "messages": messages,
                "response_text": None,
                "tool_calls": [],
                "raw_response": None,
                "created_at": now_utc_iso(),
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "metadata": metadata or {},
            }

            if self.use_cache:
                self._append_cache_record(failure_record)

            self.logger.exception(
                "LLM request failed | task=%s | model=%s | request_hash=%s",
                task_name,
                self.model,
                request_hash,
            )
            raise

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        task_name: str = "generic_generation",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Main public method to call the LLM with retry, logging, and caching.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result = self.chat(
            messages=messages,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            task_name=task_name,
            metadata=metadata,
        )
        return {
            "task_name": result["task_name"],
            "request_hash": result["request_hash"],
            "model": result["model"],
            "response_text": result["response_text"],
            "raw_response": result.get("raw_response"),
            "created_at": result["created_at"],
            "from_cache": result["from_cache"],
            "metadata": result.get("metadata", {}),
        }

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        task_name: str,
        metadata: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Convenience method for tasks where the model is asked to return JSON.
        """
        current_max_output_tokens = (
            self.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        last_error: json.JSONDecodeError | None = None

        while True:
            result = self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_output_tokens=current_max_output_tokens,
                response_format={"type": "json_object"},
                task_name=task_name,
                metadata=metadata,
            )

            try:
                parsed_json = json.loads(result["response_text"])
                result["parsed_json"] = parsed_json
                if last_error is not None:
                    result["json_retry_max_output_tokens"] = current_max_output_tokens
                return result
            except json.JSONDecodeError as e:
                self.logger.error(
                    "Failed to parse JSON response for task=%s: %s\nResponse text: %s",
                    task_name,
                    e,
                    result["response_text"],
                )
                last_error = e
                retry_max_output_tokens = min(
                    max(
                        current_max_output_tokens + 1,
                        current_max_output_tokens * JSON_RETRY_TOKEN_MULTIPLIER,
                    ),
                    MAX_JSON_RETRY_OUTPUT_TOKENS,
                )
                if retry_max_output_tokens <= current_max_output_tokens:
                    raise ValueError(
                        f"LLM returned invalid JSON for task '{task_name}' after retry: {e}"
                    ) from e

                self.logger.warning(
                    "Retrying invalid JSON for task=%s with higher max_output_tokens=%s",
                    task_name,
                    retry_max_output_tokens,
                )
                current_max_output_tokens = retry_max_output_tokens

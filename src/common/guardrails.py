from __future__ import annotations

import re
from typing import Any

import pandas as pd

PROMPT_INJECTION_PATTERNS = [
    r"\bignore (all|any|the|previous|prior)\b",
    r"\boverride\b",
    r"\bdisregard\b",
    r"\bforget (the|your)\b",
    r"\bdeveloper message\b",
    r"\bsystem prompt\b",
    r"\bhidden instructions\b",
    r"\buse outside knowledge\b",
    r"\bbrowse the web\b",
    r"\bfabricate citations?\b",
]

BIAS_SENSITIVE_PATTERNS = [
    r"\bscam\b",
    r"\bfraud\b",
    r"\bcorrupt\b",
    r"\billegal\b",
    r"\bdishonest\b",
    r"\bworst company\b",
    r"\bterrible company\b",
    r"\bevil\b",
    r"\bcriminal\b",
]

REFUSAL_PATTERNS = [
    r"\bnot available in the provided context\b",
    r"\binsufficient context\b",
    r"\bprovided context does not contain\b",
    r"\bi cannot verify\b",
    r"\bi cannot provide a grounded answer\b",
]

LOADED_LANGUAGE_PATTERNS = [
    r"\bscam\b",
    r"\bfraud\b",
    r"\bcriminal\b",
    r"\bevil\b",
    r"\bdishonest\b",
    r"\bterrible\b",
    r"\bworthless\b",
]


def clean_text(value: Any) -> str:
    """Normalize free-form text values for lightweight heuristic checks."""
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _contains_pattern(text: str, patterns: list[str]) -> bool:
    normalized_text = clean_text(text).lower()
    return any(re.search(pattern, normalized_text) for pattern in patterns)


def detect_prompt_injection_risk(text: str) -> bool:
    """Return True when the user input contains obvious prompt-injection signals."""
    return _contains_pattern(text, PROMPT_INJECTION_PATTERNS)


def detect_bias_sensitive_request(text: str) -> bool:
    """Return True when the prompt appears framed in a loaded or biased way."""
    return _contains_pattern(text, BIAS_SENSITIVE_PATTERNS)


def detect_grounded_refusal(answer: str) -> bool:
    """Return True when the answer explicitly refuses due to missing grounded evidence."""
    return _contains_pattern(answer, REFUSAL_PATTERNS)


def contains_loaded_language(text: str) -> bool:
    """Return True when the answer itself uses obviously loaded language."""
    return _contains_pattern(text, LOADED_LANGUAGE_PATTERNS)


def build_safety_flags(
    question: str,
    answer: str,
    citations: list[dict[str, Any]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Build lightweight safety flags for UI display and downstream evaluation."""
    citations = citations or []
    tool_calls = tool_calls or []

    flags: list[str] = []

    if detect_prompt_injection_risk(question):
        flags.append("prompt_injection_risk")

    if detect_bias_sensitive_request(question):
        flags.append("bias_sensitive_request")

    if not citations:
        flags.append("low_support")

    if detect_grounded_refusal(answer):
        flags.append("grounded_refusal")

    if tool_calls:
        flags.append("tool_augmented")

    if contains_loaded_language(answer):
        flags.append("loaded_language_in_answer")

    return flags

from __future__ import annotations

import re
from typing import Any

import pandas as pd


def _compile_patterns(*patterns: str) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


SECTION_DEFINITIONS: list[dict[str, Any]] = [
    {
        "code": "item_1",
        "title": "Business",
        "group": "business",
        "line_patterns": _compile_patterns(
            r"^item\s*1[\.\-:\s]+business$",
            r"^business$",
        ),
        "body_patterns": _compile_patterns(r"\bbusiness\b"),
        "priority": 1.00,
    },
    {
        "code": "competition",
        "title": "Competition",
        "group": "competition",
        "line_patterns": _compile_patterns(r"^competition$"),
        "body_patterns": _compile_patterns(r"\bcompetition\b", r"\bcompetitive\b"),
        "priority": 1.05,
    },
    {
        "code": "item_1a",
        "title": "Risk Factors",
        "group": "risk_factors",
        "line_patterns": _compile_patterns(
            r"^item\s*1a[\.\-:\s]+risk factors$",
            r"^risk factors$",
        ),
        "body_patterns": _compile_patterns(r"\brisk factors\b"),
        "priority": 1.15,
    },
    {
        "code": "item_1c",
        "title": "Cybersecurity",
        "group": "cybersecurity",
        "line_patterns": _compile_patterns(
            r"^item\s*1c[\.\-:\s]+cybersecurity$",
            r"^cybersecurity$",
        ),
        "body_patterns": _compile_patterns(r"\bcybersecurity\b"),
        "priority": 1.10,
    },
    {
        "code": "item_2",
        "title": "Properties",
        "group": "properties",
        "line_patterns": _compile_patterns(
            r"^item\s*2[\.\-:\s]+properties$",
            r"^properties$",
        ),
        "body_patterns": _compile_patterns(r"\bproperties\b"),
        "priority": 1.00,
    },
    {
        "code": "item_3",
        "title": "Legal Proceedings",
        "group": "legal_proceedings",
        "line_patterns": _compile_patterns(
            r"^item\s*3[\.\-:\s]+legal proceedings$",
            r"^legal proceedings$",
        ),
        "body_patterns": _compile_patterns(r"\blegal proceedings\b"),
        "priority": 1.10,
    },
    {
        "code": "executive_officers",
        "title": "Information About Our Executive Officers",
        "group": "executives",
        "line_patterns": _compile_patterns(
            r"^item\s*4a?[\.\-:\s]+information about (our )?executive officers$",
            r"^information about (our )?executive officers$",
        ),
        "body_patterns": _compile_patterns(r"\bexecutive officers\b"),
        "priority": 1.00,
    },
    {
        "code": "item_5",
        "title": "Market For Registrant's Common Equity",
        "group": "market_and_equity",
        "line_patterns": _compile_patterns(
            r"^item\s*5[\.\-:\s]+market for registrant'?s common equity.*$",
            r"^market for registrant'?s common equity.*$",
        ),
        "body_patterns": _compile_patterns(r"\bmarket for registrant'?s common equity\b"),
        "priority": 1.05,
    },
    {
        "code": "item_7",
        "title": "Management's Discussion and Analysis",
        "group": "management_discussion",
        "line_patterns": _compile_patterns(
            r"^item\s*7[\.\-:\s]+management'?s discussion.*analysis.*$",
            r"^management'?s discussion.*analysis.*$",
            r"^management discussion and analysis.*$",
        ),
        "body_patterns": _compile_patterns(r"\bmanagement'?s discussion\b", r"\bresults of operations\b"),
        "priority": 1.15,
    },
    {
        "code": "item_7a",
        "title": "Quantitative And Qualitative Disclosures About Market Risk",
        "group": "market_risk",
        "line_patterns": _compile_patterns(
            r"^item\s*7a[\.\-:\s]+quantitative and qualitative disclosures about market risk$",
            r"^quantitative and qualitative disclosures about market risk$",
        ),
        "body_patterns": _compile_patterns(r"\bmarket risk\b"),
        "priority": 1.10,
    },
    {
        "code": "item_8",
        "title": "Financial Statements And Supplementary Data",
        "group": "financial_statements",
        "line_patterns": _compile_patterns(
            r"^item\s*8[\.\-:\s]+financial statements.*$",
            r"^financial statements.*supplementary data$",
            r"^consolidated statements of (income|operations|cash flows|financial position).*$",
        ),
        "body_patterns": _compile_patterns(
            r"\bfinancial statements\b",
            r"\bconsolidated statements of income\b",
            r"\bconsolidated balance sheets\b",
        ),
        "priority": 1.20,
    },
    {
        "code": "principal_products",
        "title": "Principal Products, Services And Solutions",
        "group": "products_and_offerings",
        "line_patterns": _compile_patterns(
            r"^principal products, services and solutions$",
            r"^products, services and solutions$",
        ),
        "body_patterns": _compile_patterns(r"\bprincipal products, services and solutions\b"),
        "priority": 1.05,
    },
    {
        "code": "operations",
        "title": "Operations",
        "group": "operations",
        "line_patterns": _compile_patterns(r"^operations$"),
        "body_patterns": _compile_patterns(r"\boperations\b"),
        "priority": 1.00,
    },
]


PREFERRED_SECTION_GROUPS = {
    "risk_factors": {
        "risk_factors": 1.00,
        "cybersecurity": 0.45,
        "management_discussion": 0.20,
    },
    "competition": {
        "competition": 1.00,
        "business": 0.35,
        "products_and_offerings": 0.20,
        "risk_factors": 0.15,
    },
    "business_segments": {
        "business": 0.95,
        "products_and_offerings": 0.80,
        "operations": 0.35,
    },
    "financial_metrics": {
        "financial_statements": 1.00,
        "management_discussion": 0.90,
        "market_risk": 0.30,
    },
    "ai": {
        "products_and_offerings": 0.80,
        "business": 0.55,
        "competition": 0.20,
    },
}


AVOID_SECTION_GROUPS = {
    "risk_factors": {
        "front_matter": 0.45,
        "business": 0.18,
        "financial_statements": 0.55,
        "market_and_equity": 0.20,
    },
    "competition": {
        "front_matter": 0.25,
        "financial_statements": 0.40,
    },
    "business_segments": {
        "front_matter": 0.25,
        "financial_statements": 0.30,
    },
    "ai": {
        "front_matter": 0.20,
        "financial_statements": 0.25,
    },
}


def normalize_section_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized.strip())
    return normalized


def normalize_section_token(text: str) -> str:
    normalized = normalize_section_text(text).lower()
    normalized = normalized.replace("&", "and")
    normalized = re.sub(r"[^\w\s\-'/]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def is_table_of_contents_page(text: str) -> bool:
    normalized = normalize_section_token(text)
    head = normalized[:1500]
    if "table of contents" in head:
        return True

    toc_signals = [
        "risk factors",
        "legal proceedings",
        "management's discussion",
        "management discussion",
        "quantitative and qualitative disclosures about market risk",
        "financial statements",
    ]
    signal_hits = sum(1 for signal in toc_signals if signal in head)
    return ("page no" in head or "form 10-k" in head) and signal_hits >= 3


def looks_like_header_line(line: str) -> bool:
    normalized = normalize_section_text(line)
    if not normalized:
        return False

    words = normalized.split()
    if len(words) > 18:
        return False

    if normalized.endswith(".") and not normalized.lower().startswith("item "):
        return False

    uppercase_ratio = sum(char.isupper() for char in normalized) / max(
        1,
        sum(char.isalpha() for char in normalized),
    )
    return (
        normalized.lower().startswith("item ")
        or uppercase_ratio >= 0.55
        or normalized.istitle()
    )


def match_section_header_line(line: str) -> dict[str, Any] | None:
    normalized_line = normalize_section_text(line)
    if not normalized_line or not looks_like_header_line(normalized_line):
        return None

    best_match: dict[str, Any] | None = None

    for section in SECTION_DEFINITIONS:
        for pattern in section["line_patterns"]:
            if pattern.match(normalized_line):
                score = section["priority"]
                if normalized_line.lower().startswith("item "):
                    score += 0.15
                if normalized_line.isupper():
                    score += 0.05

                candidate = {
                    "section_code": section["code"],
                    "section_title": section["title"],
                    "section_group": section["group"],
                    "match_score": score,
                    "matched_line": normalized_line,
                }

                if best_match is None or candidate["match_score"] > best_match["match_score"]:
                    best_match = candidate

    return best_match


def detect_page_section_transition(text: str, max_lines: int = 40) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None

    if is_table_of_contents_page(text):
        return None

    best_match: dict[str, Any] | None = None
    lines = [normalize_section_text(line) for line in text.split("\n")[:max_lines]]

    for line in lines:
        if not line:
            continue

        match = match_section_header_line(line)
        if match is None:
            continue

        if best_match is None or match["match_score"] > best_match["match_score"]:
            best_match = match

    return best_match


def annotate_document_sections(
    doc_df: pd.DataFrame,
    text_column: str = "clean_text",
) -> pd.DataFrame:
    if doc_df.empty:
        return doc_df.copy()

    annotated_df = doc_df.sort_values("page_num").reset_index(drop=True).copy()
    current_section = {
        "section_id": "front_matter__01",
        "section_code": "front_matter",
        "section_title": "Front Matter",
        "section_group": "front_matter",
    }
    section_counts: dict[str, int] = {}

    section_ids: list[str] = []
    section_codes: list[str] = []
    section_titles: list[str] = []
    section_groups: list[str] = []

    for row in annotated_df.itertuples(index=False):
        page_text = getattr(row, text_column, "")
        match = detect_page_section_transition(page_text)

        if match is not None and match["section_code"] != current_section["section_code"]:
            section_counts[match["section_code"]] = section_counts.get(match["section_code"], 0) + 1
            current_section = {
                "section_id": f"{match['section_code']}__{section_counts[match['section_code']]:02d}",
                "section_code": match["section_code"],
                "section_title": match["section_title"],
                "section_group": match["section_group"],
            }

        section_ids.append(current_section["section_id"])
        section_codes.append(current_section["section_code"])
        section_titles.append(current_section["section_title"])
        section_groups.append(current_section["section_group"])

    annotated_df["section_id"] = section_ids
    annotated_df["section_code"] = section_codes
    annotated_df["section_title"] = section_titles
    annotated_df["section_group"] = section_groups
    return annotated_df


def infer_chunk_section_metadata(chunk_text: str) -> dict[str, str | None]:
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        return {
            "section_code": None,
            "section_title": None,
            "section_group": None,
        }

    search_window = normalize_section_text(chunk_text[:3000])
    body_search_window = normalize_section_text(chunk_text[:1200])
    normalized_window = normalize_section_token(search_window)

    if is_table_of_contents_page(search_window):
        return {
            "section_code": "front_matter",
            "section_title": "Front Matter",
            "section_group": "front_matter",
        }

    item_mentions = len(re.findall(r"\bitem\s+\d+[a-z]?\b", normalized_window))
    if "form 10-k" in normalized_window and item_mentions >= 3:
        return {
            "section_code": "front_matter",
            "section_title": "Front Matter",
            "section_group": "front_matter",
        }

    lines = [normalize_section_text(line) for line in chunk_text.split("\n")[:20]]

    for line in lines:
        match = match_section_header_line(line)
        if match is not None:
            return {
                "section_code": match["section_code"],
                "section_title": match["section_title"],
                "section_group": match["section_group"],
            }

    for section in sorted(SECTION_DEFINITIONS, key=lambda item: item["priority"], reverse=True):
        if any(pattern.search(body_search_window) for pattern in section["body_patterns"]):
            return {
                "section_code": section["code"],
                "section_title": section["title"],
                "section_group": section["group"],
            }

    return {
        "section_code": None,
        "section_title": None,
        "section_group": None,
    }


def compute_section_intent_score(
    query_intents: list[str],
    section_group: str | None,
) -> float:
    if not query_intents or not section_group:
        return 0.0

    score = 0.0
    for intent in query_intents:
        score = max(score, PREFERRED_SECTION_GROUPS.get(intent, {}).get(section_group, 0.0))

    return float(score)


def compute_section_mismatch_penalty(
    query_intents: list[str],
    section_group: str | None,
) -> float:
    if not query_intents or not section_group:
        return 0.0

    penalty = 0.0
    for intent in query_intents:
        penalty = max(penalty, AVOID_SECTION_GROUPS.get(intent, {}).get(section_group, 0.0))

    return float(penalty)

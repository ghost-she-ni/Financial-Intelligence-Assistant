from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.run_pipeline import select_pipeline_steps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON_VERSION = "3.12.2"
EXPECTED_PIPELINE_STEPS = 19

REQUIRED_VERSIONED_INPUTS = {
    "data/evaluation/security/security_cases.csv": (
        "c9b2ebe31de1525faf0f654b0e3705f6ab3c5bd291fa70a2d735809248d96a7f"
    ),
    "data/queries/queries.parquet": (
        "15eb388aca1eff6f29d2e236b5ecd42a105dafbc46980dcf49307bb917015f3c"
    ),
    "data/raw_pdfs/adobe_2022_10k.pdf": (
        "d3e62062ed7d5d8b7397739546ef66ef536a861579f925cf2106d18b068877ff"
    ),
    "data/raw_pdfs/adobe_2023_10k.pdf": (
        "35ebf575f402ca271800e10e5626536006f6e660234c9ca97c3ba5c82f90e1fd"
    ),
    "data/raw_pdfs/adobe_2024_10k.pdf": (
        "cfb267ee47f3576a36c418833e867a6c15e3f93f0bf272c07921d062eeae24f8"
    ),
    "data/raw_pdfs/lockheedmartin_2022_10k.pdf": (
        "02d6146204601dff753fe32e36182e6ce14d3b4ebace8f441192a47b03570ae6"
    ),
    "data/raw_pdfs/lockheedmartin_2023_10k.pdf": (
        "ae017ed32d2f1e9d927757cd251b2ff50b4a271878bb335a5b2f4ac7b5227780"
    ),
    "data/raw_pdfs/lockheedmartin_2024_10k.pdf": (
        "0973c443a7c722607d1b68e96e8118a55aa1a23dbbe5c681742b0452cddeb6a2"
    ),
    "data/raw_pdfs/pfizer_2022_10k.pdf": (
        "178ce93924b30038fb0682c8b947a24e304dbb97ce914a77cfb7fb5146893282"
    ),
    "data/raw_pdfs/pfizer_2023_10k.pdf": (
        "847a37aca35982da9e0f8ea1c00668fbcf5a599ec3e014a4cd0028dfb21d08a7"
    ),
    "data/raw_pdfs/pfizer_2024_10k.pdf": (
        "56a1a635f056d6dd128212fc3912d3f9e2c132e951cb70939888a895e3b65612"
    ),
}

GENERATED_ARTIFACTS = {
    "data/metadata/documents_metadata.parquet": "metadata from raw PDFs",
    "data/extracted/extracted_pages.parquet": "page-level PDF extraction",
    "data/processed/processed_pages.parquet": "cleaned page text",
    "data/processed/chunks.parquet": "chunk corpus",
    "data/processed/entities.parquet": "entity extraction output",
    "data/processed/triplets.parquet": "triplet extraction output",
    "data/embeddings/chunk_embeddings.parquet": "chunk embedding cache",
    "data/embeddings/query_embeddings.parquet": "query embedding cache",
    "data/evaluation/financebench/raw/financebench_merged.jsonl": (
        "downloaded FinanceBench raw source"
    ),
    "data/evaluation/financebench/financebench_subset_local_smoke.parquet": (
        "local smoke evaluation subset"
    ),
    "data/indexes/retrieval/sentence_transformers_all_minilm_l6_v2/native_dense/manifest.json": (
        "persistent retrieval index manifest"
    ),
}

PERSISTENT_INDEX_MANIFEST = (
    "data/indexes/retrieval/sentence_transformers_all_minilm_l6_v2/native_dense/manifest.json"
)
FINANCEBENCH_RAW_PATH = "data/evaluation/financebench/raw/financebench_merged.jsonl"
FINANCEBENCH_RAW_SHA256 = "7a1c81789e0fd2f1c37057a7ec0097756d726b05e7228e68e57db8e18c54fd0b"


@dataclass(frozen=True)
class CheckResult:
    status: str
    message: str


def normalize_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def check_python_version(strict: bool) -> CheckResult:
    actual = ".".join(str(part) for part in sys.version_info[:3])
    if actual == EXPECTED_PYTHON_VERSION:
        return CheckResult("ok", f"Python version matches {EXPECTED_PYTHON_VERSION}.")

    message = f"Python version is {actual}; validated version is {EXPECTED_PYTHON_VERSION}."
    return CheckResult("fail" if strict else "warn", message)


def check_required_files() -> list[CheckResult]:
    results: list[CheckResult] = []

    for relative_path in [".python-version", "requirements/lock.txt", ".env.example"]:
        path = PROJECT_ROOT / relative_path
        if path.exists():
            results.append(CheckResult("ok", f"Required project file exists: {relative_path}."))
        else:
            results.append(CheckResult("fail", f"Missing required project file: {relative_path}."))

    for relative_path, expected_hash in REQUIRED_VERSIONED_INPUTS.items():
        path = PROJECT_ROOT / relative_path
        if not path.exists():
            results.append(CheckResult("fail", f"Missing required input: {relative_path}."))
            continue

        actual_hash = compute_sha256(path)
        if actual_hash == expected_hash:
            results.append(CheckResult("ok", f"Input checksum matches: {relative_path}."))
        else:
            results.append(
                CheckResult(
                    "fail",
                    f"Checksum mismatch for {relative_path}: expected {expected_hash}, got {actual_hash}.",
                )
            )

    return results


def check_generated_artifacts(require_generated: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    missing_status = "fail" if require_generated else "warn"

    for relative_path, description in GENERATED_ARTIFACTS.items():
        path = PROJECT_ROOT / relative_path
        if path.exists():
            results.append(CheckResult("ok", f"Generated artifact present: {relative_path}."))
        else:
            results.append(
                CheckResult(
                    missing_status,
                    f"Missing generated artifact ({description}): {relative_path}.",
                )
            )

    return results


def check_persistent_index_manifest(require_generated: bool) -> list[CheckResult]:
    path = PROJECT_ROOT / PERSISTENT_INDEX_MANIFEST
    if not path.exists():
        status = "fail" if require_generated else "warn"
        return [CheckResult(status, f"Persistent index manifest is missing: {PERSISTENT_INDEX_MANIFEST}.")]

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [CheckResult("fail", f"Persistent index manifest is invalid JSON: {exc}.")]

    if manifest.get("format_version") != 3:
        return [
            CheckResult(
                "fail",
                f"Persistent index manifest format is {manifest.get('format_version')}; expected 3.",
            )
        ]

    source_signatures = manifest.get("source_signatures")
    if not isinstance(source_signatures, dict):
        return [CheckResult("fail", "Persistent index manifest has no source_signatures object.")]

    problems: list[str] = []
    for key in ["chunks", "chunk_embeddings"]:
        signature = source_signatures.get(key)
        if not isinstance(signature, dict):
            problems.append(f"{key}: missing signature")
            continue
        if "sha256" not in signature:
            problems.append(f"{key}: missing sha256")
        if "mtime_ns" in signature:
            problems.append(f"{key}: still uses mtime_ns")
        signature_path = signature.get("path")
        if not isinstance(signature_path, str) or Path(signature_path).is_absolute():
            problems.append(f"{key}: path is not project-relative")

    if problems:
        return [CheckResult("fail", "Persistent index manifest is not portable: " + "; ".join(problems))]

    return [CheckResult("ok", "Persistent index manifest uses portable SHA256 source signatures.")]


def check_financebench_raw_checksum(require_generated: bool) -> CheckResult:
    path = PROJECT_ROOT / FINANCEBENCH_RAW_PATH
    if not path.exists():
        status = "fail" if require_generated else "warn"
        return CheckResult(status, f"FinanceBench raw source is missing: {FINANCEBENCH_RAW_PATH}.")

    actual_hash = compute_sha256(path)
    if actual_hash != FINANCEBENCH_RAW_SHA256:
        return CheckResult(
            "fail",
            f"FinanceBench raw checksum mismatch: expected {FINANCEBENCH_RAW_SHA256}, got {actual_hash}.",
        )
    return CheckResult("ok", "FinanceBench raw checksum matches pinned source hash.")


def parse_lock_versions(lock_path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    if not lock_path.exists():
        return versions

    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        requirement = line.split(";", 1)[0].strip()
        if "==" not in requirement:
            continue
        name, version = requirement.split("==", 1)
        versions[normalize_package_name(name)] = version
    return versions


def installed_versions() -> dict[str, str]:
    return {
        normalize_package_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }


def check_locked_packages(skip_package_check: bool) -> list[CheckResult]:
    if skip_package_check:
        return [CheckResult("warn", "Package lock check skipped.")]

    lock_path = PROJECT_ROOT / "requirements" / "lock.txt"
    expected_versions = parse_lock_versions(lock_path)
    if not expected_versions:
        return [CheckResult("fail", "No pinned packages found in requirements/lock.txt.")]

    current_versions = installed_versions()
    missing: list[str] = []
    mismatched: list[str] = []

    for name, expected_version in sorted(expected_versions.items()):
        actual_version = current_versions.get(name)
        if actual_version is None:
            missing.append(name)
        elif actual_version != expected_version:
            mismatched.append(f"{name}: expected {expected_version}, got {actual_version}")

    if missing or mismatched:
        detail = "; ".join(
            [
                f"missing={missing}" if missing else "",
                f"mismatched={mismatched}" if mismatched else "",
            ]
        ).strip("; ")
        return [CheckResult("fail", f"Installed packages do not match lock file: {detail}.")]

    return [CheckResult("ok", "Installed packages match requirements/lock.txt.")]


def check_openai_configuration() -> CheckResult:
    dotenv_values = parse_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or dotenv_values.get("OPENAI_API_KEY")
    if api_key:
        return CheckResult("ok", "OPENAI_API_KEY is configured for live LLM steps.")
    return CheckResult(
        "warn",
        "OPENAI_API_KEY is not configured; live generation, extraction, and judge steps will fail.",
    )


def check_security_cases_not_ignored() -> CheckResult:
    relative_path = "data/evaluation/security/security_cases.csv"
    process = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=PROJECT_ROOT,
        check=False,
    )
    if process.returncode == 1:
        return CheckResult("ok", f"{relative_path} is not ignored by git.")
    if process.returncode == 0:
        return CheckResult("fail", f"{relative_path} is still ignored by git.")
    return CheckResult("warn", "Could not verify git ignore status for security cases.")


def check_pipeline_step_count() -> CheckResult:
    actual_count = len(select_pipeline_steps())
    if actual_count == EXPECTED_PIPELINE_STEPS:
        return CheckResult("ok", f"Pipeline dry-run step count is {EXPECTED_PIPELINE_STEPS}.")
    return CheckResult(
        "fail",
        f"Pipeline step count is {actual_count}; expected {EXPECTED_PIPELINE_STEPS}.",
    )


def build_results(args: argparse.Namespace) -> list[CheckResult]:
    results = [
        check_python_version(strict=args.strict),
        check_security_cases_not_ignored(),
        check_pipeline_step_count(),
        check_openai_configuration(),
    ]
    results.extend(check_required_files())
    results.extend(check_generated_artifacts(require_generated=args.require_generated))
    results.extend(check_persistent_index_manifest(require_generated=args.require_generated))
    results.append(check_financebench_raw_checksum(require_generated=args.require_generated))
    results.extend(check_locked_packages(skip_package_check=args.skip_package_check))
    return results


def print_results(results: list[CheckResult]) -> None:
    labels = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}
    for result in results:
        print(f"[{labels[result.status]}] {result.message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify reproducibility prerequisites and local generated artifacts."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat Python version drift as a failure instead of a warning.",
    )
    parser.add_argument(
        "--require-generated",
        action="store_true",
        help="Fail when generated data/index artifacts are missing.",
    )
    parser.add_argument(
        "--skip-package-check",
        action="store_true",
        help="Skip installed-package comparison against requirements/lock.txt.",
    )
    args = parser.parse_args()

    results = build_results(args)
    print_results(results)

    failures = [result for result in results if result.status == "fail"]
    warnings = [result for result in results if result.status == "warn"]
    print(f"\nSummary: {len(failures)} failure(s), {len(warnings)} warning(s).")
    if failures:
        print("Regenerate generated artifacts with: python -m scripts.run_pipeline")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

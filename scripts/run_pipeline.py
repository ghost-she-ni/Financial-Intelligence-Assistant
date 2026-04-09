from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineStep:
    phase: str
    module_name: str
    args: tuple[str, ...] = ()


PHASE_ORDER = (
    "preparation",
    "indexing",
    "extraction",
    "financebench",
    "evaluation",
    "judge",
    "metrics",
)

COMPARISON_RETRIEVAL_MODES = (
    "classical_ml",
    "naive",
    "improved",
)

PIPELINE_STEPS = (
    PipelineStep(phase="preparation", module_name="scripts.run_ingestion"),
    PipelineStep(phase="preparation", module_name="scripts.run_pdf_extraction"),
    PipelineStep(phase="preparation", module_name="scripts.run_text_cleaning"),
    PipelineStep(phase="preparation", module_name="scripts.run_chunking"),
    PipelineStep(phase="indexing", module_name="scripts.run_chunk_embeddings"),
    PipelineStep(
        phase="indexing",
        module_name="scripts.build_retrieval_index",
        args=("--verbose",),
    ),
    PipelineStep(phase="extraction", module_name="scripts.run_entity_extraction"),
    PipelineStep(phase="extraction", module_name="scripts.run_triplet_extraction"),
    PipelineStep(phase="extraction", module_name="scripts.run_competitor_analysis"),
    PipelineStep(phase="financebench", module_name="scripts.run_financebench_loader"),
    *tuple(
        PipelineStep(
            phase="evaluation",
            module_name="scripts.run_evaluation_pipeline",
            args=("--retrieval_mode", retrieval_mode),
        )
        for retrieval_mode in COMPARISON_RETRIEVAL_MODES
    ),
    *tuple(
        PipelineStep(
            phase="judge",
            module_name="scripts.run_judge",
            args=("--retrieval_mode", retrieval_mode),
        )
        for retrieval_mode in COMPARISON_RETRIEVAL_MODES
    ),
    *tuple(
        PipelineStep(
            phase="metrics",
            module_name="scripts.run_metrics",
            args=("--retrieval_mode", retrieval_mode),
        )
        for retrieval_mode in COMPARISON_RETRIEVAL_MODES
    ),
)


def normalize_phase_name(phase_name: str) -> str:
    """Validate and normalize one pipeline phase name."""
    normalized = phase_name.strip().lower()
    if normalized not in PHASE_ORDER:
        raise ValueError(
            f"Unknown phase '{phase_name}'. Expected one of: {', '.join(PHASE_ORDER)}"
        )
    return normalized


def select_pipeline_steps(
    from_phase: str = "preparation",
    to_phase: str = "metrics",
) -> list[PipelineStep]:
    """Return the ordered subset of pipeline steps between two phases."""
    normalized_from_phase = normalize_phase_name(from_phase)
    normalized_to_phase = normalize_phase_name(to_phase)

    start_index = PHASE_ORDER.index(normalized_from_phase)
    end_index = PHASE_ORDER.index(normalized_to_phase)
    if start_index > end_index:
        raise ValueError(
            f"Invalid phase range: from_phase='{normalized_from_phase}' comes after "
            f"to_phase='{normalized_to_phase}'."
        )

    selected_phases = set(PHASE_ORDER[start_index : end_index + 1])
    return [step for step in PIPELINE_STEPS if step.phase in selected_phases]


def build_step_command(
    python_executable: str,
    step: PipelineStep,
) -> list[str]:
    """Build the subprocess command for one module entry point."""
    return [python_executable, "-m", step.module_name, *step.args]


def print_pipeline_plan(steps: list[PipelineStep]) -> None:
    """Print the selected pipeline plan without executing it."""
    print("\nSelected pipeline steps:")
    for index, step in enumerate(steps, start=1):
        args_suffix = f" {' '.join(step.args)}" if step.args else ""
        print(f"{index:02d}. [{step.phase}] {step.module_name}{args_suffix}")


def run_step(
    python_executable: str,
    project_root: Path,
    step: PipelineStep,
) -> None:
    """Execute one pipeline module step and stop on failure."""
    command = build_step_command(
        python_executable=python_executable,
        step=step,
    )

    print(f"\n>>> START: [{step.phase}] {step.module_name}")
    try:
        subprocess.run(command, check=True, cwd=project_root)
        print(f">>> OK: [{step.phase}] {step.module_name}")
    except subprocess.CalledProcessError:
        print(f"!!! FAILED: [{step.phase}] {step.module_name}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the end-to-end project pipeline, including FinanceBench preparation, "
            "all three retrieval-mode evaluations, judge runs, and metrics by default."
        )
    )
    parser.add_argument(
        "--from-phase",
        type=str,
        default="preparation",
        help=f"First phase to execute. Choices: {', '.join(PHASE_ORDER)}.",
    )
    parser.add_argument(
        "--to-phase",
        type=str,
        default="metrics",
        help=f"Last phase to execute. Choices: {', '.join(PHASE_ORDER)}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected steps without executing them.",
    )
    parser.add_argument(
        "--list-phases",
        action="store_true",
        help="Print the ordered phase names and exit.",
    )

    args = parser.parse_args()

    if args.list_phases:
        print("Available phases:")
        for phase_name in PHASE_ORDER:
            print(f"- {phase_name}")
        return

    try:
        selected_steps = select_pipeline_steps(
            from_phase=args.from_phase,
            to_phase=args.to_phase,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print_pipeline_plan(selected_steps)

    if args.dry_run:
        return

    project_root = Path(__file__).resolve().parents[1]
    for step in selected_steps:
        run_step(
            python_executable=sys.executable,
            project_root=project_root,
            step=step,
        )

    print("\n" + "=" * 40)
    print("FULL PIPELINE COMPLETED")
    print("=" * 40)


if __name__ == "__main__":
    main()

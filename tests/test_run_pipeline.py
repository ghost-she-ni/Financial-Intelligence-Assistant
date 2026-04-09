from __future__ import annotations

import sys
from importlib import util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
SPEC = util.spec_from_file_location("run_pipeline_module", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
RUN_PIPELINE = util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUN_PIPELINE
SPEC.loader.exec_module(RUN_PIPELINE)


def test_select_pipeline_steps_defaults_to_full_pipeline() -> None:
    steps = RUN_PIPELINE.select_pipeline_steps()

    assert steps[0].module_name == "scripts.run_ingestion"
    assert steps[-1].module_name == "scripts.run_metrics"
    assert [step.args for step in steps if step.module_name == "scripts.run_evaluation_pipeline"] == [
        ("--retrieval_mode", "classical_ml"),
        ("--retrieval_mode", "naive"),
        ("--retrieval_mode", "improved"),
    ]
    assert [step.args for step in steps if step.module_name == "scripts.run_judge"] == [
        ("--retrieval_mode", "classical_ml"),
        ("--retrieval_mode", "naive"),
        ("--retrieval_mode", "improved"),
    ]
    assert [step.args for step in steps if step.module_name == "scripts.run_metrics"] == [
        ("--retrieval_mode", "classical_ml"),
        ("--retrieval_mode", "naive"),
        ("--retrieval_mode", "improved"),
    ]


def test_select_pipeline_steps_supports_phase_ranges() -> None:
    steps = RUN_PIPELINE.select_pipeline_steps(
        from_phase="financebench",
        to_phase="judge",
    )

    assert [step.module_name for step in steps] == [
        "scripts.run_financebench_loader",
        "scripts.run_evaluation_pipeline",
        "scripts.run_evaluation_pipeline",
        "scripts.run_evaluation_pipeline",
        "scripts.run_judge",
        "scripts.run_judge",
        "scripts.run_judge",
    ]


def test_build_step_command_uses_module_execution() -> None:
    step = RUN_PIPELINE.PipelineStep(
        phase="evaluation",
        module_name="scripts.run_evaluation_pipeline",
        args=("--verbose",),
    )

    assert RUN_PIPELINE.build_step_command("python", step) == [
        "python",
        "-m",
        "scripts.run_evaluation_pipeline",
        "--verbose",
    ]


def test_select_pipeline_steps_rejects_reversed_phase_ranges() -> None:
    try:
        RUN_PIPELINE.select_pipeline_steps(
            from_phase="metrics",
            to_phase="preparation",
        )
    except ValueError as exc:
        assert "Invalid phase range" in str(exc)
    else:
        raise AssertionError("Expected ValueError for reversed phase range.")

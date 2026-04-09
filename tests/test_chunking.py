from __future__ import annotations

import sys
from importlib import util
from pathlib import Path

import pandas as pd

from src.preprocessing.chunking import chunk_all_documents


def test_chunk_all_documents_token_method_respects_fixed_token_budget(monkeypatch) -> None:
    processed_pages_df = pd.DataFrame(
        [
            {
                "doc_id": "acme_2024_10k",
                "company": "Acme",
                "fiscal_year": 2024,
                "document_type": "10k",
                "file_name": "acme.pdf",
                "page_num": 1,
                "clean_text": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
            }
        ]
    )

    def fake_annotate_document_sections(doc_df: pd.DataFrame) -> pd.DataFrame:
        annotated_df = doc_df.copy()
        annotated_df["section_id"] = "business__01"
        annotated_df["section_code"] = "business"
        annotated_df["section_title"] = "Business"
        annotated_df["section_group"] = "business"
        return annotated_df

    monkeypatch.setattr(
        "src.preprocessing.chunking.annotate_document_sections",
        fake_annotate_document_sections,
    )

    chunks_df = chunk_all_documents(
        processed_pages_df=processed_pages_df,
        chunk_size=5,
        overlap=1,
        min_chunk_words=1,
        method="token",
        progress=False,
    )

    assert chunks_df["token_count"].tolist() == [5, 5, 4]
    assert chunks_df["word_count"].tolist() == [5, 5, 4]
    assert chunks_df["doc_id"].tolist() == ["acme_2024_10k"] * 3


def test_run_entity_and_triplet_wrappers_default_to_full_corpus_extraction(monkeypatch) -> None:
    captured_commands: list[list[str]] = []

    def fake_run(command: list[str], check: bool, cwd: Path) -> None:
        assert check is True
        assert (cwd / "scripts").exists()
        assert (cwd / "src").exists()
        captured_commands.append(command)

    monkeypatch.setattr("subprocess.run", fake_run)

    for module_name in ["run_entity_extraction", "run_triplet_extraction"]:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{module_name}.py"
        spec = util.spec_from_file_location(f"test_{module_name}", script_path)
        assert spec is not None
        assert spec.loader is not None
        module = util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        monkeypatch.setattr(sys, "argv", [module_name])
        module.main()

    assert len(captured_commands) == 2
    for command in captured_commands:
        assert "--mode" in command
        assert command[command.index("--mode") + 1] == "all"
        assert "--max_chunks" not in command


def test_entity_and_triplet_cli_defaults_do_not_limit_chunk_count(monkeypatch) -> None:
    from src.extraction import entity_extractor, triplet_extractor

    captured_entity_kwargs: dict[str, object] = {}
    captured_triplet_kwargs: dict[str, object] = {}

    def fake_run_entity_extraction(**kwargs):
        captured_entity_kwargs.update(kwargs)
        return pd.DataFrame()

    def fake_run_triplet_extraction(**kwargs):
        captured_triplet_kwargs.update(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr(entity_extractor, "run_entity_extraction", fake_run_entity_extraction)
    monkeypatch.setattr(triplet_extractor, "run_triplet_extraction", fake_run_triplet_extraction)
    monkeypatch.setattr(entity_extractor, "print_summary", lambda df: None)
    monkeypatch.setattr(triplet_extractor, "print_summary", lambda df: None)

    monkeypatch.setattr(sys, "argv", ["entity_extractor"])
    entity_extractor.main()

    monkeypatch.setattr(sys, "argv", ["triplet_extractor"])
    triplet_extractor.main()

    assert captured_entity_kwargs["mode"] == "all"
    assert captured_entity_kwargs["max_chunks"] is None
    assert captured_triplet_kwargs["mode"] == "all"
    assert captured_triplet_kwargs["max_chunks"] is None

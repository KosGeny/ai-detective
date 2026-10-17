from pathlib import Path
import pytest

from src.data_io import load_dossier
from src.evaluation import compare_modes, compare_stress, summarize
from src.experiment import load_results, load_run, make_contexts, run_experiment
from src.llm import LLMResult
from src.schemas import ModelAnswer
from generate_report import build_report
from revalidate_results import revalidate


def test_real_dossier_and_all_contexts():
    documents, claims, questions, cases = load_dossier(Path(__file__).parents[1] / "data")
    assert (len(documents), len(claims), len(questions), len(cases)) == (10, 38, 12, 36)
    count = 0
    for question in questions:
        for case in [None, *(case for case in cases if case.question_id == question.question_id)]:
            contexts = dict(make_contexts(question, documents, claims, case))
            assert set(contexts["B"].claim_ids) == set(contexts["C"].claim_ids)
            assert len(contexts["B"].claim_ids) <= 15
            if case and case.case_id == "Q03_delete":
                assert "C12" not in contexts["A"].claim_ids
                assert "C12" not in contexts["A"].text
            if case and case.case_id == "Q03_contradiction":
                assert "DX" in contexts["A"].source_document_ids
                assert "CX03" in contexts["B"].source_claim_ids
            count += len(contexts)
    assert count == 144


def test_shuffle_changes_only_mode_a_context():
    documents, claims, questions, cases = load_dossier(Path(__file__).parents[1] / "data")
    for question in questions:
        case = next(c for c in cases if c.question_id == question.question_id and c.kind == "shuffle")
        base = dict(make_contexts(question, documents, claims))
        shuffled = dict(make_contexts(question, documents, claims, case))
        assert base["A"].text != shuffled["A"].text
        assert base["B"].text == shuffled["B"].text
        assert base["C"].text == shuffled["C"].text


def test_revalidation_preserves_raw_and_original_error():
    raw = ('{"question_id":"Q01","status":"contradicted","answer":"Нет",'
           '"evidence_ids":["C01"],"counterevidence_ids":[],"confidence":0.8,'
           '"explanation":"Основание C01."}')
    row = {"question_id": "Q01", "context_claim_ids": ["C01"], "raw_output": raw,
           "answer": None, "error": "Для contradicted нужны counterevidence_ids",
           "error_type": "InvalidModelAnswer"}
    result = revalidate([row])[0]
    assert result["raw_output"] == raw
    assert result["original_error"] == row["error"]
    assert result["revalidated_from_raw"] is True
    assert result["answer"]["counterevidence_ids"] == ["C01"]


class FakeLLM:
    model = "fake-test-model"
    system_prompt = "Test prompt"

    def ask(self, question_id, question, context):
        return LLMResult(
            answer=ModelAnswer(question_id=question_id, status="insufficient", answer=None,
                               evidence_ids=[], counterevidence_ids=[], confidence=0.3,
                               explanation="Недостаточно данных для ответа."),
            input_tokens=10, output_tokens=5, elapsed_seconds=0.01, response_id="fake", raw_output="{}",
        )


def test_full_pipeline_writes_144_results(tmp_path):
    output = tmp_path / "runs.jsonl"
    rows = run_experiment(data_dir=Path(__file__).parents[1] / "data", output=output, llm=FakeLLM())
    assert len(rows) == 144
    assert len(load_results(output)) == 144
    assert all(row["error"] is None for row in rows)
    assert all(row["raw_output"] == "{}" for row in rows)
    assert any(row["original_document_ids"] != row["source_document_ids"] for row in rows)
    assert summarize(rows)["valid"] == 144
    assert compare_modes(rows)["same_retrieved_ids"] is True
    assert compare_stress(rows)["shuffle"]["cases"] == 36
    with pytest.raises(ValueError, match="реальную LLM"):
        build_report(rows, tmp_path / "report.pdf")


def test_partial_modes_can_resume_same_run(tmp_path):
    output = tmp_path / "runs.jsonl"
    data = Path(__file__).parents[1] / "data"
    first = run_experiment(data_dir=data, output=output, llm=FakeLLM(), modes=("B", "C"), run_id="integration")
    second = run_experiment(data_dir=data, output=output, llm=FakeLLM(), modes=("A",), run_id="integration")
    assert len(first) == 96
    assert len(second) == 48
    assert len(load_run(output, "integration")) == 144
    assert run_experiment(data_dir=data, output=output, llm=FakeLLM(), modes=("A", "B", "C"), run_id="integration") == []

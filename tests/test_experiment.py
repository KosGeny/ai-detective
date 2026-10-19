from pathlib import Path
import pytest

from src.data_io import evidence_issues, load_dossier
from src.evaluation import compare_modes, compare_stress, summarize
from src.experiment import load_results, load_run, make_contexts, run_experiment
from src.llm import LLMResult
from src.schemas import ModelAnswer
from revalidate_results import revalidate


def test_real_dossier_and_all_contexts():
    documents, claims, questions, cases = load_dossier(Path(__file__).parents[1] / "data", allow_nonverbatim_evidence=True)
    assert (len(documents), len(claims), len(questions), len(cases)) == (10, 37, 12, 36)
    count = 0
    for question in questions:
        for case in [None, *(case for case in cases if case.question_id == question.question_id)]:
            contexts = dict(make_contexts(question, documents, claims, case, allow_nonverbatim_evidence=True))
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
    documents, claims, questions, cases = load_dossier(Path(__file__).parents[1] / "data", allow_nonverbatim_evidence=True)
    for question in questions:
        case = next(c for c in cases if c.question_id == question.question_id and c.kind == "shuffle")
        base = dict(make_contexts(question, documents, claims))
        shuffled = dict(make_contexts(question, documents, claims, case, allow_nonverbatim_evidence=True))
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
    rows = run_experiment(data_dir=Path(__file__).parents[1] / "data", output=output, llm=FakeLLM(), allow_nonverbatim_evidence=True)
    assert len(rows) == 144
    assert len(load_results(output)) == 144
    assert all(row["error"] is None for row in rows)
    assert all(row["raw_output"] == "{}" for row in rows)
    assert any(row["original_document_ids"] != row["source_document_ids"] for row in rows)
    assert summarize(rows)["valid"] == 144
    assert compare_modes(rows)["same_retrieved_ids"] is True
    assert compare_stress(rows)["shuffle"]["cases"] == 36


def test_partial_modes_can_resume_same_run(tmp_path):
    output = tmp_path / "runs.jsonl"
    data = Path(__file__).parents[1] / "data"
    first = run_experiment(data_dir=data, output=output, llm=FakeLLM(), allow_nonverbatim_evidence=True, modes=("B", "C"), run_id="integration")
    second = run_experiment(data_dir=data, output=output, llm=FakeLLM(), allow_nonverbatim_evidence=True, modes=("A",), run_id="integration")
    assert len(first) == 96
    assert len(second) == 48
    assert len(load_run(output, "integration")) == 144
    assert run_experiment(data_dir=data, output=output, llm=FakeLLM(), allow_nonverbatim_evidence=True, modes=("A", "B", "C"), run_id="integration") == []


def test_approved_main_reports_nonverbatim_evidence_without_rewriting():
    data = Path(__file__).parents[1] / "data"
    before = {p.name: p.read_bytes() for p in data.glob("*.jsonl")}
    with pytest.raises(ValueError, match="C28"):
        load_dossier(data)
    documents, claims, _, cases = load_dossier(data, allow_nonverbatim_evidence=True)
    assert {issue["owner"] for issue in evidence_issues(documents, claims, cases)} == {
        "C28", "C35", "Q07_contradiction", "Q12_contradiction"
    }
    assert before == {p.name: p.read_bytes() for p in data.glob("*.jsonl")}


def test_nonverbatim_option_does_not_allow_unknown_document(tmp_path):
    import json
    source = Path(__file__).parents[1] / "data"
    for path in source.glob("*.jsonl"):
        (tmp_path / path.name).write_bytes(path.read_bytes())
    path = tmp_path / "claims.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["document_id"] = "DOES_NOT_EXIST"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    with pytest.raises(ValueError, match="неизвестный документ"):
        load_dossier(tmp_path, allow_nonverbatim_evidence=True)


def test_extension_reuses_120_observations_and_requests_only_24(tmp_path):
    import json
    source = Path(__file__).parents[1] / "data"
    baseline = tmp_path / "original"
    baseline.mkdir()
    for path in source.glob("*.jsonl"):
        (baseline / path.name).write_bytes(path.read_bytes())
    questions_path = baseline / "questions.jsonl"
    original_questions = [json.loads(line) for line in questions_path.read_text().splitlines()]
    for question in original_questions:
        question.pop("question_type", None)
    questions_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in original_questions) + "\n")
    controls = {f"Q{i:02d}_delete" for i in [1, 2, 7, 8, 9, 10, 11, 12]}
    path = baseline / "stress_cases.jsonl"
    cases = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in cases if row["case_id"] not in controls) + "\n")
    output = tmp_path / "extended.jsonl"
    first = run_experiment(data_dir=baseline, output=output, llm=FakeLLM(), run_id="extend", allow_nonverbatim_evidence=True)
    second = run_experiment(data_dir=source, output=output, llm=FakeLLM(), run_id="extend", allow_nonverbatim_evidence=True)
    assert len(first) == 120
    assert len(second) == 24
    assert {row["case_id"] for row in second} == controls
    assert load_results(output)[:120] == first
    assert len(load_run(output, "extend")) == 144

"""Reproducible baseline and stress runs with append-only JSONL results."""

import json
import gzip
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .context_builder import build_context
from .data_io import load_dossier
from .llm import DeepSeekLLM, InvalidModelAnswer, LLMUnavailable
from .schemas import BuiltContext, Claim, Document, Question, StressCase
from .stress import apply_case


def load_results(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def make_contexts(question: Question, documents: list[Document], claims: list[Claim], case: StressCase | None = None) -> list[tuple[str, BuiltContext]]:
    if case is not None:
        contexts = [(mode, context) for mode, context, _ in apply_case(case, question.question, documents, claims)]
    else:
        contexts = [(mode, build_context(mode, question.question_id, question.question, documents, claims)) for mode in ("A", "B", "C")]
    by_mode = dict(contexts)
    if set(by_mode["B"].claim_ids) != set(by_mode["C"].claim_ids):
        raise ValueError("B и C получили разные наборы claim_id")
    return contexts


def run_one(llm: DeepSeekLLM, question: Question, mode: str, context: BuiltContext, case: StressCase | None = None, run_id: str = "manual", original_document_ids: list[str] | None = None, original_claim_ids: list[str] | None = None) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    result: dict[str, Any] = {
        "run_id": run_id, "created_at": now, "model": llm.model,
        "question_id": question.question_id, "question": question.question,
        "mode": mode, "case_id": case.case_id if case else "baseline", "kind": case.kind if case else "baseline",
        "expected_status": case.expected_status if case else question.gold_status,
        "gold_evidence_ids": case.gold_evidence_ids if case else question.gold_evidence_ids,
        "gold_counterevidence_ids": case.gold_counterevidence_ids if case else question.gold_counterevidence_ids,
        "context_claim_ids": context.claim_ids,
        "original_document_ids": original_document_ids if original_document_ids is not None else context.source_document_ids,
        "original_claim_ids": original_claim_ids if original_claim_ids is not None else context.source_claim_ids,
        "source_document_ids": context.source_document_ids,
        "source_claim_ids": context.source_claim_ids,
        "context_text": context.text, "branches": {str(key): value for key, value in context.branches.items()},
        "parameters": {"model": llm.model, "max_tokens": 1200, "temperature": getattr(llm, "temperature", 0.0), "thinking": "disabled", "response_format": "json_object", "system_prompt": llm.system_prompt},
        "answer": None, "raw_output": None, "error": None, "error_type": None, "input_tokens": None, "output_tokens": None,
        "elapsed_seconds": None, "response_id": None,
    }
    try:
        response = llm.ask(question.question_id, question.question, context)
        result.update(answer=response.answer.model_dump(), input_tokens=response.input_tokens,
                      output_tokens=response.output_tokens, elapsed_seconds=response.elapsed_seconds,
                      response_id=response.response_id, raw_output=response.raw_output)
    except (LLMUnavailable, InvalidModelAnswer) as exc:
        result["error"] = str(exc)
        result["error_type"] = type(exc).__name__
        if isinstance(exc, InvalidModelAnswer):
            result["raw_output"] = exc.raw_output
    return result


def run_experiment(data_dir: str | Path = "data", output: str | Path = "results/runs.jsonl", include_stress: bool = True, llm: DeepSeekLLM | None = None, progress: Callable[[int, int], None] | None = None, modes: tuple[str, ...] = ("A", "B", "C"), run_id: str | None = None) -> list[dict[str, Any]]:
    documents, claims, questions, cases = load_dossier(data_dir)
    llm = llm or DeepSeekLLM()
    if isinstance(llm, DeepSeekLLM):
        llm._get_client()  # Fail before creating result files when credentials are absent.
    tasks: list[tuple[Question, StressCase | None, str, BuiltContext]] = []
    for question in questions:
        for mode, context in make_contexts(question, documents, claims):
            if mode in modes:
                tasks.append((question, None, mode, context))
        if include_stress:
            for case in cases:
                if case.question_id == question.question_id:
                    for mode, context in make_contexts(question, documents, claims, case):
                        if mode in modes:
                            tasks.append((question, case, mode, context))
    run_id = run_id or uuid.uuid4().hex
    output = Path(output)
    if output.suffix == ".gz":
        raise ValueError("Для записи прогона укажите обычный JSONL; .gz поддерживается только для чтения архива")
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = [row for row in load_results(output) if row["run_id"] == run_id]
    if existing and any(row["model"] != llm.model or row["parameters"]["system_prompt"] != llm.system_prompt
                        or row["parameters"].get("temperature") != getattr(llm, "temperature", 0.0) for row in existing):
        raise ValueError("Нельзя продолжить run_id с другой моделью, промптом или температурой")
    # A model answer with invalid logic is still an observed result of this run.
    # Retry only transport/API failures, so resuming does not change the sample.
    finished = {(row["question_id"], row["case_id"], row["mode"]) for row in existing
                if row["answer"] is not None or row["error_type"] == "InvalidModelAnswer"}
    tasks = [task for task in tasks if (task[0].question_id, task[1].case_id if task[1] else "baseline", task[2]) not in finished]
    rows: list[dict[str, Any]] = []
    with output.open("a", encoding="utf-8") as stream:
        for index, (question, case, mode, context) in enumerate(tasks, 1):
            row = run_one(llm, question, mode, context, case, run_id,
                          original_document_ids=[document.document_id for document in documents],
                          original_claim_ids=[claim.claim_id for claim in claims])
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            if progress:
                progress(index, len(tasks))
            if row["error_type"] == "LLMUnavailable":
                break
    return rows


def load_run(path: str | Path, run_id: str) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in load_results(path):
        if row["run_id"] == run_id:
            latest[(row["question_id"], row["case_id"], row["mode"])] = row
    return list(latest.values())

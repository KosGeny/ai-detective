"""Loading and validating the dossier before an experiment starts."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from .schemas import Claim, Document, Question, StressCase

T = TypeVar("T", bound=BaseModel)


def load_jsonl(path: str | Path, model: type[T]) -> list[T]:
    rows: list[T] = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(model.model_validate_json(line))
                except Exception as exc:
                    raise ValueError(f"{path}:{number}: {exc}") from exc
    return rows


def load_dossier(directory: str | Path = "data") -> tuple[list[Document], list[Claim], list[Question], list[StressCase]]:
    directory = Path(directory)
    documents = load_jsonl(directory / "documents.jsonl", Document)
    claims = load_jsonl(directory / "claims.jsonl", Claim)
    questions = load_jsonl(directory / "questions.jsonl", Question)
    cases = load_jsonl(directory / "stress_cases.jsonl", StressCase)
    _unique(documents, "document_id")
    _unique(claims, "claim_id")
    _unique(questions, "question_id")
    _unique(cases, "case_id")
    by_document = {document.document_id: document for document in documents}
    claim_ids = {claim.claim_id for claim in claims}
    question_ids = {question.question_id for question in questions}
    for claim in claims:
        document = by_document.get(claim.document_id)
        if document is None or claim.evidence not in document.text:
            raise ValueError(f"{claim.claim_id}: evidence отсутствует дословно в {claim.document_id}")
    for question in questions:
        _check_ids(question.gold_evidence_ids + question.gold_counterevidence_ids, claim_ids, question.question_id)
    for case in cases:
        if case.question_id not in question_ids:
            raise ValueError(f"{case.case_id}: неизвестный question_id")
        if case.kind == "delete" and case.remove_claim_id not in claim_ids:
            raise ValueError(f"{case.case_id}: неизвестный удаляемый claim_id")
        if case.kind == "contradiction":
            if case.added_claim is None or case.added_document is None:
                raise ValueError(f"{case.case_id}: нужны added_claim и added_document")
            if case.added_claim.document_id != case.added_document.document_id or case.added_claim.evidence not in case.added_document.text:
                raise ValueError(f"{case.case_id}: added_claim не подтверждён added_document")
        allowed = claim_ids | ({case.added_claim.claim_id} if case.added_claim else set())
        _check_ids(case.gold_evidence_ids + case.gold_counterevidence_ids, allowed, case.case_id)
    return documents, claims, questions, cases


def _unique(rows: list[BaseModel], field: str) -> None:
    values = [getattr(row, field) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Повторяющиеся {field}")


def _check_ids(ids: list[str], allowed: set[str], owner: str) -> None:
    unknown = set(ids) - allowed
    if unknown:
        raise ValueError(f"{owner}: неизвестные claim_id {sorted(unknown)}")

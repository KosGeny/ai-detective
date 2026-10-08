import random
from pathlib import Path

from .context_builder import BuiltContext, build_context
from .schemas import Claim, Document, StressCase, StressResult


def load_stress_cases(path: str | Path) -> list[StressCase]:
    cases: list[StressCase] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(StressCase.model_validate_json(line))
    return cases


def cases_for_question(cases: list[StressCase], question_id: str) -> list[StressCase]:
    return [c for c in cases if c.question_id == question_id]


def _shuffle_documents(documents: list[Document], seed: int = 42) -> list[Document]:
    rng = random.Random(seed)
    shuffled = list(documents)
    rng.shuffle(shuffled)
    return shuffled


def _delete_claim(claims: list[Claim], remove_claim_id: str) -> list[Claim]:
    if remove_claim_id is None:
        raise ValueError("delete-кейс без remove_claim_id")
    result = [c for c in claims if c.claim_id != remove_claim_id]
    if len(result) == len(claims):
        raise ValueError(
            f"delete-кейс: claim_id={remove_claim_id} не найден в корпусе"
        )
    return result

def _delete_evidence_from_documents(
    documents: list[Document],
    claims: list[Claim],
    remove_claim_id: str,
) -> list[Document]:
    target = next((c for c in claims if c.claim_id == remove_claim_id), None)
    if target is None:
        raise ValueError(f"delete: claim_id={remove_claim_id} не найден")

    result: list[Document] = []
    for d in documents:
        if d.document_id != target.document_id:
            result.append(d)
            continue

        if target.evidence not in d.text:
            raise ValueError(
                f"delete: evidence claim'а {remove_claim_id} не найдено "
                f"в документе {d.document_id}"
            )

        new_text = d.text.replace(target.evidence, "", 1)
        new_text = " ".join(new_text.split())

        result.append(d.model_copy(update={"text": new_text}))
    return result


def _add_contradiction(claims: list[Claim], added: Claim) -> list[Claim]:
    if any(c.claim_id == added.claim_id for c in claims):
        raise ValueError(f"claim_id={added.claim_id} уже существует")
    return list(claims) + [added]


def apply_case(
    case: StressCase,
    question: str,
    documents: list[Document],
    claims: list[Claim],
    modes: tuple[str, ...] = ("A", "B", "C"),
    k: int = 15,
) -> list[tuple[str, BuiltContext, StressResult]]:
    docs = list(documents)
    cls = list(claims)
    removed_id: str | None = None
    added_id: str | None = None

    if case.kind == "shuffle":
        docs = _shuffle_documents(docs, seed=42)

    elif case.kind == "delete":
        cls = _delete_claim(cls, case.remove_claim_id)
        docs = _delete_evidence_from_documents(docs, claims, case.remove_claim_id)
        removed_id = case.remove_claim_id

    elif case.kind == "contradiction":
        if case.added_claim is None:
            raise ValueError("contradiction-кейс без added_claim")
        cls = _add_contradiction(cls, case.added_claim)
        added_id = case.added_claim.claim_id

    else:
        raise ValueError(f"Неизвестный kind: {case.kind}")

    out: list[tuple[str, BuiltContext, StressResult]] = []
    for mode in modes:
        mode_docs = docs
        if mode == "A" and case.added_document is not None:
            mode_docs = docs + [case.added_document]

        ctx = build_context(
            mode=mode,
            question_id=case.question_id,
            question=question,
            documents=mode_docs,
            claims=cls,
            k=k,
        )
        meta = StressResult(
            case_id=case.case_id,
            question_id=case.question_id,
            kind=case.kind,
            expected_status=case.expected_status,
            gold_evidence_ids=list(case.gold_evidence_ids),
            gold_counterevidence_ids=list(case.gold_counterevidence_ids),
            context_b=ctx,
            context_c=ctx,
            removed_claim_id=removed_id,
            added_claim_id=added_id,
            review_note=case.review_note,
        )
        out.append((mode, ctx, meta))
    return out
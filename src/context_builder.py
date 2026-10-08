from datetime import datetime, timezone
from typing import Iterable

from .padic import encode_all, group_by_branch
from .retrieval import BM25Retriever, K
from .schemas import BuiltContext, Claim, Document, Mode


def build_mode_a(
    question_id: str,
    documents: Iterable[Document],
) -> BuiltContext:
    parts: list[str] = []
    for d in documents:
        parts.append(
            f"=== {d.document_id} | {d.title} | source_type={d.source_type} | "
            f"created_at={d.created_at.isoformat()} | author={d.author} ===\n"
            f"{d.text}"
        )
    text = "\n\n".join(parts)
    return BuiltContext(
        mode="A",
        question_id=question_id,
        text=text,
        claim_ids=[],
        branches={},
    )


def _format_claim_line(claim: Claim) -> str:
    return f"{claim.claim_id}: {claim.text}"


def build_mode_b(
    question_id: str,
    question: str,
    claims: Iterable[Claim],
    k: int = K,
) -> BuiltContext:
    retriever = BM25Retriever(claims)
    claim_ids = retriever.retrieve(question, k=k)
    by_id = {c.claim_id: c for c in retriever.claims}

    text = "\n".join(_format_claim_line(by_id[cid]) for cid in claim_ids)
    return BuiltContext(
        mode="B",
        question_id=question_id,
        text=text,
        claim_ids=list(claim_ids),
        branches={},
    )


_SENTINEL = datetime.max.replace(tzinfo=timezone.utc)


def _normalize(et: datetime | None) -> datetime:
    if et is None:
        return _SENTINEL
    if et.tzinfo is None:
        return et.replace(tzinfo=timezone.utc)
    return et


def build_mode_c(
    question_id: str,
    question: str,
    claims: Iterable[Claim],
    k: int = K,
) -> BuiltContext:
    retriever = BM25Retriever(claims)
    claim_ids = retriever.retrieve(question, k=k)
    by_id = {c.claim_id: c for c in retriever.claims}

    selected_claims = [by_id[cid] for cid in claim_ids]
    paths = encode_all(selected_claims)
    groups = group_by_branch(claim_ids, paths)

    parts: list[str] = []
    branches_out: dict[tuple[int, int, int], list[str]] = {}

    for path in sorted(groups.keys(), key=lambda p: p.as_tuple()):
        cids = groups[path]
        cids_sorted = sorted(
            cids,
            key=lambda cid: (_normalize(by_id[cid].event_time), cid),
        )
        branches_out[path.as_tuple()] = cids_sorted
        header = f"ВЕТВЬ {path.label()}"
        body = "\n".join(f"{cid}: {by_id[cid].text}" for cid in cids_sorted)
        parts.append(f"{header}\n{body}")

    text = "\n\n".join(parts)
    return BuiltContext(
        mode="C",
        question_id=question_id,
        text=text,
        claim_ids=list(claim_ids),
        branches=branches_out,
    )


def build_context(
    mode: Mode,
    question_id: str,
    question: str,
    documents: Iterable[Document],
    claims: Iterable[Claim],
    k: int = K,
) -> BuiltContext:
    documents = list(documents)
    claims = list(claims)

    if mode == "A":
        return build_mode_a(question_id, documents)
    if mode == "B":
        return build_mode_b(question_id, question, claims, k)
    if mode == "C":
        return build_mode_c(question_id, question, claims, k)
    raise ValueError(f"Неизвестный режим: {mode}")
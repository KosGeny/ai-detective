from datetime import datetime

import pytest

from src.schemas import Claim, Document


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s)


@pytest.fixture
def make_claim():
    counter = {"n": 0}

    def _make(
        text: str = "тестовое утверждение",
        document_id: str = "D01",
        evidence: str = "тестовое утверждение",
        status: str = "direct",
        path: tuple[int, int, int] = (0, 0, 0),
        event_time: datetime | None = None,
        claim_id: str | None = None,
    ) -> Claim:
        counter["n"] += 1
        cid = claim_id or f"C{counter['n']:02d}"
        return Claim(
            claim_id=cid,
            text=text,
            document_id=document_id,
            evidence=evidence,
            status=status,
            path=path,
            event_time=event_time,
        )

    return _make


@pytest.fixture
def make_document():
    def _make(
        document_id: str = "D01",
        title: str = "Документ",
        source_type: str = "log",
        created_at: str = "2026-03-14T22:00:00+03:00",
        author: str = "автор",
        text: str = "текст документа",
    ) -> Document:
        return Document(
            document_id=document_id,
            title=title,
            source_type=source_type,
            created_at=_ts(created_at),
            author=author,
            text=text,
        )

    return _make


@pytest.fixture
def sample_claims(make_claim):
    claims: list[Claim] = []
    for i in range(20):
        a0 = i % 3
        a1 = (i // 3) % 3
        a2 = (i // 9) % 3
        minute = i
        claims.append(
            make_claim(
                claim_id=f"C{i + 1:02d}",
                text=f"утверждение номер {i + 1} про шкаф Ш-4 и Нереиду",
                evidence=f"evidence {i + 1}",
                document_id=f"D{(i % 3) + 1:02d}",
                path=(a0, a1, a2),
                event_time=_ts(f"2026-03-14T22:{minute:02d}:00+03:00"),
            )
        )
    return claims


@pytest.fixture
def sample_documents(make_document):
    return [
        make_document(document_id="D01", source_type="log",
                      created_at="2026-03-14T22:00:00+03:00"),
        make_document(document_id="D02", source_type="statement",
                      created_at="2026-03-14T23:00:00+03:00"),
        make_document(document_id="D03", source_type="rule",
                      created_at="2026-03-01T09:00:00+03:00"),
    ]
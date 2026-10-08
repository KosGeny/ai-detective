from typing import Literal, Optional
from datetime import datetime

from pydantic import BaseModel, Field

from dataclasses import dataclass, field


ClaimStatus = Literal["direct", "reported", "inferred", "disputed"]
LogicStatus = Literal["entailed", "contradicted", "both", "insufficient"]
Mode = Literal["A", "B", "C"]
StressKind = Literal["shuffle", "delete", "contradiction"]


class Document(BaseModel):
    document_id: str
    title: str
    source_type: str
    created_at: datetime
    author: str
    text: str


class Claim(BaseModel):
    claim_id: str
    text: str
    document_id: str
    evidence: str
    status: ClaimStatus
    path: tuple[int, int, int]
    event_time: Optional[datetime] = None


class Question(BaseModel):
    question_id: str
    question: str
    gold_status: LogicStatus
    gold_answer: Optional[str] = None
    gold_evidence_ids: list[str] = Field(default_factory=list)
    gold_counterevidence_ids: list[str] = Field(default_factory=list)


@dataclass
class BuiltContext:
    mode: Mode
    question_id: str
    text: str
    claim_ids: list[str] = field(default_factory=list)
    branches: dict[tuple[int, int, int], list[str]] = field(default_factory=dict)
    token_estimate: int = 0


class StressCase(BaseModel):
    case_id: str
    question_id: str
    kind: StressKind
    remove_claim_id: Optional[str] = None
    added_claim: Optional[Claim] = None
    added_document: Optional[Document] = None
    expected_status: LogicStatus
    gold_evidence_ids: list[str] = Field(default_factory=list)
    gold_counterevidence_ids: list[str] = Field(default_factory=list)
    review_note: Optional[str] = None


class StressResult(BaseModel):
    case_id: str
    question_id: str
    kind: StressKind
    expected_status: LogicStatus
    gold_evidence_ids: list[str]
    gold_counterevidence_ids: list[str]
    context_b: BuiltContext
    context_c: BuiltContext
    removed_claim_id: Optional[str] = None
    added_claim_id: Optional[str] = None
    review_note: str = ""

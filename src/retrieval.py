from typing import Iterable

from rank_bm25 import BM25Okapi
from razdel import tokenize

from .schemas import Claim

K = 15


def _tokenize(text: str) -> list[str]:
    return [t.text for t in tokenize(text)]


class BM25Retriever:
    def __init__(self, claims: Iterable[Claim]):
        self.claims: list[Claim] = list(claims)
        if not self.claims:
            raise ValueError("BM25Retriever: пустой список утверждений")

        self.claims.sort(key=lambda c: c.claim_id)

        self._corpus_tokens = [_tokenize(c.text) for c in self.claims]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def retrieve(self, question: str, k: int = K) -> list[str]:
        if k <= 0:
            return []

        query_tokens = _tokenize(question)
        if not query_tokens:
            return [c.claim_id for c in self.claims[:k]]

        scores = self._bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.claims, scores),
            key=lambda pair: (-float(pair[1]), pair[0].claim_id),
        )
        return [c.claim_id for c, _ in ranked[:k]]

    def retrieve_claims(self, question: str, k: int = K) -> list[Claim]:
        ids = self.retrieve(question, k=k)
        by_id = {c.claim_id: c for c in self.claims}
        return [by_id[i] for i in ids]


def retrieve_claims(question: str, claims: Iterable[Claim], k: int = K) -> list[str]:
    return BM25Retriever(claims).retrieve(question, k=k)
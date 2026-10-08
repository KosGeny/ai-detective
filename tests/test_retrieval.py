from src.retrieval import K, BM25Retriever, retrieve_claims


def test_bm25_returns_at_most_15(sample_claims):
    assert len(sample_claims) > 15
    retriever = BM25Retriever(sample_claims)
    ids = retriever.retrieve("шкаф Ш-4", k=15)
    assert len(ids) == 15
    assert len(set(ids)) == 15
    assert K == 15


def test_bm25_respects_smaller_k(sample_claims):
    retriever = BM25Retriever(sample_claims)
    ids = retriever.retrieve("Нереида", k=5)
    assert len(ids) <= 5


def test_bm25_returns_all_when_corpus_smaller(make_claim):
    claims = [make_claim(claim_id=f"C{i:02d}", text=f"текст {i}") for i in range(3)]
    ids = retrieve_claims("текст", claims, k=15)
    assert len(ids) == 3
    assert len(set(ids)) == 3
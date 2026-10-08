from src.context_builder import BuiltContext, build_context

import pytest


def assert_same_claim_ids(ctx_b: BuiltContext, ctx_c: BuiltContext) -> None:
    """B и C обязаны получить одинаковый набор claim_id."""
    if set(ctx_b.claim_ids) != set(ctx_c.claim_ids):
        only_b = set(ctx_b.claim_ids) - set(ctx_c.claim_ids)
        only_c = set(ctx_c.claim_ids) - set(ctx_b.claim_ids)
        raise AssertionError(
            f"Режимы B и C получили разные claim_id: only_B={only_b}, only_C={only_c}"
        )
    
def test_modes_b_and_c_get_same_claim_ids(sample_claims, sample_documents):
    question = "Что произошло с зарядным шкафом Ш-4 ночью?"
    ctx_b = build_context(
        mode="B",
        question_id="Q01",
        question=question,
        documents=sample_documents,
        claims=sample_claims,
    )
    ctx_c = build_context(
        mode="C",
        question_id="Q01",
        question=question,
        documents=sample_documents,
        claims=sample_claims,
    )

    assert set(ctx_b.claim_ids) == set(ctx_c.claim_ids)
    assert len(ctx_b.claim_ids) == len(ctx_c.claim_ids)
    assert len(ctx_b.claim_ids) <= 15

    # явная проверка через публичный хелпер
    assert_same_claim_ids(ctx_b, ctx_c)


def test_assert_same_claim_ids_raises_on_mismatch():
    """Хелпер должен падать, если наборы разные — иначе он бесполезен."""
    b = BuiltContext(mode="B", question_id="Q01", text="",
                     claim_ids=["C01", "C02"])
    c = BuiltContext(mode="C", question_id="Q01", text="",
                     claim_ids=["C01", "C03"])
    with pytest.raises(AssertionError):
        assert_same_claim_ids(b, c)
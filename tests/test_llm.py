from types import SimpleNamespace

import pytest

from src.llm import DeepSeekLLM, InvalidModelAnswer, LLMUnavailable, get_api_key, parse_model_answer
from src.schemas import BuiltContext


def payload(**changes):
    value = {
        "question_id": "Q01", "status": "entailed", "answer": "Да",
        "evidence_ids": ["C01"], "counterevidence_ids": [],
        "confidence": 0.8, "explanation": "Это следует из C01.",
    }
    value.update(changes)
    return value


def test_invalid_json():
    with pytest.raises(InvalidModelAnswer):
        parse_model_answer("{oops", "Q01", {"C01"})


def test_unknown_evidence_id():
    with pytest.raises(InvalidModelAnswer, match="claim_id"):
        parse_model_answer(payload(evidence_ids=["C99"]), "Q01", {"C01"})


def test_contradicted_refuting_id_in_evidence_is_normalized():
    answer = parse_model_answer(payload(status="contradicted", answer="Нет"), "Q01", {"C01"})
    assert answer.evidence_ids == []
    assert answer.counterevidence_ids == ["C01"]


def test_both_without_counterevidence():
    with pytest.raises(InvalidModelAnswer, match="both"):
        parse_model_answer(payload(status="both"), "Q01", {"C01"})


def test_insufficient_with_invented_answer():
    with pytest.raises(InvalidModelAnswer, match="insufficient"):
        parse_model_answer(payload(status="insufficient", answer="Илья управлял катером", evidence_ids=[]), "Q01", {"C01"})


def test_reference_to_unavailable_claim_in_explanation():
    with pytest.raises(InvalidModelAnswer, match="Объяснение"):
        parse_model_answer(payload(explanation="Это следует из C99."), "Q01", {"C01"})


def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("src.llm.ENV_FILE", tmp_path / "missing.env")
    llm = DeepSeekLLM()
    with pytest.raises(LLMUnavailable, match="DEEPSEEK_API_KEY"):
        llm.ask("Q01", "Вопрос", BuiltContext(mode="B", question_id="Q01", text="C01: факт", claim_ids=["C01"]))


def test_key_is_loaded_from_local_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nDEEPSEEK_API_KEY='test-key'\n", encoding="utf-8")
    monkeypatch.setattr("src.llm.ENV_FILE", env_file)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert get_api_key() == "test-key"
    monkeypatch.setenv("DEEPSEEK_API_KEY", "process-key")
    assert get_api_key() == "process-key"
    assert get_api_key("explicit-key") == "explicit-key"


def test_adapter_parses_response_from_fake_client():
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=__import__("json").dumps(payload())))],
                               usage=SimpleNamespace(prompt_tokens=15, completion_tokens=20), id="chatcmpl-1")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    result = DeepSeekLLM(client=client).ask("Q01", "Вопрос", BuiltContext(mode="B", question_id="Q01", text="C01: факт", claim_ids=["C01"]))
    assert result.answer.evidence_ids == ["C01"]
    assert (result.input_tokens, result.output_tokens) == (15, 20)


def test_api_failure_is_reported():
    def fail(**kwargs):
        raise ConnectionError("offline")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(LLMUnavailable, match="offline"):
        DeepSeekLLM(client=client).ask("Q01", "Вопрос", BuiltContext(mode="B", question_id="Q01", text="C01: факт", claim_ids=["C01"]))


def test_invalid_raw_output_is_available_for_review():
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="not json"))], usage=None, id="chatcmpl-1")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    with pytest.raises(InvalidModelAnswer) as error:
        DeepSeekLLM(client=client).ask("Q01", "Вопрос", BuiltContext(mode="B", question_id="Q01", text="C01: факт", claim_ids=["C01"]))
    assert error.value.raw_output == "not json"

"""DeepSeek Chat Completions adapter and strict answer validation."""

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import BuiltContext, ModelAnswer

DEFAULT_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "system.txt"
DEFAULT_MODEL = "deepseek-flash"

class LLMUnavailable(RuntimeError):
    pass


class InvalidModelAnswer(ValueError):
    def __init__(self, message: str, raw_output: str | None = None):
        super().__init__(message)
        self.raw_output = raw_output


@dataclass(frozen=True)
class LLMResult:
    answer: ModelAnswer
    input_tokens: int | None
    output_tokens: int | None
    elapsed_seconds: float
    response_id: str | None
    raw_output: str


def parse_model_answer(raw: str | dict[str, Any], question_id: str, available_ids: set[str]) -> ModelAnswer:
    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
        answer = ModelAnswer.model_validate(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise InvalidModelAnswer(f"Некорректный JSON или схема ответа: {exc}") from exc
    if answer.question_id != question_id:
        raise InvalidModelAnswer("Ответ относится к другому вопросу")
    cited = answer.evidence_ids + answer.counterevidence_ids
    if len(cited) != len(set(cited)):
        raise InvalidModelAnswer("Повторяющиеся ID доказательств")
    unknown = set(cited) - available_ids
    if unknown:
        raise InvalidModelAnswer(f"Несуществующие или недоступные claim_id: {sorted(unknown)}")
    explanation_ids = set(re.findall(r"\bC[A-Z]?\d+\b", answer.explanation))
    if explanation_ids - available_ids:
        raise InvalidModelAnswer("Объяснение ссылается на недоступные claim_id")
    if not answer.explanation.strip():
        raise InvalidModelAnswer("Пустое объяснение")
    if answer.status == "both" and (not answer.evidence_ids or not answer.counterevidence_ids):
        raise InvalidModelAnswer("Для both нужны доказательства с обеих сторон")
    if answer.status == "insufficient" and (answer.answer is not None or cited):
        raise InvalidModelAnswer("При insufficient нельзя утверждать ответ или ссылаться на решающие доказательства")
    if answer.status == "entailed" and not answer.evidence_ids:
        raise InvalidModelAnswer("Для entailed нужны evidence_ids")
    if answer.status == "contradicted" and not answer.counterevidence_ids:
        raise InvalidModelAnswer("Для contradicted нужны counterevidence_ids")
    return answer


class DeepSeekLLM:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None, prompt_path: str | Path = DEFAULT_PROMPT, client: Any = None):
        self.model = model
        self.system_prompt = Path(prompt_path).read_text(encoding="utf-8")
        self._api_key = api_key
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        key = self._api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise LLMUnavailable("Задайте DEEPSEEK_API_KEY в окружении или используйте --ask-key")
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=120.0, max_retries=2)
        except ImportError as exc:
            raise LLMUnavailable("Установите зависимости из requirements.txt") from exc
        return self._client

    def ask(self, question_id: str, question: str, context: BuiltContext) -> LLMResult:
        input_text = f"Вопрос {question_id}: {question}\n\nКонтекст режима {context.mode}:\n{context.text}"
        start = time.perf_counter()
        for attempt in range(4):
            try:
                response = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": input_text},
                    ],
                    response_format={"type": "json_object"},
                    temperature=1.0,
                    extra_body={"thinking": {"type": "disabled"}},
                    max_tokens=1200,
                )
                break
            except LLMUnavailable:
                raise
            except Exception as exc:
                if getattr(exc, "status_code", None) == 429 and attempt < 3:
                    header = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
                    try:
                        delay = float(header) if header else min(15 * (attempt + 1), 60)
                    except ValueError:
                        delay = 15
                    time.sleep(min(max(delay, 1), 60))
                    continue
                detail = str(exc)
                key = self._api_key or os.environ.get("DEEPSEEK_API_KEY")
                if key:
                    detail = detail.replace(key, "[REDACTED]")
                raise LLMUnavailable(f"LLM недоступна: {type(exc).__name__}: {detail}") from None
        elapsed = time.perf_counter() - start
        choice = response.choices[0] if getattr(response, "choices", None) else None
        if choice is None or getattr(choice, "finish_reason", None) != "stop":
            raise InvalidModelAnswer(f"LLM вернула незавершённый ответ: {getattr(choice, 'finish_reason', None)}")
        raw = getattr(choice.message, "content", None)
        if not raw:
            raise InvalidModelAnswer("В ответе нет текста")
        try:
            answer = parse_model_answer(raw, question_id, set(context.claim_ids))
        except InvalidModelAnswer as exc:
            exc.raw_output = raw
            raise
        usage = getattr(response, "usage", None)
        return LLMResult(
            answer=answer,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            elapsed_seconds=elapsed,
            response_id=getattr(response, "id", None),
            raw_output=raw,
        )

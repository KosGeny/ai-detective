"""Recheck saved raw LLM responses without making new model requests."""

import argparse
import gzip
import json
from pathlib import Path

from src.experiment import load_results
from src.llm import InvalidModelAnswer, parse_model_answer


def revalidate(rows: list[dict]) -> list[dict]:
    updated = []
    for original in rows:
        row = dict(original)
        row["original_error"] = original.get("error")
        row["original_error_type"] = original.get("error_type")
        row["revalidated_from_raw"] = False
        row["normalized_fields"] = False
        if original.get("raw_output"):
            try:
                answer = parse_model_answer(
                    original["raw_output"], original["question_id"],
                    set(original["context_claim_ids"]),
                )
            except InvalidModelAnswer:
                pass
            else:
                if original.get("answer") != answer.model_dump():
                    row["original_answer"] = original.get("answer")
                    row["normalized_fields"] = True
                row["answer"] = answer.model_dump()
                row["error"] = None
                row["error_type"] = None
                row["revalidated_from_raw"] = original.get("answer") is None
        updated.append(row)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/deepseek_flash.jsonl.gz")
    parser.add_argument("--output", default="results/deepseek_flash_revalidated.jsonl.gz")
    args = parser.parse_args()
    source = Path(args.input)
    target = Path(args.output)
    if source.resolve() == target.resolve():
        parser.error("Исходный архив нельзя перезаписывать")
    rows = revalidate(load_results(source))
    target.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} ответов, восстановлено из raw: {sum(r['revalidated_from_raw'] for r in rows)}; нормализованы поля: {sum(r['normalized_fields'] for r in rows)}")


if __name__ == "__main__":
    main()

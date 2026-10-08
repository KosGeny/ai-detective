import argparse
import getpass
import json

from src.evaluation import compare_modes, compare_stress, summarize
from src.experiment import load_run, run_experiment
from src.llm import DeepSeekLLM


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон лабораторной работы №6")
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--ask-key", action="store_true", help="Скрытый ввод ключа без записи в окружение")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=("A", "B", "C"), default=["A", "B", "C"])
    parser.add_argument("--run-id", help="Продолжить существующий прогон теми же настройками")
    parser.add_argument("--output", default="results/runs.jsonl")
    args = parser.parse_args()
    key = getpass.getpass("DeepSeek API key: ") if args.ask_key else None
    rows = run_experiment(output=args.output, include_stress=not args.baseline_only, llm=DeepSeekLLM(model=args.model, api_key=key),
                          modes=tuple(args.modes), run_id=args.run_id,
                          progress=lambda n, total: print(f"{n}/{total}", end="\r", flush=True))
    print()
    run_id = args.run_id or rows[0]["run_id"]
    print(f"run_id={run_id}")
    rows = load_run(args.output, run_id)
    for mode in ("A", "B", "C"):
        subset = [row for row in rows if row["mode"] == mode and row["case_id"] == "baseline"]
        print(mode, json.dumps(summarize(subset), ensure_ascii=False))
    print("stress", json.dumps(compare_stress(rows), ensure_ascii=False))
    print("B/C", json.dumps(compare_modes(rows), ensure_ascii=False))
    invalid = sum(row["error_type"] == "InvalidModelAnswer" for row in rows)
    if invalid:
        print(f"Невалидных ответов модели: {invalid}; исходный JSON сохранён")
    if any(row["error_type"] == "LLMUnavailable" for row in rows):
        raise SystemExit("Есть незавершённые запросы API; повторите с тем же --run-id")


if __name__ == "__main__":
    main()

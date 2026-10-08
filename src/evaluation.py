"""Metrics for saved model responses; failed calls are reported separately."""

from collections import defaultdict
from statistics import mean
from typing import Any, Iterable


def _prf(predicted: set[str], gold: set[str]) -> tuple[int, int, int]:
    return len(predicted & gold), len(predicted - gold), len(gold - predicted)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    valid = [row for row in rows if row.get("answer") is not None]
    tp = fp = fn = counter_tp = counter_fn = 0
    for row in valid:
        answer = row["answer"]
        a, b, c = _prf(set(answer["evidence_ids"]), set(row["gold_evidence_ids"]))
        tp += a; fp += b; fn += c
        a, _, c = _prf(set(answer["counterevidence_ids"]), set(row["gold_counterevidence_ids"]))
        counter_tp += a; counter_fn += c
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    both = [row for row in valid if row["expected_status"] == "both"]
    insufficient = [row for row in valid if row["expected_status"] == "insufficient"]
    predicted_both = [row for row in valid if row["answer"]["status"] == "both"]
    predicted_insufficient = [row for row in valid if row["answer"]["status"] == "insufficient"]
    uncertain = [row for row in valid if row["expected_status"] in ("both", "insufficient")]
    confident_wrong = [row for row in uncertain if row["answer"]["status"] in ("entailed", "contradicted") and row["answer"]["confidence"] >= 0.8]
    def avg(name: str) -> float | None:
        numbers = [row[name] for row in valid if row.get(name) is not None]
        return mean(numbers) if numbers else None
    return {
        "total": len(rows), "valid": len(valid), "failed": len(rows) - len(valid),
        "status_accuracy": _ratio(sum(row["answer"]["status"] == row["expected_status"] for row in valid), len(valid)),
        "status_accuracy_all": _ratio(sum(row["answer"]["status"] == row["expected_status"] for row in valid), len(rows)),
        "evidence_precision": precision, "evidence_recall": recall, "evidence_f1": f1,
        "counterevidence_recall": _ratio(counter_tp, counter_tp + counter_fn),
        "both_recall": _ratio(sum(row["answer"]["status"] == "both" for row in both), len(both)),
        "both_precision": _ratio(sum(row["expected_status"] == "both" for row in predicted_both), len(predicted_both)),
        "insufficient_recall": _ratio(sum(row["answer"]["status"] == "insufficient" for row in insufficient), len(insufficient)),
        "insufficient_precision": _ratio(sum(row["expected_status"] == "insufficient" for row in predicted_insufficient), len(predicted_insufficient)),
        "confident_guess_rate_on_insufficient": _ratio(sum(row["answer"]["status"] in ("entailed", "contradicted") and row["answer"]["confidence"] >= 0.8 for row in insufficient), len(insufficient)),
        "confident_unsupported_rate": _ratio(len(confident_wrong), len(uncertain)),
        "mean_confidence": mean(row["answer"]["confidence"] for row in valid) if valid else None,
        "mean_input_tokens": avg("input_tokens"), "mean_output_tokens": avg("output_tokens"),
        "mean_elapsed_seconds": avg("elapsed_seconds"),
    }


def compare_stress(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in rows if row.get("answer") is not None]
    baseline = {(row["question_id"], row["mode"]): row for row in rows if row["case_id"] == "baseline"}
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["case_id"] == "baseline":
            continue
        original = baseline.get((row["question_id"], row["mode"]))
        if original is None:
            continue
        answer = row["answer"]
        previous = original["answer"]
        by_kind[row["kind"]].append({
            "mode": row["mode"],
            "status_changed": answer["status"] != previous["status"],
            "expected_reaction": answer["status"] == row["expected_status"],
            "evidence_stable": set(answer["evidence_ids"]) == set(previous["evidence_ids"]),
        })
    out: dict[str, Any] = {}
    for kind, items in by_kind.items():
        out[kind] = {
            "cases": len(items),
            "status_changed_rate": mean(item["status_changed"] for item in items),
            "expected_reaction_rate": mean(item["expected_reaction"] for item in items),
            "evidence_stability_rate": mean(item["evidence_stable"] for item in items),
        }
    return out


def compare_modes(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("answer") is not None:
            groups[(row["question_id"], row["case_id"])][row["mode"]] = row
    pairs = [group for group in groups.values() if "B" in group and "C" in group]
    return {
        "pairs": len(pairs),
        "flat_padic_status_difference_rate": _ratio(sum(group["B"]["answer"]["status"] != group["C"]["answer"]["status"] for group in pairs), len(pairs)),
        "same_retrieved_ids": all(set(group["B"]["context_claim_ids"]) == set(group["C"]["context_claim_ids"]) for group in pairs),
    }

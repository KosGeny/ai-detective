"""Generate the final PDF only from a complete, successful real LLM run."""

import argparse
from collections import defaultdict
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

from src.evaluation import compare_modes, compare_stress, summarize
from src.experiment import load_run


def font() -> str:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    for path in candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont("ReportUnicode", str(path)))
            return "ReportUnicode"
    raise RuntimeError("Нужен Unicode TTF-шрифт (например DejaVu Sans)")


def fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def build_report(rows: list[dict], output: Path) -> None:
    coverage = {(row["question_id"], row["case_id"], row["mode"]) for row in rows}
    by_mode = {
        mode: [row for row in rows if row["mode"] == mode] for mode in ("A", "B", "C")
    }
    if (len(rows) != 120 or len(coverage) != 120
            or any(len(by_mode[mode]) != 40 or sum(row["case_id"] == "baseline" for row in by_mode[mode]) != 12
                   for mode in ("A", "B", "C"))
            or any(row.get("error_type") == "LLMUnavailable" for row in rows)):
        raise ValueError("Для отчёта нужны 120 полученных ответов без ошибок доступа к LLM")
    if (len({row["model"] for row in rows}) != 1
            or rows[0]["model"].startswith(("fake", "synthetic"))
            or any(not str(row.get("response_id", "")).startswith("chatcmpl-") and not row.get("raw_output") for row in rows)):
        raise ValueError("Отчёт требует одну реальную LLM")
    base = {mode: summarize(row for row in rows if row["mode"] == mode and row["case_id"] == "baseline") for mode in ("A", "B", "C")}
    stress = compare_stress(rows)
    modes = compare_modes(rows)
    face = font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", fontName=face, fontSize=16, leading=20, spaceAfter=10, alignment=TA_CENTER))
    styles.add(ParagraphStyle(name="ReportBody", fontName=face, fontSize=9, leading=13, spaceAfter=7))
    styles.add(ParagraphStyle(name="ReportHead", fontName=face, fontSize=11, leading=14, spaceBefore=8, spaceAfter=5))
    P = lambda text: Paragraph(text, styles["ReportBody"])
    H = lambda text: Paragraph(text, styles["ReportHead"])
    story = [Paragraph("AI-детектив: результаты лабораторной работы 6", styles["ReportTitle"])]
    story.append(P(f"Модель: {rows[0]['model']}. В полном прогоне: 12 вопросов × 3 режима и 28 стресс-кейсов × 3 режима. "
                   "Во всех режимах использован один системный промпт и одинаковые параметры. "
                   "B и C получали одинаковые ID утверждений; в C менялась только их организация."))
    story.append(H("Основные метрики"))
    table = Table([
        ["Режим", "Статус", "Evidence F1", "Both recall", "Insuff. recall", "Токены ввода"],
        *[[mode, fmt(base[mode]["status_accuracy_all"]), fmt(base[mode]["evidence_f1"]),
           fmt(base[mode]["both_recall"]), fmt(base[mode]["insufficient_recall"]),
           fmt(base[mode]["mean_input_tokens"])] for mode in ("A", "B", "C")],
    ], colWidths=[22*mm, 22*mm, 30*mm, 29*mm, 32*mm, 35*mm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), face), ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CDD5DF")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 4*mm)])
    top_accuracy = max(base[mode]["status_accuracy_all"] or 0 for mode in ("A", "B", "C"))
    leaders = ", ".join(mode for mode in ("A", "B", "C") if base[mode]["status_accuracy_all"] == top_accuracy)
    story.append(P(f"На этих 12 вопросах максимальная точность статуса у режима(ов) {leaders} ({fmt(top_accuracy)}). "
                   f"Retrieval B относительно полного текста A: {fmt(base['B']['status_accuracy_all'])} против {fmt(base['A']['status_accuracy_all'])}. "
                   f"p-adic C относительно B: {fmt(base['C']['status_accuracy_all'])} против {fmt(base['B']['status_accuracy_all'])}. "
                   "Разница описательная; выборка мала и не доказывает общего преимущества метода."))
    story.append(H("Противоречия, неизвестность и уверенность"))
    story.append(P(f"Recall статуса both: A {fmt(base['A']['both_recall'])}, B {fmt(base['B']['both_recall'])}, C {fmt(base['C']['both_recall'])}. "
                   f"Recall insufficient: A {fmt(base['A']['insufficient_recall'])}, B {fmt(base['B']['insufficient_recall'])}, C {fmt(base['C']['insufficient_recall'])}. "
                   f"Доля уверенных однозначных ответов там, где эталон insufficient: A {fmt(base['A']['confident_guess_rate_on_insufficient'])}, "
                   f"B {fmt(base['B']['confident_guess_rate_on_insufficient'])}, C {fmt(base['C']['confident_guess_rate_on_insufficient'])}. "
                   f"Средняя confidence и фактическая точность: A {fmt(base['A']['mean_confidence'])}/{fmt(base['A']['status_accuracy'])}, "
                   f"B {fmt(base['B']['mean_confidence'])}/{fmt(base['B']['status_accuracy'])}, "
                   f"C {fmt(base['C']['mean_confidence'])}/{fmt(base['C']['status_accuracy'])}."))
    story.append(P("Невалидных по логическим правилам ответов: " + "; ".join(
        f"{mode}: {base[mode]['failed']} из {base[mode]['total']}" for mode in ("A", "B", "C")
    ) + ". В точности статуса по всем вопросам они считаются ошибками; остальные метрики вычислены по валидным ответам."))
    story.append(P("Evidence precision/recall и recall контрдоказательств: " + "; ".join(
        f"{mode}: {fmt(base[mode]['evidence_precision'])}/{fmt(base[mode]['evidence_recall'])}, "
        f"counter {fmt(base[mode]['counterevidence_recall'])}" for mode in ("A", "B", "C")
    ) + ". Среднее время ответа и токены вывода: " + "; ".join(
        f"{mode}: {fmt(base[mode]['mean_elapsed_seconds'])} с, {fmt(base[mode]['mean_output_tokens'])}" for mode in ("A", "B", "C")
    ) + "."))
    story.append(H("Стресс-тесты"))
    for kind, label in (("shuffle", "Перестановка документов"), ("delete", "Удаление посылки"), ("contradiction", "Добавление противоречия")):
        item = stress.get(kind, {})
        story.append(P(f"{label}: {item.get('cases', 0)} валидных пар ответов; изменился статус в доле {fmt(item.get('status_changed_rate'))}, "
                       f"ожидаемая реакция получена в доле {fmt(item.get('expected_reaction_rate'))}, "
                       f"доказательства сохранились в доле {fmt(item.get('evidence_stability_rate'))}."))
    story.append(P(f"Между flat и p-adic статус различается в доле {fmt(modes['flat_padic_status_difference_rate'])} пар; "
                   f"одинаковый набор ID в B/C: {'да' if modes['same_retrieved_ids'] else 'нет'}. "
                   "Тип источника поставлен первым уровнем p-adic пути, чтобы сначала разделить системные записи, свидетельства и правила; "
                   "при ином порядке уровней соседство утверждений и форма контекста изменились бы."))
    baseline = defaultdict(dict)
    for row in rows:
        if row["case_id"] == "baseline":
            baseline[row["question_id"]][row["mode"]] = row
    helped = [qid for qid, group in baseline.items() if group["C"]["answer"] and group["C"]["answer"]["status"] == group["C"]["expected_status"] and (not group["B"]["answer"] or group["B"]["answer"]["status"] != group["B"]["expected_status"])]
    harmed = [qid for qid, group in baseline.items() if group["B"]["answer"] and group["B"]["answer"]["status"] == group["B"]["expected_status"] and (not group["C"]["answer"] or group["C"]["answer"]["status"] != group["C"]["expected_status"])]
    story.append(P("Примеры, где C помог относительно B: " + (", ".join(helped) if helped else "не выявлены") +
                   "; где C ухудшил статус: " + (", ".join(harmed) if harmed else "не выявлены") + "."))
    story.append(H("Ограничения и вклад"))
    story.append(P("Эксперимент проведён на одном вымышленном досье и одной модели. Метрики относятся к этим 12 вопросам и подготовленным изменениям; "
                   "они не измеряют причинный эффект группировки за пределами данного набора. "
                   "Дедкова: досье и эталоны; Генне: retrieval, p-adic, контекст и стресс-подготовка; "
                   "Маракулин: LLM, интерфейс, метрики и интеграционные тесты."))
    output.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(output), pagesize=(210*mm, 297*mm), rightMargin=20*mm, leftMargin=20*mm,
                      topMargin=18*mm, bottomMargin=18*mm).build(story)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/runs.jsonl")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default="report.pdf")
    args = parser.parse_args()
    rows = load_run(args.results, args.run_id)
    try:
        build_report(rows, Path(args.output))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(args.output)


if __name__ == "__main__":
    main()

"""Interactive stand for laboratory work 6."""

import json
import tempfile
import uuid
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_io import load_dossier
from src.evaluation import compare_modes, compare_stress, summarize
from src.experiment import load_results, load_run, make_contexts, run_experiment, run_one
from src.llm import DeepSeekLLM

RESULTS = Path("results/runs.jsonl")
CURRENT_ARCHIVE = Path("results/deepseek_flash_current_20261017.jsonl.gz")
FINAL_ARCHIVE = Path("results/deepseek_flash_final.jsonl.gz")
LEGACY_ARCHIVE = Path("results/deepseek_flash_revalidated.jsonl.gz")
REQUIRED_UPLOADS = {"documents.jsonl", "claims.jsonl", "questions.jsonl", "stress_cases.jsonl"}

st.set_page_config(page_title="AI-детектив", layout="wide")
st.title("AI-детектив · лабораторная работа 6")
st.caption("Одна модель, один промпт; A — все документы, B — BM25 top-15, C — те же ID по p-adic ветвям")

st.sidebar.header("Досье и модель")
uploaded = st.sidebar.file_uploader("Загрузить четыре JSONL файла", type="jsonl", accept_multiple_files=True)
model = st.sidebar.text_input("Модель", value="deepseek-flash")
api_key = st.sidebar.text_input("DeepSeek API key", type="password", help="Можно оставить пустым: приложение прочитает ключ из локального .env")
data_dir = Path("data")
if uploaded:
    names = {item.name for item in uploaded}
    if names != REQUIRED_UPLOADS:
        st.sidebar.error("Нужны ровно documents.jsonl, claims.jsonl, questions.jsonl, stress_cases.jsonl")
        st.stop()
    if "upload_dir" not in st.session_state:
        st.session_state["upload_dir"] = Path(tempfile.mkdtemp(prefix="ai-detective-"))
    upload_dir = st.session_state["upload_dir"]
    for item in uploaded:
        (upload_dir / item.name).write_bytes(item.getvalue())
    data_dir = upload_dir

try:
    documents, claims, questions, cases = load_dossier(data_dir)
except Exception as exc:
    st.error(f"Ошибка данных: {exc}")
    st.stop()

question = st.selectbox("Вопрос", questions, format_func=lambda item: f"{item.question_id}: {item.question}")
matching_cases = [case for case in cases if case.question_id == question.question_id]
case_choice = st.selectbox("Условие", [None, *matching_cases], format_func=lambda item: "Исходное досье" if item is None else f"{item.kind}: {item.case_id}")
mode = st.radio("Режим", ("A", "B", "C"), horizontal=True)
try:
    contexts = dict(make_contexts(question, documents, claims, case_choice))
except Exception as exc:
    st.error(f"Ошибка построения контекста: {exc}")
    st.stop()
context = contexts[mode]
context_hash = sha256(context.text.encode("utf-8")).hexdigest()

left, right = st.columns(2)
with left:
    st.subheader("Контекст")
    st.write(f"Документы: {', '.join(context.source_document_ids)}")
    st.write(f"Доступные ID: {', '.join(context.claim_ids)}")
    with st.expander("Текст контекста", expanded=False):
        st.text(context.text)
    if mode == "C":
        st.subheader("p-adic ветви")
        for path, ids in context.branches.items():
            st.write(f"{path}: {', '.join(ids)}")

with right:
    st.subheader("Ответ")
    if st.button("Запросить ответ модели", type="primary"):
        llm = DeepSeekLLM(model=model, api_key=api_key or None)
        row = run_one(llm, question, mode, context, case_choice, run_id=uuid.uuid4().hex,
                      original_document_ids=[document.document_id for document in documents],
                      original_claim_ids=[claim.claim_id for claim in claims])
        RESULTS.parent.mkdir(exist_ok=True)
        with RESULTS.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        if row["error"]:
            st.error(row["error"])
        else:
            st.session_state["last_answer"] = row
            st.session_state["last_context_hash"] = context_hash
            st.session_state["last_model"] = model
    last = st.session_state.get("last_answer")
    if last and st.session_state.get("last_context_hash") == context_hash and st.session_state.get("last_model") == model and last["question_id"] == question.question_id and last["mode"] == mode and last["case_id"] == (case_choice.case_id if case_choice else "baseline"):
        answer = last["answer"]
        visible_claims = {claim.claim_id: claim for claim in claims}
        if case_choice and case_choice.added_claim:
            visible_claims[case_choice.added_claim.claim_id] = case_choice.added_claim
        st.metric("Статус", answer["status"])
        st.metric("Confidence", f"{answer['confidence']:.2f}")
        st.write(answer["answer"] or "Однозначного ответа нет")
        st.write("Доказательства:")
        for claim_id in answer["evidence_ids"]:
            st.write(f"{claim_id}: {visible_claims[claim_id].text}")
        st.write("Контрдоказательства:")
        for claim_id in answer["counterevidence_ids"]:
            st.write(f"{claim_id}: {visible_claims[claim_id].text}")
        st.write(answer["explanation"])
        st.caption(f"Токены: {last['input_tokens']} + {last['output_tokens']}; время: {last['elapsed_seconds']:.2f} с")

st.divider()
st.subheader("Полный эксперимент")
st.write(f"{len(questions) * 3} исходных ответов и {len(cases) * 3} стресс-ответов. Каждый ответ сохраняется сразу после получения.")
include_stress = st.checkbox("Включить стресс-кейсы", value=True)
selected_modes = st.multiselect("Режимы для прогона", ("A", "B", "C"), default=("A", "B", "C"))
if st.button("Запустить полный прогон"):
    if not selected_modes:
        st.error("Выберите хотя бы один режим")
        st.stop()
    bar = st.progress(0.0)
    try:
        with st.spinner("Выполняется прогон"):
            rows = run_experiment(data_dir=data_dir, output=RESULTS, include_stress=include_stress,
                                  llm=DeepSeekLLM(model=model, api_key=api_key or None), modes=tuple(selected_modes),
                                  progress=lambda n, total: bar.progress(n / total))
        if any(row["error"] for row in rows):
            st.error(f"Прогон остановлен после {len(rows)} записей; run_id={rows[0]['run_id']}. {rows[-1]['error']}")
        else:
            st.success(f"Сохранено {len(rows)} ответов; run_id={rows[0]['run_id']}")
    except Exception as exc:
        st.error(str(exc))

available_results = {}
if CURRENT_ARCHIVE.exists():
    available_results["Итоговый прогон 17 октября: текущие данные"] = CURRENT_ARCHIVE
if FINAL_ARCHIVE.exists():
    available_results["Прогон 9 октября: прежняя редакция данных"] = FINAL_ARCHIVE
if LEGACY_ARCHIVE.exists():
    available_results["Предыдущий прогон после повторной проверки"] = LEGACY_ARCHIVE
if RESULTS.exists():
    available_results["Локальные прогоны"] = RESULTS
saved_path = available_results[st.sidebar.selectbox("Источник результатов", list(available_results))] if available_results else RESULTS
saved = load_results(saved_path)
if saved:
    st.divider()
    st.subheader("Сохранённые результаты")
    if saved_path == CURRENT_ARCHIVE:
        st.caption("Итоговый прогон текущего досье: 38 утверждений, 36 стресс-кейсов, 144 ответа; шесть ответов не прошли логическую проверку формата.")
    elif saved_path == FINAL_ARCHIVE:
        st.caption("Показан полный прогон 9 октября с 37 утверждениями и 28 стресс-кейсами. Текущие данные из CLOUD_DOCS отличаются и требуют отдельного прогона.")
    elif saved_path == LEGACY_ARCHIVE:
        st.caption("Показан предыдущий прогон после повторной проверки сохранённых ответов.")
    run_ids = list(dict.fromkeys(row["run_id"] for row in reversed(saved)))
    selected = st.selectbox("Прогон", run_ids)
    selected_rows = load_run(saved_path, selected)
    overview = []
    for selected_mode in ("A", "B", "C"):
        baseline = [row for row in selected_rows if row["mode"] == selected_mode and row["case_id"] == "baseline"]
        overview.append({"mode": selected_mode, **summarize(baseline)})
    st.dataframe(pd.DataFrame(overview), width="stretch")
    st.write("Устойчивость", compare_stress(selected_rows))
    st.write("B против C", compare_modes(selected_rows))
    st.dataframe(pd.DataFrame([{
        "question_id": row["question_id"], "case_id": row["case_id"], "mode": row["mode"],
        "expected": row["expected_status"], "status": row["answer"]["status"] if row["answer"] else "ERROR",
        "evidence_ids": ", ".join(row["answer"]["evidence_ids"]) if row["answer"] else "",
        "tokens": (row["input_tokens"] or 0) + (row["output_tokens"] or 0),
        "error": row["error"],
    } for row in selected_rows]), width="stretch")

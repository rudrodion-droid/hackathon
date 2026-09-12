"""
app.py

Streamlit-приложение для автоматического обнаружения дефектов бетонных
конструкций (трещины, сколы, высолы, раковины) по фотографии, с
использованием Roboflow Inference API.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st
from PIL import Image

from detector import (
    RoboflowAuthError,
    RoboflowDetector,
    RoboflowDetectorError,
    RoboflowEmptyResponseError,
    RoboflowNetworkError,
)
from report import generate_pdf_report

# ---------------------------------------------------------------------------
# API-ключ Roboflow.
#
# Ключ зашит в код по умолчанию, чтобы приложение работало "из коробки" без
# необходимости вводить что-либо в интерфейсе — он нигде не отображается в UI.
# Переменная окружения ROBOFLOW_API_KEY (если задана на сервере/в Railway) имеет
# приоритет над значением по умолчанию — это позволяет заменить ключ при
# деплое, не редактируя код и не публикуя реальный ключ в открытом репозитории.
# ---------------------------------------------------------------------------
_DEFAULT_API_KEY = "Z1QEDPvGPkNLNYEDRRN6"
API_KEY = os.environ.get("ROBOFLOW_API_KEY", _DEFAULT_API_KEY)

st.set_page_config(
    page_title="Детектор дефектов бетонных конструкций",
    page_icon="🏗️",
    layout="wide",
)

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# ---------------------------------------------------------------------------
# Оформление
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .block-container {
            padding-top: 1.6rem;
            max-width: 1180px;
        }

        /* --- Hero-заголовок ------------------------------------------- */
        .hero {
            background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 55%, #3B82F6 100%);
            border-radius: 18px;
            padding: 2.1rem 2.4rem;
            margin-bottom: 1.6rem;
            box-shadow: 0 10px 30px -12px rgba(37, 99, 235, 0.45);
        }
        .hero h1 {
            color: #FFFFFF;
            font-weight: 800;
            font-size: 2rem;
            margin: 0 0 0.5rem 0;
        }
        .hero p {
            color: #DBEAFE;
            font-size: 1.02rem;
            margin: 0;
            max-width: 720px;
        }

        /* --- Карточки метрик -------------------------------------------- */
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 0.9rem 1.1rem 0.7rem 1.1rem;
            box-shadow: 0 4px 14px -8px rgba(15, 23, 42, 0.12);
        }
        div[data-testid="stMetricLabel"] {
            color: #64748B;
        }
        div[data-testid="stMetricValue"] {
            color: #1E3A8A;
            font-weight: 700;
        }

        /* --- Кнопки ------------------------------------------------------ */
        .stButton > button, .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 600;
            border: none;
        }
        .stButton > button[kind="primary"], .stButton > button {
            background: linear-gradient(120deg, #2563EB, #1E3A8A);
            color: #FFFFFF;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            filter: brightness(1.08);
        }

        /* --- Секции результатов ------------------------------------------ */
        .section-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #1E3A8A;
            margin: 1.6rem 0 0.6rem 0;
            padding-bottom: 0.35rem;
            border-bottom: 2px solid #DBEAFE;
        }

        /* --- Боковая панель ----------------------------------------------- */
        section[data-testid="stSidebar"] {
            background: #F8FAFC;
        }
        section[data-testid="stSidebar"] h2 {
            color: #1E3A8A;
        }

        div[data-testid="stExpander"] {
            border: 1px solid #E2E8F0;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Боковая панель
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Настройки")

    model_id = st.text_input(
        "ID модели Roboflow",
        value="concrete-crack-detection/1",
        help="Формат: project-slug/version. Можно заменить на свою обученную модель.",
    )

    confidence_threshold = st.slider(
        "Confidence Threshold (порог уверенности)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=(
            "Дефекты с уверенностью модели ниже этого значения не будут "
            "учитываться. Передаётся напрямую в Roboflow API — как одноимённый "
            "ползунок в веб-интерфейсе Roboflow."
        ),
    )

    overlap_threshold = st.slider(
        "Overlap Threshold (порог перекрытия)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=(
            "IoU-порог для объединения (NMS) перекрывающихся рамок одного и "
            "того же дефекта. Чем меньше значение, тем агрессивнее "
            "схлопываются дублирующиеся рамки. Передаётся напрямую в "
            "Roboflow API — как одноимённый ползунок в веб-интерфейсе Roboflow."
        ),
    )

    opacity_threshold = st.slider(
        "Opacity Threshold (непрозрачность заливки)",
        min_value=0.0,
        max_value=1.0,
        value=0.75,
        step=0.05,
        help=(
            "Непрозрачность цветной заливки внутри рамок на итоговом "
            "изображении. Влияет только на отображение результатов, не на "
            "сам поиск дефектов."
        ),
    )

    st.markdown("---")
    st.caption(
        "🔒 Подключение к Roboflow уже настроено — вводить API-ключ не нужно. "
        "Готовую модель детекции дефектов можно найти в "
        "[Roboflow Universe](https://universe.roboflow.com), либо указать свою "
        "выше."
    )

# ---------------------------------------------------------------------------
# Основная область — hero-заголовок и загрузка файла
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏗️ Автоматическое обнаружение дефектов конструкций</h1>
        <p>Загрузите фотографию бетонной конструкции — сервис автоматически
        найдёт трещины, сколы, высолы и раковины, используя предобученную
        ИИ-модель на базе Roboflow Inference API.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Загрузите фото конструкции",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Загруженное изображение", width=400)

run_disabled = uploaded_file is None
run_button = st.button(
    "🔍 Запустить анализ",
    disabled=run_disabled,
    use_container_width=True,
    type="primary",
)

if run_disabled:
    st.info("Загрузите фотографию конструкции, чтобы начать анализ.")

# ---------------------------------------------------------------------------
# Запуск анализа
# ---------------------------------------------------------------------------
if run_button and uploaded_file is not None:
    with st.spinner("Анализируем изображение, пожалуйста подождите..."):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_path = str(Path(tmp_dir) / uploaded_file.name)
                output_path = str(Path(tmp_dir) / f"annotated_{uploaded_file.name}")

                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                detector = RoboflowDetector(api_key=API_KEY, model_id=model_id)
                results: Dict[str, Any] = detector.detect(
                    image_path=input_path,
                    confidence=confidence_threshold,
                    overlap=overlap_threshold,
                )
                annotated_image = detector.visualize(
                    image_path=input_path,
                    results=results,
                    output_path=output_path,
                    opacity=opacity_threshold,
                )

                # Изображения загружаются в память как объекты Pillow, а не как
                # пути к файлам: временная папка удаляется сразу по выходу из
                # блока "with", поэтому пути на диск станут недействительны ещё
                # до отображения результатов ниже.
                original_image = Image.open(input_path).convert("RGB")
                original_image.load()

                st.session_state["results"] = results
                st.session_state["original_image"] = original_image
                st.session_state["annotated_image"] = annotated_image.copy()
                st.session_state["analysis_done"] = True
                # Снимок параметров, использованных именно в этом запуске —
                # нужен для PDF-отчёта, даже если пользователь потом подвинет
                # ползунки в сайдбаре.
                st.session_state["run_params"] = {
                    "model_id": model_id,
                    "confidence_threshold": confidence_threshold,
                    "overlap_threshold": overlap_threshold,
                    "opacity_threshold": opacity_threshold,
                    "source_filename": uploaded_file.name,
                }

            st.success("✅ Анализ успешно завершён.")

        except RoboflowAuthError as exc:
            st.session_state["analysis_done"] = False
            st.error(f"❌ Ошибка авторизации: {exc}")
        except RoboflowNetworkError as exc:
            st.session_state["analysis_done"] = False
            st.error(f"❌ Ошибка сети: {exc}")
        except RoboflowEmptyResponseError as exc:
            st.session_state["analysis_done"] = False
            st.warning(f"⚠️ {exc}")
        except RoboflowDetectorError as exc:
            st.session_state["analysis_done"] = False
            st.error(f"❌ Ошибка детектора: {exc}")
        except Exception as exc:  # noqa: BLE001 — страхуемся от непредвиденных ошибок
            st.session_state["analysis_done"] = False
            st.error(f"❌ Непредвиденная ошибка: {exc}")

# ---------------------------------------------------------------------------
# Блок результатов
# ---------------------------------------------------------------------------
if st.session_state.get("analysis_done"):
    results = st.session_state["results"]
    run_params = st.session_state.get("run_params", {})

    st.markdown('<div class="section-title">📊 Результаты анализа</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Исходное изображение**")
        st.image(st.session_state["original_image"], use_container_width=True)
    with col2:
        st.markdown("**Обнаруженные дефекты**")
        st.image(st.session_state["annotated_image"], use_container_width=True)

    stats = results.get("stats_by_class", {})
    total_defects = results.get("total_defects", 0)
    cracks_count = stats.get("Трещина", 0)
    other_count = max(total_defects - cracks_count, 0)

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Всего дефектов", total_defects)
    metric_col2.metric("Трещины", cracks_count)
    metric_col3.metric("Сколы / прочее", other_count)

    if total_defects == 0:
        st.success("✅ Явных дефектов не обнаружено при заданном пороге уверенности.")
    else:
        st.markdown('<div class="section-title">🔎 Детализация по каждому дефекту</div>', unsafe_allow_html=True)
        for i, defect in enumerate(results["defects"], start=1):
            with st.expander(
                f"Дефект №{i}: {defect['class']} "
                f"(уверенность {defect['confidence']:.0%})"
            ):
                st.json(defect)

    st.markdown('<div class="section-title">⬇️ Скачать результаты</div>', unsafe_allow_html=True)

    json_payload = {
        "total_defects": results["total_defects"],
        "stats_by_class": results["stats_by_class"],
        "defects": results["defects"],
    }
    json_bytes = json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")

    download_col1, download_col2 = st.columns(2)
    with download_col1:
        st.download_button(
            label="🧾 Скачать отчёт (PDF)",
            data=generate_pdf_report(
                results=results,
                original_image=st.session_state["original_image"],
                annotated_image=st.session_state["annotated_image"],
                model_id=run_params.get("model_id", model_id),
                confidence_threshold=run_params.get("confidence_threshold", confidence_threshold),
                overlap_threshold=run_params.get("overlap_threshold", overlap_threshold),
                opacity_threshold=run_params.get("opacity_threshold", opacity_threshold),
                source_filename=run_params.get("source_filename", "photo"),
            ),
            file_name="defect_analysis_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with download_col2:
        st.download_button(
            label="🗂️ Скачать результаты (JSON)",
            data=json_bytes,
            file_name="defect_analysis_results.json",
            mime="application/json",
            use_container_width=True,
        )

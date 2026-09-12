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
from typing import Any, Dict, List

import streamlit as st
from PIL import Image

from detector import (
    RoboflowAuthError,
    RoboflowDetectorError,
    RoboflowEmptyResponseError,
    RoboflowNetworkError,
)
from ensemble_detector import DEFAULT_REGISTRY, EnsembleDetector, ModelSpec
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

DEFAULT_MODEL_ID = "concrete-defect-detection-zuym8/1"

st.set_page_config(
    page_title="Детектор дефектов бетонных конструкций",
    page_icon="🏗️",
    layout="wide",
)

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# ---------------------------------------------------------------------------
# Немного оформления — минимально и точечно, чтобы не ломать верстку
# Streamlit. Основной внешний вид (цвета кнопок, слайдеров, ссылок) задаётся
# через .streamlit/config.toml, а не через хрупкие CSS-хаки.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; max-width: 1100px; }
        .hero {
            background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 100%);
            border-radius: 14px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.5rem;
        }
        .hero h1 { color: #FFFFFF; font-size: 1.7rem; margin: 0 0 0.4rem 0; }
        .hero p { color: #DBEAFE; font-size: 1rem; margin: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Боковая панель
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Настройки")

    st.caption(
        "⚙️ Анализ всегда выполняется ансамблем из нескольких моделей — "
        "каждая отвечает за свой тип дефекта, а результаты объединяются с "
        "устранением дублей между моделями. Набор моделей фиксирован: "
        "можно включать/отключать модели и настраивать их confidence, но "
        "не менять сам список."
    )

    ensemble_selection: List[ModelSpec] = []
    for spec in DEFAULT_REGISTRY:
        col_check, col_conf = st.columns([2.2, 1])
        with col_check:
            is_enabled = st.checkbox(
                f"**{spec.name}**  \n`{spec.model_id}`",
                value=spec.enabled,
                key=f"ensemble_enabled__{spec.model_id}",
            )
        with col_conf:
            model_confidence = st.number_input(
                "Confidence",
                min_value=0.0,
                max_value=1.0,
                value=spec.confidence,
                step=0.05,
                key=f"ensemble_conf__{spec.model_id}",
                disabled=not is_enabled,
            )
        ensemble_selection.append(
            ModelSpec(
                name=spec.name,
                model_id=spec.model_id,
                responsible_classes=spec.responsible_classes,
                confidence=model_confidence,
                enabled=is_enabled,
            )
        )

    cross_model_iou = st.slider(
        "IoU для объединения между моделями",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help=(
            "Если рамки от разных моделей пересекаются сильнее этого "
            "значения — считаем, что это один и тот же дефект, и "
            "оставляем только более уверенное предсказание."
        ),
    )

    with st.expander("Пороги детекции", expanded=True):
        confidence_threshold = None
        st.caption(
            "Confidence Threshold задаётся отдельно для каждой модели "
            "в таблице ансамбля выше."
        )
        overlap_threshold = st.slider(
            "Overlap Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help=(
                "IoU-порог для объединения (NMS) перекрывающихся рамок одного "
                "дефекта. Меньше значение — агрессивнее схлопывание дублей."
            ),
        )
        opacity_threshold = st.slider(
            "Opacity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help=(
                "Непрозрачность цветной заливки внутри рамок на итоговом "
                "изображении. Влияет только на отображение, не на детекцию."
            ),
        )

    st.divider()
    st.caption(
        "🔒 Подключение к Roboflow уже настроено — вводить API-ключ не нужно."
    )
    st.caption(
        "Готовую модель можно найти в "
        "[Roboflow Universe](https://universe.roboflow.com), либо указать "
        "свою выше."
    )

# ---------------------------------------------------------------------------
# Основная область — hero-заголовок и загрузка файла
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🏗️ Автоматическое обнаружение дефектов конструкций</h1>
        <p>Загрузите фотографию бетонной конструкции — сервис найдёт трещины,
        сколы, высолы и раковины с помощью ансамбля ИИ-моделей на базе Roboflow.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, preview_col = st.columns([2, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Загрузите фото конструкции",
        type=["jpg", "jpeg", "png"],
    )
    no_models_enabled = not any(spec.enabled for spec in ensemble_selection)
    run_disabled = uploaded_file is None or no_models_enabled
    run_button = st.button(
        "🔍 Запустить анализ",
        disabled=run_disabled,
        use_container_width=True,
        type="primary",
    )
    if uploaded_file is None:
        st.info("Загрузите фотографию конструкции, чтобы начать анализ.")
    elif no_models_enabled:
        st.info("Включите хотя бы одну модель ансамбля в сайдбаре.")

with preview_col:
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Загруженное изображение", use_container_width=True)

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

                detector = EnsembleDetector(api_key=API_KEY, registry=ensemble_selection)
                results: Dict[str, Any] = detector.detect(
                    image_path=input_path,
                    overlap=overlap_threshold,
                    cross_model_iou=cross_model_iou,
                )
                model_id_for_report = " + ".join(
                    spec.model_id for spec in ensemble_selection if spec.enabled
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
                    "model_id": model_id_for_report,
                    "confidence_threshold": confidence_threshold if confidence_threshold is not None else "см. ансамбль",
                    "overlap_threshold": overlap_threshold,
                    "opacity_threshold": opacity_threshold,
                    "source_filename": uploaded_file.name,
                }

            st.success("✅ Анализ успешно завершён.")
            model_status = results.get("model_status")
            if model_status:
                failed = [name for name, s in model_status.items() if s["status"] == "error"]
                if failed:
                    st.warning(
                        "⚠️ Не удалось получить ответ от моделей: "
                        + ", ".join(failed)
                        + ". Результат построен по остальным моделям ансамбля."
                    )

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

    st.divider()
    st.subheader("📊 Результаты анализа")

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
    with metric_col1.container(border=True):
        st.metric("Всего дефектов", total_defects)
    with metric_col2.container(border=True):
        st.metric("Трещины", cracks_count)
    with metric_col3.container(border=True):
        st.metric("Сколы / прочее", other_count)

    model_status = results.get("model_status")
    if model_status:
        with st.expander("🧩 Статус моделей ансамбля"):
            for name, status in model_status.items():
                if status["status"] == "ok":
                    st.markdown(
                        f"✅ **{name}** (`{status['model_id']}`) — найдено "
                        f"{status['raw_count']}, учтено после фильтра по "
                        f"классу: {status['used_count']}"
                    )
                else:
                    st.markdown(
                        f"❌ **{name}** (`{status['model_id']}`) — ошибка: "
                        f"{status['error']}"
                    )

    if total_defects == 0:
        st.success("✅ Явных дефектов не обнаружено при заданном пороге уверенности.")
    else:
        st.subheader("🔎 Детализация по каждому дефекту")
        for i, defect in enumerate(results["defects"], start=1):
            with st.expander(
                f"Дефект №{i}: {defect['class']} "
                f"(уверенность {defect['confidence']:.0%})"
            ):
                st.json(defect)

    st.divider()
    st.subheader("⬇️ Скачать результаты")

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
                model_id=run_params.get("model_id", DEFAULT_MODEL_ID),
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

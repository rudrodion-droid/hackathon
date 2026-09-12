"""
app.py

Streamlit-приложение для автоматического обнаружения дефектов бетонных
конструкций (трещины, сколы, высолы, раковины) по фотографии, с
использованием Roboflow Inference API.
"""

import json
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

st.set_page_config(
    page_title="Детектор дефектов бетонных конструкций",
    page_icon="🏗️",
    layout="wide",
)

if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

# ---------------------------------------------------------------------------
# Боковая панель
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Настройки")

    api_key = st.text_input(
        "API-ключ Roboflow",
        type="password",
        placeholder="Введите ваш API-ключ",
        help="Ключ используется только в рамках текущей сессии и никуда не сохраняется.",
    )

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
    st.markdown(
        "**Где взять API-ключ?**\n\n"
        "1. Зарегистрируйтесь на [Roboflow](https://roboflow.com).\n"
        "2. Откройте настройки рабочей области (Workspace Settings).\n"
        "3. Скопируйте значение **Private API Key** во вкладке *Roboflow API*.\n"
        "4. Готовую модель детекции дефектов можно найти в "
        "[Roboflow Universe](https://universe.roboflow.com), либо использовать свою."
    )

# ---------------------------------------------------------------------------
# Основная область — заголовок и загрузка файла
# ---------------------------------------------------------------------------
st.title("🏗️ Автоматическое обнаружение дефектов конструкций")
st.markdown(
    "Загрузите фотографию бетонной конструкции — сервис автоматически найдёт "
    "**трещины, сколы, высолы и раковины**, используя предобученную ИИ-модель "
    "на базе Roboflow Inference API."
)

uploaded_file = st.file_uploader(
    "Загрузите фото конструкции",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Загруженное изображение", width=400)

run_disabled = not api_key or uploaded_file is None
run_button = st.button(
    "🔍 Запустить анализ",
    disabled=run_disabled,
    use_container_width=True,
)

if run_disabled:
    if uploaded_file is None and not api_key:
        st.info("Введите API-ключ и загрузите фотографию, чтобы начать анализ.")
    elif uploaded_file is None:
        st.info("Загрузите фотографию конструкции, чтобы начать анализ.")
    else:
        st.info("Введите API-ключ Roboflow в боковой панели, чтобы активировать анализ.")

# ---------------------------------------------------------------------------
# Запуск анализа
# ---------------------------------------------------------------------------
if run_button and uploaded_file is not None and api_key:
    with st.spinner("Анализируем изображение, пожалуйста подождите..."):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_path = str(Path(tmp_dir) / uploaded_file.name)
                output_path = str(Path(tmp_dir) / f"annotated_{uploaded_file.name}")

                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                detector = RoboflowDetector(api_key=api_key, model_id=model_id)
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
    st.markdown("---")
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
    metric_col1.metric("Всего дефектов", total_defects)
    metric_col2.metric("Трещины", cracks_count)
    metric_col3.metric("Сколы / прочее", other_count)

    if total_defects == 0:
        st.success("✅ Явных дефектов не обнаружено при заданном пороге уверенности.")
    else:
        st.markdown("### 🔎 Детализация по каждому дефекту")
        for i, defect in enumerate(results["defects"], start=1):
            with st.expander(
                f"Дефект №{i}: {defect['class']} "
                f"(уверенность {defect['confidence']:.0%})"
            ):
                st.json(defect)

    json_payload = {
        "total_defects": results["total_defects"],
        "stats_by_class": results["stats_by_class"],
        "defects": results["defects"],
    }
    json_bytes = json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")

    st.download_button(
        label="⬇️ Скачать результаты (JSON)",
        data=json_bytes,
        file_name="defect_analysis_results.json",
        mime="application/json",
        use_container_width=True,
    )

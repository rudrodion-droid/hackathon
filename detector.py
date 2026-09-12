"""
detector.py

Модуль для работы с Roboflow Inference API: отправка изображений на
детекцию дефектов бетонных конструкций, разбор результатов и
визуализация bounding box'ов поверх исходного фото.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Не найден пакет 'inference-sdk'. Установите его командой: "
        "pip install --upgrade inference-sdk"
    ) from exc


# Маппинг английских названий классов дефектов на русский язык.
CLASS_NAME_RU: Dict[str, str] = {
    "crack": "Трещина",
    "cracks": "Трещина",
    "concrete-crack": "Трещина",
    "spalling": "Скол",
    "spall": "Скол",
    "efflorescence": "Высол",
    "honeycomb": "Раковина",
    "honeycombing": "Раковина",
    "rebar-exposure": "Оголение арматуры",
    "corrosion": "Коррозия",
}

# Цветовая кодировка (в формате RGB) по типу дефекта.
CLASS_COLOR_RGB: Dict[str, Tuple[int, int, int]] = {
    "Трещина": (220, 30, 30),    # красный
    "Скол": (230, 200, 20),      # желтый
    "Высол": (30, 90, 220),      # синий
    "Раковина": (30, 170, 60),   # зеленый
}
DEFAULT_COLOR_RGB: Tuple[int, int, int] = (230, 230, 230)  # серый — для неизвестного класса

# Возможные пути к TTF-шрифтам с поддержкой кириллицы (для подписей на фото).
_FONT_CANDIDATES: List[str] = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]


def _load_font(size: int = 18) -> ImageFont.FreeTypeFont:
    """
    Пытается загрузить TrueType-шрифт с поддержкой кириллицы.

    Стандартный битмап-шрифт Pillow (load_default) не умеет корректно
    отрисовывать кириллические символы, поэтому подписи на изображении
    рисуются через Pillow с явно найденным TTF-шрифтом.

    Args:
        size: Размер шрифта в пунктах.

    Returns:
        Объект шрифта Pillow, пригодный для кириллицы (либо стандартный
        шрифт, если подходящий TTF-файл не найден на диске).
    """
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


@dataclass
class DetectionResult:
    """Структура для хранения результата детекции одного дефекта."""

    class_name_en: str
    class_name_ru: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    def to_bbox_xyxy(self) -> Tuple[int, int, int, int]:
        """Преобразует центр+размер bbox'а (формат Roboflow) в координаты x1, y1, x2, y2."""
        x1 = int(self.x - self.width / 2)
        y1 = int(self.y - self.height / 2)
        x2 = int(self.x + self.width / 2)
        y2 = int(self.y + self.height / 2)
        return x1, y1, x2, y2

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует результат в словарь, пригодный для сериализации в JSON."""
        x1, y1, x2, y2 = self.to_bbox_xyxy()
        return {
            "class": self.class_name_ru,
            "class_en": self.class_name_en,
            "confidence": round(self.confidence, 4),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "size_px": {"width": int(self.width), "height": int(self.height)},
        }


class RoboflowDetectorError(Exception):
    """Базовое исключение модуля детектора."""


class RoboflowAuthError(RoboflowDetectorError):
    """Ошибка авторизации (неверный или просроченный API-ключ)."""


class RoboflowNetworkError(RoboflowDetectorError):
    """Ошибка сети при обращении к Roboflow API."""


class RoboflowEmptyResponseError(RoboflowDetectorError):
    """API вернул пустой или некорректно сформированный ответ."""


class RoboflowDetector:
    """
    Обёртка над Roboflow Inference API для детекции дефектов бетонных
    конструкций (трещины, сколы, высолы, раковины и другие дефекты).
    """

    DEFAULT_MODEL_ID = "concrete-crack-detection/1"
    # Актуальный endpoint Roboflow Serverless Hosted API (v2).
    # Старый "https://detect.roboflow.com" с ключом в query-параметре
    # считается устаревшим и может блокироваться на стороне Roboflow.
    DEFAULT_API_URL = "https://serverless.roboflow.com"

    def __init__(
        self,
        api_key: str,
        model_id: str = DEFAULT_MODEL_ID,
        api_url: str = DEFAULT_API_URL,
    ) -> None:
        """
        Args:
            api_key: API-ключ пользователя Roboflow.
            model_id: Идентификатор модели в формате "project-slug/version".
            api_url: Базовый URL инференс-сервиса Roboflow.

        Raises:
            RoboflowAuthError: если api_key пустой.
            RoboflowDetectorError: если клиент не удалось инициализировать.
        """
        if not api_key or not api_key.strip():
            raise RoboflowAuthError("API-ключ не может быть пустым.")

        self.api_key = api_key.strip()
        self.model_id = model_id
        self.api_url = api_url

        try:
            self.client = InferenceHTTPClient(
                api_url=self.api_url, api_key=self.api_key
            ).configure(InferenceConfiguration(api_key_transport="header"))
        except Exception as exc:
            raise RoboflowDetectorError(
                f"Не удалось инициализировать клиент Roboflow: {exc}"
            ) from exc

    @staticmethod
    def _translate_class(class_name_en: str) -> str:
        """Переводит английское имя класса дефекта на русский язык."""
        key = class_name_en.strip().lower()
        return CLASS_NAME_RU.get(key, class_name_en.capitalize())

    def detect(self, image_path: str, confidence: float = 0.5) -> Dict[str, Any]:
        """
        Отправляет изображение на анализ в Roboflow API и разбирает ответ.

        Args:
            image_path: Путь к файлу изображения на диске.
            confidence: Порог уверенности (0.0-1.0) для отбора предсказаний.

        Returns:
            Словарь вида:
                {
                    "total_defects": int,
                    "defects": List[Dict[str, Any]],
                    "stats_by_class": Dict[str, int],
                    "raw_response": Dict[str, Any],
                }

        Raises:
            RoboflowDetectorError: если файл изображения не найден.
            RoboflowAuthError: при неверном/просроченном API-ключе.
            RoboflowNetworkError: при сетевых сбоях (нет соединения, таймаут и т.п.).
            RoboflowEmptyResponseError: если API вернул пустой/некорректный ответ.
        """
        if not os.path.isfile(image_path):
            raise RoboflowDetectorError(f"Файл изображения не найден: {image_path}")

        try:
            raw_response = self.client.infer(image_path, model_id=self.model_id)
        except Exception as exc:
            message = str(exc).lower()
            if any(term in message for term in ("unauthorized", "401", "invalid api key", "forbidden")):
                raise RoboflowAuthError(
                    "Неверный API-ключ или нет доступа к указанной модели."
                ) from exc
            if any(term in message for term in ("connection", "timeout", "network", "resolve", "dns")):
                raise RoboflowNetworkError(
                    f"Проблема с сетевым подключением к Roboflow API: {exc}"
                ) from exc
            raise RoboflowDetectorError(f"Ошибка при обращении к Roboflow API: {exc}") from exc

        if not raw_response or not isinstance(raw_response, dict):
            raise RoboflowEmptyResponseError("API вернул пустой или некорректный ответ.")

        predictions = raw_response.get("predictions") or []

        defects: List[DetectionResult] = []
        for pred in predictions:
            try:
                conf = float(pred.get("confidence", 0.0))
                if conf < confidence:
                    continue
                class_en = str(pred.get("class", "unknown"))
                defects.append(
                    DetectionResult(
                        class_name_en=class_en,
                        class_name_ru=self._translate_class(class_en),
                        confidence=conf,
                        x=float(pred.get("x", 0.0)),
                        y=float(pred.get("y", 0.0)),
                        width=float(pred.get("width", 0.0)),
                        height=float(pred.get("height", 0.0)),
                    )
                )
            except (TypeError, ValueError):
                # Пропускаем некорректно сформированные предсказания,
                # не прерывая обработку остальных.
                continue

        stats_by_class: Dict[str, int] = {}
        for defect in defects:
            stats_by_class[defect.class_name_ru] = stats_by_class.get(defect.class_name_ru, 0) + 1

        return {
            "total_defects": len(defects),
            "defects": [d.to_dict() for d in defects],
            "stats_by_class": stats_by_class,
            "raw_response": raw_response,
            "_defect_objects": defects,  # используется внутренне методом visualize()
        }

    def visualize(
        self,
        image_path: str,
        results: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Отрисовывает bounding box'ы найденных дефектов поверх исходного изображения.

        Рамки рисуются через OpenCV, а подписи (на русском языке) — через
        Pillow с TTF-шрифтом, так как стандартные шрифты OpenCV не
        поддерживают кириллицу.

        Args:
            image_path: Путь к исходному изображению.
            results: Результат, возвращённый методом detect().
            output_path: Если указан, итоговое изображение сохраняется по этому пути.

        Returns:
            Изображение в формате Pillow (RGB) с отрисованными дефектами.

        Raises:
            RoboflowDetectorError: если изображение не удалось прочитать.
        """
        cv_image = cv2.imread(image_path)
        if cv_image is None:
            try:
                pil_source = Image.open(image_path).convert("RGB")
                cv_image = cv2.cvtColor(np.array(pil_source), cv2.COLOR_RGB2BGR)
            except Exception as exc:
                raise RoboflowDetectorError(
                    f"Не удалось прочитать изображение для визуализации: {exc}"
                ) from exc

        defects: List[DetectionResult] = results.get("_defect_objects", [])

        # 1) Рисуем прямоугольники через OpenCV (BGR).
        for defect in defects:
            x1, y1, x2, y2 = defect.to_bbox_xyxy()
            r, g, b = CLASS_COLOR_RGB.get(defect.class_name_ru, DEFAULT_COLOR_RGB)
            color_bgr = (b, g, r)
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color_bgr, 2)

        # 2) Переводим изображение в Pillow (RGB) и рисуем подписи на кириллице.
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        font = _load_font(size=16)

        for defect in defects:
            x1, y1, _, _ = defect.to_bbox_xyxy()
            color_rgb = CLASS_COLOR_RGB.get(defect.class_name_ru, DEFAULT_COLOR_RGB)
            label = f"{defect.class_name_ru} {defect.confidence:.0%}"

            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            label_y = max(0, y1 - text_h - 6)
            draw.rectangle(
                [x1, label_y, x1 + text_w + 6, label_y + text_h + 6],
                fill=color_rgb,
            )
            draw.text((x1 + 3, label_y + 2), label, fill=(0, 0, 0), font=font)

        if output_path:
            pil_image.save(output_path)

        return pil_image

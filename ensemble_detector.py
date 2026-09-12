"""
ensemble_detector.py

Оркестрация нескольких моделей Roboflow, каждая из которых специализируется
на своём типе дефекта бетонных конструкций, для анализа одной фотографии:

- модели опрашиваются параллельно (иначе время анализа росло бы линейно
  с числом моделей);
- у каждой модели есть "зона ответственности" по классам — так общая
  (запасная) модель не спорит со специализированной моделью по тем классам,
  для которых уже есть специалист, а только дополняет результат по классам,
  которые больше никто не покрывает;
- результаты разных моделей объединяются с устранением дублей по IoU
  (cross-model NMS) — иначе один и тот же физический дефект может попасть
  в отчёт дважды, если его нашли две разные модели;
- отказ одной модели (таймаут, квота, неверный ключ) не должен ронять
  анализ целиком — статус по каждой модели возвращается отдельно.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set

from detector import (
    DetectionResult,
    RoboflowDetector,
    RoboflowDetectorError,
    render_annotated_image,
)


@dataclass
class ModelSpec:
    """Описание одной модели в ансамбле."""

    name: str
    model_id: str
    # None означает "запасная" модель — она покрывает любые классы, для
    # которых нет отдельного специалиста. Непустое множество — модель
    # учитывается только по перечисленным русским названиям классов.
    responsible_classes: Optional[Set[str]] = None
    confidence: float = 0.5
    enabled: bool = True


# ---------------------------------------------------------------------------
# Реестр моделей ансамбля — фиксированный набор из 4 моделей, каждая
# отвечает за свой тип дефекта. Список умышленно не редактируется через
# UI (только включение/отключение) — чтобы в ансамбль не попадали
# случайные/непроверенные model_id.
# ---------------------------------------------------------------------------
DEFAULT_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="Трещины",
        model_id="concrete-crack-dfd3i/3",
        responsible_classes={"Трещина"},
        confidence=0.5,
    ),
    ModelSpec(
        name="Сколы",
        model_id="cracks-and-spalling-800-img/2",
        responsible_classes={"Скол"},
        confidence=0.5,
    ),
    ModelSpec(
        name="Высолы",
        model_id="efflorescence-hsa7w/1",
        responsible_classes={"Высол"},
        confidence=0.5,
    ),
    ModelSpec(
        name="Общая модель (запасная)",
        model_id="concrete-defect-detection-zuym8/1",
        responsible_classes=None,
        confidence=0.5,
    ),
]


def _iou(a: DetectionResult, b: DetectionResult) -> float:
    """IoU (Intersection over Union) двух bbox'ов в формате x1,y1,x2,y2."""
    ax1, ay1, ax2, ay2 = a.to_bbox_xyxy()
    bx1, by1, bx2, by2 = b.to_bbox_xyxy()

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _cross_model_dedup(
    defects: List[DetectionResult], iou_threshold: float = 0.5
) -> List[DetectionResult]:
    """
    Убирает дубли между результатами разных моделей.

    Если рамки двух предсказаний (даже разных классов — например, одна
    модель назвала пятно "Скол", а другая "Высол") пересекаются сильнее
    iou_threshold, это, скорее всего, один и тот же физический дефект.
    Оставляется предсказание с более высокой уверенностью.
    """
    ordered = sorted(defects, key=lambda d: d.confidence, reverse=True)
    kept: List[DetectionResult] = []

    for candidate in ordered:
        if any(_iou(candidate, existing) >= iou_threshold for existing in kept):
            continue
        kept.append(candidate)

    return kept


class EnsembleDetector:
    """
    Прогоняет одно изображение параллельно через несколько моделей Roboflow
    и объединяет результаты в единый список — в том же формате, что
    возвращает RoboflowDetector.detect(), плюс поле "model_status".
    """

    def __init__(
        self,
        api_key: str,
        registry: Optional[Sequence[ModelSpec]] = None,
        api_url: str = RoboflowDetector.DEFAULT_API_URL,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url
        self.registry: List[ModelSpec] = [
            spec for spec in (registry or DEFAULT_REGISTRY) if spec.enabled
        ]
        if not self.registry:
            raise RoboflowDetectorError("В ансамбле нет ни одной активной модели.")

    def _run_single_model(
        self, spec: ModelSpec, image_path: str, overlap: float
    ) -> Dict[str, Any]:
        detector = RoboflowDetector(
            api_key=self.api_key, model_id=spec.model_id, api_url=self.api_url
        )
        return detector.detect(
            image_path=image_path, confidence=spec.confidence, overlap=overlap
        )

    def detect(
        self,
        image_path: str,
        overlap: float = 0.5,
        cross_model_iou: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Args:
            image_path: Путь к изображению.
            overlap: IoU-порог NMS внутри каждой отдельной модели (передаётся
                напрямую в Roboflow API, как и в RoboflowDetector.detect()).
            cross_model_iou: IoU-порог для устранения дублей МЕЖДУ моделями.

        Returns:
            {
                "total_defects": int,
                "defects": List[Dict[str, Any]],
                "stats_by_class": Dict[str, int],
                "model_status": Dict[str, Dict[str, Any]],  # по каждой модели
                "_defect_objects": List[DetectionResult],
            }

        Raises:
            RoboflowDetectorError: если абсолютно все модели ансамбля упали
                (сетевая ошибка, неверный ключ и т.п.) — тогда показывать
                частичный результат нечего.
        """
        all_defects: List[DetectionResult] = []
        model_status: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max(1, len(self.registry))) as pool:
            futures = {
                pool.submit(self._run_single_model, spec, image_path, overlap): spec
                for spec in self.registry
            }
            for future in as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                    defect_objects: List[DetectionResult] = list(
                        result.get("_defect_objects", [])
                    )
                    raw_count = len(defect_objects)

                    # Модель учитывается только по своей "зоне ответственности" —
                    # это и есть механизм, который не даёт запасной
                    # мультиклассовой модели дублировать специализированную.
                    if spec.responsible_classes is not None:
                        defect_objects = [
                            d
                            for d in defect_objects
                            if d.class_name_ru in spec.responsible_classes
                        ]

                    all_defects.extend(defect_objects)
                    model_status[spec.name] = {
                        "status": "ok",
                        "model_id": spec.model_id,
                        "raw_count": raw_count,
                        "used_count": len(defect_objects),
                    }
                except Exception as exc:  # noqa: BLE001 — одна упавшая модель не должна ронять весь анализ
                    model_status[spec.name] = {
                        "status": "error",
                        "model_id": spec.model_id,
                        "error": str(exc),
                    }

        if all(status["status"] == "error" for status in model_status.values()):
            details = "; ".join(
                f"{name}: {status['error']}" for name, status in model_status.items()
            )
            raise RoboflowDetectorError(
                f"Ни одна из моделей ансамбля не смогла обработать изображение. {details}"
            )

        merged_defects = _cross_model_dedup(all_defects, iou_threshold=cross_model_iou)

        stats_by_class: Dict[str, int] = {}
        for defect in merged_defects:
            stats_by_class[defect.class_name_ru] = (
                stats_by_class.get(defect.class_name_ru, 0) + 1
            )

        return {
            "total_defects": len(merged_defects),
            "defects": [d.to_dict() for d in merged_defects],
            "stats_by_class": stats_by_class,
            "model_status": model_status,
            "_defect_objects": merged_defects,
        }

    def visualize(
        self,
        image_path: str,
        results: Dict[str, Any],
        output_path: Optional[str] = None,
        opacity: float = 0.75,
    ):
        """Отрисовка — используется тот же общий рендерер, что и в RoboflowDetector."""
        defects: List[DetectionResult] = results.get("_defect_objects", [])
        return render_annotated_image(
            image_path=image_path,
            defects=defects,
            output_path=output_path,
            opacity=opacity,
        )

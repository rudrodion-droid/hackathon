"""
report.py

Формирует PDF-отчёт по результатам анализа дефектов бетонных конструкций
(сводка, изображения "до/после", таблица найденных дефектов) с помощью
reportlab — без зависимости от системных шрифтовых движков или браузера.
"""

from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Шрифт с поддержкой кириллицы.
#
# Встроенные PDF-шрифты reportlab (Helvetica и т.п.) не содержат кириллических
# глифов — любой русский текст ими рисуется как пустые квадраты ("тофу").
# Чтобы отчёт не зависел от того, какие шрифты установлены на сервере
# деплоя (Railway/Docker их не ставит), в репозиторий добавлен файл шрифта
# DejaVu Sans (fonts/DejaVuSans.ttf, fonts/DejaVuSans-Bold.ttf) — он
# встраивается прямо в PDF при генерации.
# ---------------------------------------------------------------------------
_FONTS_DIR = Path(__file__).resolve().parent / "fonts"
FONT_REGULAR = "DejaVuSans"
FONT_BOLD = "DejaVuSans-Bold"


def _register_fonts() -> None:
    """Регистрирует встроенный шрифт с поддержкой кириллицы в reportlab.

    Если файлы шрифта по какой-то причине отсутствуют (например, при
    локальном запуске не из корня репозитория), молча откатываемся на
    стандартный Helvetica — отчёт всё равно сформируется, просто кириллица
    в нём снова будет нечитаемой. Это сознательный компромисс: лучше
    показать отчёт с проблемой шрифта, чем не показать отчёт вообще.
    """
    global FONT_REGULAR, FONT_BOLD

    regular_path = _FONTS_DIR / "DejaVuSans.ttf"
    bold_path = _FONTS_DIR / "DejaVuSans-Bold.ttf"

    if not (regular_path.is_file() and bold_path.is_file()):
        FONT_REGULAR = "Helvetica"
        FONT_BOLD = "Helvetica-Bold"
        return

    pdfmetrics.registerFont(TTFont("DejaVuSans", str(regular_path)))
    pdfmetrics.registerFont(TTFont("DejaVuSans-Bold", str(bold_path)))
    # Регистрация "семейства" нужна, чтобы теги <b>...</b> внутри Paragraph
    # переключались на жирный DejaVuSans-Bold, а не на жирный Helvetica.
    pdfmetrics.registerFontFamily(
        "DejaVuSans",
        normal="DejaVuSans",
        bold="DejaVuSans-Bold",
        italic="DejaVuSans",
        boldItalic="DejaVuSans-Bold",
    )


_register_fonts()

# Единая цветовая палитра отчёта (согласована с интерфейсом приложения).
COLOR_PRIMARY = colors.HexColor("#2563EB")
COLOR_PRIMARY_DARK = colors.HexColor("#1E3A8A")
COLOR_ACCENT = colors.HexColor("#F1F5F9")
COLOR_TEXT = colors.HexColor("#1E293B")
COLOR_MUTED = colors.HexColor("#64748B")
COLOR_OK = colors.HexColor("#16A34A")
COLOR_WARN = colors.HexColor("#DC2626")

CLASS_COLOR_HEX: Dict[str, str] = {
    "Трещина": "#DC1E1E",
    "Скол": "#E6C814",
    "Высол": "#1E5ADC",
    "Раковина": "#1EAA3C",
}


def _pil_to_flowable(pil_image: PILImage.Image, max_width_mm: float, max_height_mm: float) -> RLImage:
    """Конвертирует PIL-изображение во flowable reportlab с сохранением пропорций."""
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    max_w = max_width_mm * mm
    max_h = max_height_mm * mm
    ratio = min(max_w / pil_image.width, max_h / pil_image.height)
    draw_w = pil_image.width * ratio
    draw_h = pil_image.height * ratio

    return RLImage(buf, width=draw_w, height=draw_h)


def _build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: Dict[str, ParagraphStyle] = {}

    styles["Title"] = ParagraphStyle(
        "ReportTitle",
        parent=base["Title"],
        fontName=FONT_BOLD,
        fontSize=20,
        leading=24,
        textColor=COLOR_PRIMARY_DARK,
        spaceAfter=4,
    )
    styles["Subtitle"] = ParagraphStyle(
        "ReportSubtitle",
        parent=base["Normal"],
        fontName=FONT_REGULAR,
        fontSize=10.5,
        textColor=COLOR_MUTED,
        spaceAfter=14,
    )
    styles["H2"] = ParagraphStyle(
        "ReportH2",
        parent=base["Heading2"],
        fontName=FONT_BOLD,
        fontSize=13,
        textColor=COLOR_PRIMARY_DARK,
        spaceBefore=14,
        spaceAfter=8,
    )
    styles["Body"] = ParagraphStyle(
        "ReportBody",
        parent=base["Normal"],
        fontName=FONT_REGULAR,
        fontSize=9.5,
        leading=13,
        textColor=COLOR_TEXT,
    )
    styles["Caption"] = ParagraphStyle(
        "ReportCaption",
        parent=base["Normal"],
        fontName=FONT_REGULAR,
        fontSize=8.5,
        textColor=COLOR_MUTED,
        alignment=1,  # center
        spaceBefore=4,
    )
    styles["Footer"] = ParagraphStyle(
        "ReportFooter",
        parent=base["Normal"],
        fontName=FONT_REGULAR,
        fontSize=7.5,
        textColor=COLOR_MUTED,
    )
    return styles


def generate_pdf_report(
    results: Dict[str, Any],
    original_image: PILImage.Image,
    annotated_image: PILImage.Image,
    model_id: str,
    confidence_threshold: float,
    overlap_threshold: float,
    opacity_threshold: float,
    source_filename: str = "photo",
) -> bytes:
    """
    Строит PDF-отчёт по результатам анализа и возвращает его как bytes,
    готовые к отдаче через st.download_button.

    Args:
        results: Результат RoboflowDetector.detect().
        original_image: Исходное изображение (PIL).
        annotated_image: Изображение с отрисованными дефектами (PIL).
        model_id: Идентификатор использованной модели Roboflow.
        confidence_threshold: Использованный порог уверенности.
        overlap_threshold: Использованный порог перекрытия (IoU).
        opacity_threshold: Использованная непрозрачность заливки.
        source_filename: Имя исходного файла (для отображения в отчёте).

    Returns:
        Содержимое PDF-файла в виде байтов.
    """
    styles = _build_styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Отчёт по обнаружению дефектов конструкций",
    )

    story = []

    # --- Заголовок -----------------------------------------------------
    story.append(Paragraph("🏗️ Отчёт по обнаружению дефектов конструкций", styles["Title"]))
    generated_at = datetime.now().strftime("%d.%m.%Y %H:%M")
    story.append(
        Paragraph(
            f"Файл: <b>{source_filename}</b> &nbsp;•&nbsp; Сформирован: {generated_at}",
            styles["Subtitle"],
        )
    )

    # Тонкая цветная линия-разделитель под заголовком.
    header_rule = Table([[""]], colWidths=[doc.width], rowHeights=[2])
    header_rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), COLOR_PRIMARY)]))
    story.append(header_rule)
    story.append(Spacer(1, 12))

    # --- Параметры анализа ----------------------------------------------
    story.append(Paragraph("Параметры анализа", styles["H2"]))

    def _fmt_pct(value: Any) -> str:
        """Форматирует порог как проценты; при ансамбле confidence задаётся
        отдельно для каждой модели, поэтому значение может быть не числом."""
        try:
            return f"{float(value):.0%}"
        except (TypeError, ValueError):
            return str(value)

    params_table = Table(
        [
            ["Модель Roboflow", Paragraph(model_id, styles["Body"])],
            ["Confidence Threshold", _fmt_pct(confidence_threshold)],
            ["Overlap Threshold", _fmt_pct(overlap_threshold)],
            ["Opacity Threshold", _fmt_pct(opacity_threshold)],
        ],
        colWidths=[55 * mm, doc.width - 55 * mm],
    )
    params_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), COLOR_ACCENT),
                ("TEXTCOLOR", (0, 0), (0, -1), COLOR_MUTED),
                ("TEXTCOLOR", (1, 0), (1, -1), COLOR_TEXT),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
                ("FONTNAME", (0, 0), (0, -1), FONT_BOLD),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(params_table)
    story.append(Spacer(1, 14))

    # --- Сводка по дефектам ----------------------------------------------
    total_defects = results.get("total_defects", 0)
    stats_by_class = results.get("stats_by_class", {})

    story.append(Paragraph("Сводка", styles["H2"]))

    summary_row = [
        Paragraph(f"<b>{total_defects}</b><br/><font size=8 color='#64748B'>Всего дефектов</font>", styles["Body"]),
    ]
    for class_name in sorted(stats_by_class.keys()):
        count = stats_by_class[class_name]
        hex_color = CLASS_COLOR_HEX.get(class_name, "#64748B")
        summary_row.append(
            Paragraph(
                f"<b><font color='{hex_color}'>{count}</font></b><br/>"
                f"<font size=8 color='#64748B'>{class_name}</font>",
                styles["Body"],
            )
        )

    summary_table = Table([summary_row], colWidths=[doc.width / len(summary_row)] * len(summary_row))
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_ACCENT),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 6))

    if total_defects == 0:
        story.append(
            Paragraph(
                "<font color='#16A34A'>✅ Явных дефектов не обнаружено при заданном пороге уверенности.</font>",
                styles["Body"],
            )
        )
    story.append(Spacer(1, 10))

    # --- Изображения до/после --------------------------------------------
    story.append(Paragraph("Сравнение изображений", styles["H2"]))
    img_col_width = (doc.width - 8 * mm) / 2
    original_flowable = _pil_to_flowable(original_image, img_col_width / mm, 90)
    annotated_flowable = _pil_to_flowable(annotated_image, img_col_width / mm, 90)

    images_table = Table(
        [
            [original_flowable, annotated_flowable],
            [
                Paragraph("Исходное изображение", styles["Caption"]),
                Paragraph("Обнаруженные дефекты", styles["Caption"]),
            ],
        ],
        colWidths=[img_col_width, img_col_width],
        hAlign="CENTER",
    )
    images_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
            ]
        )
    )
    story.append(images_table)

    # --- Детализация по дефектам -------------------------------------------
    defects = results.get("defects", [])
    if defects:
        story.append(PageBreak())
        story.append(Paragraph("Детализация по каждому дефекту", styles["H2"]))

        table_data = [["№", "Класс", "Уверенность", "BBox (x1, y1, x2, y2)", "Размер, px"]]
        for i, defect in enumerate(defects, start=1):
            bbox = defect.get("bbox", {})
            size = defect.get("size_px", {})
            table_data.append(
                [
                    str(i),
                    defect.get("class", "—"),
                    f"{defect.get('confidence', 0):.0%}",
                    f"({bbox.get('x1', 0)}, {bbox.get('y1', 0)}, {bbox.get('x2', 0)}, {bbox.get('y2', 0)})",
                    f"{size.get('width', 0)} × {size.get('height', 0)}",
                ]
            )

        defects_table = Table(
            table_data,
            colWidths=[10 * mm, 30 * mm, 25 * mm, 60 * mm, doc.width - 125 * mm],
            repeatRows=1,
        )
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
            ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#E2E8F0")),
        ]
        for row_idx in range(1, len(table_data)):
            if row_idx % 2 == 0:
                table_style.append(("BACKGROUND", (0, row_idx), (-1, row_idx), COLOR_ACCENT))
        defects_table.setStyle(TableStyle(table_style))
        story.append(defects_table)

    # --- Подвал -------------------------------------------------------------
    story.append(Spacer(1, 18))
    story.append(
        Paragraph(
            "Отчёт сформирован автоматически MVP-прототипом детектора дефектов "
            "бетонных конструкций на базе Roboflow Inference API. Результаты "
            "носят вспомогательный характер и не заменяют заключение "
            "квалифицированного специалиста по строительной экспертизе.",
            styles["Footer"],
        )
    )

    doc.build(story)
    return buf.getvalue()

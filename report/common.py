"""Shared helpers for the report builder (P4).

Every table and number in the report is read from a file in results/ or artifacts/ by the section
modules; these helpers only lay things out.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, Spacer, Table, TableStyle

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
ARTIFACTS = REPO / "artifacts"
PAGE_W = A4[0] - 4 * cm           # usable width

_ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=_ss["BodyText"], fontName="Helvetica", fontSize=9.4, leading=12.8, alignment=TA_LEFT, spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8, leading=10.2, textColor=colors.HexColor("#444444"))
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#444444"), spaceAfter=10)
H1 = ParagraphStyle("h1", parent=_ss["Heading1"], fontName="Helvetica-Bold", fontSize=15, leading=18, spaceBefore=6, spaceAfter=8,
                    textColor=colors.HexColor("#0b3d5c"))
H2 = ParagraphStyle("h2", parent=_ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=8, spaceAfter=5,
                    textColor=colors.HexColor("#0b3d5c"))
H3 = ParagraphStyle("h3", parent=_ss["Heading3"], fontName="Helvetica-Bold", fontSize=9.8, leading=12.5, spaceBefore=5, spaceAfter=3)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.2, leading=8.8, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Helvetica-Bold")


def P(text: str, style=BODY):
    return Paragraph(text, style)


def bullets(items, style=BODY):
    return [Paragraph("&bull; " + t, ParagraphStyle("b", parent=style, leftIndent=11, firstLineIndent=-8, spaceAfter=2.5)) for t in items]


def img(path: Path, caption: str, width=PAGE_W, max_h=None):
    """A figure with its caption, kept together on one page. Missing files become a visible note."""
    path = Path(path)
    if not path.exists():
        return [P(f"<i>[missing figure: {path.name}]</i>", SMALL)]
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    scale = width / w
    dh = h * scale
    if max_h and dh > max_h:
        scale = max_h / h
        width, dh = w * scale, max_h
    return [KeepTogether([Image(str(path), width=width, height=dh), P(caption, CAP)])]


def table(rows, col_widths=None, header=True, font=7.2, zebra=True, align_right_from=1):
    """rows: list of lists (strings, numbers or Paragraphs). First row is the header."""
    st = ParagraphStyle("c", parent=CELL, fontSize=font, leading=font + 1.6)
    stb = ParagraphStyle("cb", parent=st, fontName="Helvetica-Bold")
    data = []
    for i, r in enumerate(rows):
        data.append([c if not isinstance(c, str) else Paragraph(c, stb if (header and i == 0) else st) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9c3cc")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe9f0")))
    if zebra:
        for i in range(1 if header else 0, len(rows)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f6f9fb")))
    t.setStyle(TableStyle(style))
    return t


def f(v, nd=4, pct=False, na="N/A"):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return na
    return f"{v * 100:.{max(nd - 2, 1)}f}%" if pct else f"{v:.{nd}f}"


def read(name: str, folder: Path = RESULTS) -> pd.DataFrame:
    """Read a CSV; trip ids stay text so leading zeros survive."""
    return pd.read_csv(folder / name, dtype={"trip_id": str})


def pb():
    return PageBreak()


def sp(h=6):
    return Spacer(1, h)

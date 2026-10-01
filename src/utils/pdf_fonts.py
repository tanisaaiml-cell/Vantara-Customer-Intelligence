"""Portable PDF fonts embedded from ReportLab's bundled Bitstream Vera."""

from pathlib import Path

import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def register_fonts() -> None:
    """Avoid dependence on a viewer's Helvetica font substitution."""
    folder = Path(reportlab.__file__).parent / "fonts"
    for name, file in [("VantaraSans", "Vera.ttf"), ("VantaraBold", "VeraBd.ttf")]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(folder / file)))

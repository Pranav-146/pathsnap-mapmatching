"""Build a PDF of ONE report section so its owner can check it alone (P4 helper).

    python report/test_section.py sec_eda        # P1  -> report/_test_sec_eda.pdf
    python report/test_section.py sec_models     # P2
    python report/test_section.py sec_hmm        # P3
    python report/test_section.py sec_api        # P4

The section reads the same result files as the full report; a missing file raises an error that
names it, which tells you which step still has to be run. Test PDFs are gitignored.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("sec_eda", "sec_models", "sec_hmm", "sec_api"):
        sys.exit(__doc__)
    mod = importlib.import_module(f"report.{sys.argv[1]}")
    story: list = []
    mod.build(story)
    out = Path(__file__).resolve().parent / f"_test_{sys.argv[1]}.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm).build(story)
    print("wrote", out)


if __name__ == "__main__":
    main()

"""The committed public master rebuilds the committed public paper byte for byte."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_master_reproduces_public_paper(tmp_path):
    out = tmp_path / "paper_public.qmd"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_paper.py"),
                    str(ROOT / "docs/writeup/paper_source_public.qmd"), str(out), "--mode", "public"],
                   check=True, capture_output=True)
    assert out.read_bytes() == (ROOT / "docs/writeup/paper_public.qmd").read_bytes()


def test_public_master_has_no_local_markers():
    text = (ROOT / "docs/writeup/paper_source_public.qmd").read_text()
    assert "<!-- LOCAL -->" not in text and "<!-- PUBLIC -->" not in text


def test_strip_unused_icons_keeps_used_rules_only(tmp_path):
    """the rendered HTML keeps only the icon-font rules the page uses (Quarto embeds about 2,000 glyph rules)."""
    src = tmp_path / "in.html"
    src.write_text('<style>.bi-clipboard::before { content: "\\f1"; }\n.bi-zzz-unused::before { content: "\\f2"; }\n</style>'
                   '<button class="bi bi-clipboard"></button>')
    out = tmp_path / "out.html"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_paper.py"), str(src), str(out), "--strip-unused-icons"], check=True,
                   capture_output=True)
    html = out.read_text()
    assert ".bi-clipboard::before" in html and "zzz-unused" not in html


def test_committed_public_html_has_no_unused_icon_rules():
    html = (ROOT / "docs/writeup/paper_public.html").read_text()
    assert "bi-zoom-in::before" not in html  # a glyph the paper never shows

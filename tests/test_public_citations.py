"""every source cited in reader-facing text (public paper, public draft, README) is either a file of this repository or carries
the marker "not included in this repository". In the development repository every cited path exists anyway; in the public tree the test
fails on any citation of an unpublished file."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARK = "not included in this repository"
PATH_RE = re.compile(r"(?<![\w/])((?:docs|results|scripts|configs|src|app|tests|site|demo_data)/[\w./*-]*\w)")


def _readme() -> Path | None:
    for p in (ROOT / "README.md", ROOT / "scripts/release/public_overlay/README.md"):
        if p.is_file() and "Cinematic Grammar Tokenizer" in p.read_text():
            return p
    return None


def dangling(text: str) -> list[str]:
    """Cited repository paths in `text` that do not exist (wildcards and the marker excepted)."""
    out = []
    for m in PATH_RE.finditer(text):
        rel = m.group(1).replace("\\", "")
        if "*" not in rel and not (ROOT / rel).exists():
            out.append(rel)
    return sorted(set(out))


def test_paper_reference_list_points_at_files_or_the_marker():
    text = (ROOT / "docs/writeup/paper_public.qmd").read_text()
    refs = re.findall(r"^\[\d+\] \*([^*]+)\*", text, re.MULTILINE)
    assert refs
    bad = [r for r in refs if r != MARK and not (ROOT / r.replace("\\", "")).exists()]
    assert bad == []


def test_reader_facing_text_cites_no_missing_path():
    files = [ROOT / "docs/writeup/paper_public.qmd", ROOT / "docs/writeup/draft_public.md"]
    if _readme():
        files.append(_readme())
    assert {str(f.relative_to(ROOT)): dangling(f.read_text()) for f in files} == {str(f.relative_to(ROOT)): [] for f in files}


def test_guard_catches_a_missing_path():
    assert dangling("see [docs/no_such_file_here.md]") == ["docs/no_such_file_here.md"]
    assert dangling(f"[{MARK}] and docs/writeup/paper_public.qmd") == []

"""Build the public or the local paper (.qmd) from one master source.

Usage: python scripts/build_paper.py SOURCE.qmd OUT.qmd --mode public|local [--figdir DIR] [--css PATH]
       python scripts/build_paper.py MASTER.qmd OUT.qmd --emit-master public

Public build from committed files only (the committed master has no local text):
  python3 scripts/build_paper.py docs/writeup/paper_source_public.qmd docs/writeup/paper_public.qmd --mode public
  cd docs/writeup && quarto render paper_public.qmd --to html      # Quarto 1.8 (/usr/local/bin/quarto); embed-resources
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
      --virtual-time-budget=8000 --print-to-pdf=docs/writeup/paper_public.pdf file://$PWD/docs/writeup/paper_public.html
Local build (gitignored; never commit): the full master data/reports/paper_source.qmd ->
  python3 scripts/build_paper.py data/reports/paper_source.qmd data/reports/paper_full.qmd --mode local \
      --figdir ../../docs/writeup/figures --css ../../docs/writeup/report.css
  then render and print data/reports/paper_full.html the same way.
After rendering, drop the icon-font rules of Quarto's embedded Bootstrap Icons CSS that the page never uses (about 2,000 glyph names):
  python3 scripts/build_paper.py docs/writeup/paper_public.html docs/writeup/paper_public.html --strip-unused-icons
Refresh the committed public master after editing the full master:
  python3 scripts/build_paper.py data/reports/paper_source.qmd docs/writeup/paper_source_public.qmd --emit-master public

The master source is kept in the gitignored data/reports/ folder because it holds local-only text.
- Blocks between lines `<!-- LOCAL -->` and `<!-- /LOCAL -->` are kept only in the local build;
  blocks between `<!-- PUBLIC -->` and `<!-- /PUBLIC -->` only in the public build.
- `{{key}}` is a citation; keys are defined in a `<!-- REFS ... -->` block (`key | path | note`).
  Citations are numbered in order of first use in the built text, and a "References" section that
  names the source file of every number is appended. Unknown keys are an error.
- `FIGDIR/` and `CSSPATH` are replaced by --figdir and --css.
This script contains no film names; it only filters and numbers.
"""
import argparse
import re
import sys
from pathlib import Path

CITE = re.compile(r"\{\{([a-z0-9_]+)\}\}")
MARKERS = {"LOCAL", "PUBLIC"}


def strip_blocks(text: str, mode: str) -> str:
    keep_tag = "LOCAL" if mode == "local" else "PUBLIC"
    out, stack = [], []
    for line in text.splitlines():
        m = re.fullmatch(r"\s*<!-- (/?)(LOCAL|PUBLIC) -->\s*", line)
        if m:
            closing, tag = m.group(1) == "/", m.group(2)
            if closing:
                if not stack or stack[-1] != tag:
                    raise SystemExit(f"unbalanced marker: {line!r}")
                stack.pop()
            else:
                if stack:
                    raise SystemExit(f"nested marker not allowed: {line!r}")
                stack.append(tag)
            continue
        if stack and stack[-1] != keep_tag:
            continue
        out.append(line)
    if stack:
        raise SystemExit(f"unclosed marker: {stack}")
    return "\n".join(out) + "\n"


def parse_refs(text: str) -> tuple[str, dict[str, tuple[str, str]]]:
    m = re.search(r"<!-- REFS\n(.*?)\n-->\n?", text, re.DOTALL)
    if not m:
        raise SystemExit("no REFS block")
    refs = {}
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        key, path, note = (p.strip() for p in line.split("|", 2))
        if key in refs:
            raise SystemExit(f"duplicate ref key {key}")
        refs[key] = (path, note)
    return text[: m.start()] + text[m.end():], refs


def emit_master(text: str, out: Path) -> int:
    """Master for the public build: LOCAL blocks dropped, PUBLIC markers unwrapped, REFS limited to used keys."""
    body, refs = parse_refs(text)
    body = strip_blocks(body, "public")
    used = set(CITE.findall(body))
    missing = used - set(refs)
    if missing:
        raise SystemExit(f"unknown citation keys {sorted(missing)}")
    block = "\n".join(f"{k} | {refs[k][0]} | {refs[k][1]}" for k in refs if k in used)
    out.write_text(body.rstrip("\n") + "\n\n<!-- REFS\n" + block + "\n-->\n")
    print(f"public master: {len(used)} reference keys -> {out}", file=sys.stderr)
    return 0


ICON_RULE = re.compile(r"\.bi-([a-z0-9-]+)::before\s*\{[^}]*\}\n?")


def strip_unused_icons(html: str) -> tuple[str, int]:
    """Remove every `.bi-<name>::before { content: "..."; }` rule whose class `bi-<name>` is not used elsewhere in the page; returns
    (html, rules removed). The page keeps the rules of the icons it shows (e.g. the copy button of code blocks)."""
    rest = ICON_RULE.sub("", html)
    used = set(re.findall(r"\bbi-([a-z0-9-]+)", rest))
    removed = 0

    def drop(m: re.Match) -> str:
        nonlocal removed
        if m.group(1) in used:
            return m.group(0)
        removed += 1
        return ""
    return ICON_RULE.sub(drop, html), removed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out")
    ap.add_argument("--mode", choices=["public", "local"])
    ap.add_argument("--emit-master", choices=["public"], default=None,
                    help="write the master with LOCAL blocks removed (citations and used REFS kept)")
    ap.add_argument("--strip-unused-icons", action="store_true", help="post-process a rendered HTML file (SOURCE) into OUT")
    ap.add_argument("--figdir", default="figures")
    ap.add_argument("--css", default="report.css")
    a = ap.parse_args(argv)
    if a.strip_unused_icons:
        html, n = strip_unused_icons(Path(a.source).read_text())
        Path(a.out).write_text(html)
        print(f"removed {n} unused icon rules -> {a.out}", file=sys.stderr)
        return 0
    if (a.mode is None) == (a.emit_master is None):
        ap.error("give exactly one of --mode or --emit-master")
    text = Path(a.source).read_text()
    if a.emit_master:
        return emit_master(text, Path(a.out))
    text, refs = parse_refs(text)
    text = strip_blocks(text, a.mode)
    order: dict[str, int] = {}

    def num(m: re.Match) -> str:
        key = m.group(1)
        if key not in refs:
            raise SystemExit(f"unknown citation key {key}")
        order.setdefault(key, len(order) + 1)
        return str(order[key])

    text = CITE.sub(num, text)
    text = re.sub(r"\[(\d+(?:, \d+)+)\]",
                  lambda m: "[" + ", ".join(str(n) for n in sorted({int(x) for x in m.group(1).split(", ")})) + "]", text)
    text = text.replace("FIGDIR/", a.figdir.rstrip("/") + "/").replace("CSSPATH", a.css)
    lines = ["", "## References {.unnumbered .unlisted}", "",
             ("Each number in square brackets in the text refers to the source below; paths are relative "
              "to the repository, and sources marked as not included are project records that are not published."), ""]
    for key, n in sorted(order.items(), key=lambda kv: kv[1]):
        path, note = refs[key]
        lines.append(f"[{n}] *{path.replace('*', chr(92) + '*')}*" + (f". {note}" if note else "") + "\n")
    text = text.rstrip("\n") + "\n" + "\n".join(lines)
    if any(f"<!-- {t} -->" in text for t in MARKERS):
        raise SystemExit("marker left in output")
    Path(a.out).write_text(text)
    print(f"{a.mode}: {len(order)} references, {len(text.split())} tokens -> {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

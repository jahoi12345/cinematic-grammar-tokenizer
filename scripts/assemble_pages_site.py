"""Assemble the GitHub Pages site: the paper at the site root and the static demo under app/. Standard library only (no install needed).

Usage: python3 scripts/assemble_pages_site.py [--out _site] [--paper site/index.html] [--pdf site/paper_public.pdf]
       [--app/static] [--data-root demo_data]

Result:
  OUT/index.html            the paper (with one link to the demo added after <body>)
  OUT/paper_public.pdf      the paper as a PDF
  OUT/app/                  app/static/ (index.html, app.css, app.js, data/)
  OUT/app/keyframes/<film>/ the keyframes that app/static/data/<film>/shots.json names, copied from DATA_ROOT/data/interim/<film>/keyframes/
Only films listed in app/static/data/site.json (written by scripts/build_static_app.py, public-domain films only) are copied, and only
plain keyframe file names; a missing keyframe fails the build. OUT is replaced."""
import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = re.compile(r"^\d{5}_(start|mid|end)\.jpg$")
FILM = re.compile(r"^[a-z0-9_]+$")
APP_LINK = ('<div style="font:500 14px/1.4 system-ui,sans-serif;padding:8px 16px;background:#146964;color:#FFFFFF;text-align:center">'
            'Explore the token rows shot by shot in the <a href="app/" target="_top" style="color:#FFFFFF;text-decoration:underline">interactive demo</a></div>')


class AssemblyError(Exception):
    pass


def paper_with_link(html: str) -> str:
    m = re.search(r"<body\b[^>]*>", html, flags=re.IGNORECASE)
    if not m:
        raise AssemblyError("the paper has no <body> tag")
    return html[:m.end()] + APP_LINK + html[m.end():]


def keyframe_names(shots: dict) -> list[str]:
    cols = shots.get("columns", {})
    names = sorted({n for k in ("kf_start", "kf_mid", "kf_end") for n in cols.get(k, []) if n})
    bad = [n for n in names if not NAME.match(n)]
    if bad:
        raise AssemblyError(f"unexpected keyframe names: {bad[:3]}")
    return names


def assemble(out: Path, paper: Path, pdf: Path, app: Path, data_root: Path) -> dict:
    site = json.loads((app / "data/site.json").read_text(encoding="utf-8"))
    if site.get("public") is not True:
        raise AssemblyError("site.json is not a public build")
    films = [f["id"] for f in site.get("films", [])]
    for f in films:
        if not FILM.match(f):
            raise AssemblyError(f"unexpected film id {f!r}")
    if out.exists():
        shutil.rmtree(out)
    (out / "app").mkdir(parents=True)
    (out / "index.html").write_text(paper_with_link(paper.read_text(encoding="utf-8")), encoding="utf-8")
    shutil.copyfile(pdf, out / "paper_public.pdf")
    for name in ("index.html", "app.css", "app.js"):
        shutil.copyfile(app / name, out / "app" / name)
    shutil.copytree(app / "data", out / "app" / "data")
    counts = {}
    for film in films:
        shots = json.loads((app / "data" / film / "shots.json").read_text(encoding="utf-8"))
        src = data_root / "data/interim" / film / "keyframes"
        dst = out / "app/keyframes" / film
        dst.mkdir(parents=True)
        names = keyframe_names(shots)
        for n in names:
            p = src / n
            if not p.is_file() or p.is_symlink():
                raise AssemblyError(f"missing keyframe {film}/{n}")
            shutil.copyfile(p, dst / n)
        counts[film] = len(names)
    return {"films": films, "keyframes": counts}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="_site")
    ap.add_argument("--paper", default="site/index.html")
    ap.add_argument("--pdf", default="site/paper_public.pdf")
    ap.add_argument("--app", default="app/static")
    ap.add_argument("--data-root", default="demo_data")
    a = ap.parse_args(argv)
    try:
        res = assemble(Path(a.out), Path(a.paper), Path(a.pdf), Path(a.app), Path(a.data_root))
    except (AssemblyError, OSError, ValueError) as e:
        print(f"FAILED: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())

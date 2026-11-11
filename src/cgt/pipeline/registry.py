"""The film registry: the committed `configs/films.yaml` merged with the gitignored local overlay `configs/films.local.yaml`.
An id present in both files (or twice in one file) is an error, never a silent override. Overlay films are local-only: public_domain defaults to false and
nothing derived from them is published, whatever their flag says."""
import os
from pathlib import Path

import yaml

MAIN = "configs/films.yaml"
OVERLAY = "configs/films.local.yaml"
HEADER = ("# LOCAL ONLY (gitignored, the recorded rule): ad-hoc inputs registered by scripts/tokenize_video.py.\n"
          "# Never commit this file. configs/films.yaml stays the only committed film list; an id clash with it is an error.\n")


class RegistryClash(ValueError):
    pass


def _read(p: Path) -> list[dict]:
    if not p.exists():
        return []
    data = yaml.safe_load(p.read_text()) or {}
    return list(data.get("films") or [])


def read_overlay(root: Path) -> list[dict]:
    return _read(Path(root) / OVERLAY)


def write_overlay(root: Path, entries: list[dict]) -> Path:
    p = Path(root) / OVERLAY
    p.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in e.items() if k != "registry"} for e in entries]
    tmp = p.with_suffix(".tmp")
    tmp.write_text(HEADER + yaml.safe_dump({"films": clean}, sort_keys=False, allow_unicode=True))
    os.replace(tmp, p)
    return p


def load_films(root: Path, pending: dict | None = None) -> dict[str, dict]:
    """{id: entry copy with `registry` = "main" | "overlay"}. `pending` is one overlay entry not yet written (dry run); it clashes like a written one."""
    out: dict[str, dict] = {}
    overlay = read_overlay(root) + ([pending] if pending else [])
    for src, entries in (("main", _read(Path(root) / MAIN)), ("overlay", overlay)):
        for f in entries:
            fid = f["id"]
            if fid in out:
                if pending is not None and f is pending and out[fid]["registry"] == "overlay" and out[fid].get("sha256") == pending.get("sha256"):
                    out[fid] = {**out[fid], **pending, "registry": "overlay"}  # the pending entry is the update of its own overlay entry
                    continue
                raise RegistryClash(f"film id '{fid}' is registered twice ({out[fid]['registry']} and {src}); an id clash is an error, nothing is overridden "
                                    f"(rename it with --id or remove the duplicate from {OVERLAY})")
            e = {**f, "registry": src}
            if src == "overlay":
                e["public_domain"] = bool(f.get("public_domain", False))
            out[fid] = e
    return out


def is_overlay(entry: dict) -> bool:
    return entry.get("registry") == "overlay"


def local_only_ids(root: Path) -> set[str]:
    """Films whose outputs must never be published: owned films of films.yaml and every overlay film."""
    return {fid for fid, e in load_films(root).items() if is_overlay(e) or not e.get("public_domain")}

"""Download the OFFICIAL public CameraBench test set files from the Hugging Face dataset repo `syCen/CameraBench` (EVALUATION-ONLY data; approved by the author 2026-10-02;
no yt-dlp, no scraping). Lists the repo tree with the official API, downloads every file under `videos_gif/` plus `videos.csv`, `LICENSE`, `README.md`, `test.jsonl`, and verifies each
file's SHA-256 against the LFS object id (or git blob size for non-LFS files) from the same API. Label files are copied/downloaded and hashed as BYTES, never parsed here.
Resumable and idempotent. Writes results/phase4/camerabench_download.json. Usage: uv run python scripts/p4_download_camerabench.py [--an earlier change]"""
import argparse
import hashlib
import json
import shutil
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/cache/p4/camerabench"
API = "https://huggingface.co/api/datasets/syCen/CameraBench/tree/main"
RESOLVE = "https://huggingface.co/datasets/syCen/CameraBench/resolve/main/"
H = {"User-Agent": "cgt-camerabench-eval-download"}
WANT_TOP = {"videos.csv", "LICENSE", "README.md", "test.jsonl"}


def get(url: str):
    return urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=180)


def listing() -> list[dict]:
    out, url = [], API + "?recursive=true&expand=true"
    while url:
        r = get(url)
        out += json.load(r)
        link = r.headers.get("Link", "")
        url = link.split("<")[1].split(">")[0] if 'rel="next"' in link else None
    return [e for e in out if e["type"] == "file" and (e["path"].startswith("videos_gif/") and e["path"].endswith(".gif") or e["path"] in WANT_TOP)]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def fetch(e: dict) -> tuple[str, str, int]:
    dst = DEST / e["path"]
    want = (e.get("lfs") or {}).get("oid")
    size = e.get("size", 0)
    if dst.exists() and dst.stat().st_size == size and (want is None or sha256(dst) == want):
        return e["path"], "cached", size
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(".part_" + dst.name)
    for attempt in range(4):
        try:
            with get(RESOLVE + urllib.parse.quote(e["path"])) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            if tmp.stat().st_size == size and (want is None or sha256(tmp) == want):
                tmp.replace(dst)
                return e["path"], "downloaded", size
        except Exception:  # noqa: BLE001 - retried, then reported as failed
            time.sleep(2 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return e["path"], "FAILED", size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    files = listing()
    gifs = [e for e in files if e["path"].startswith("videos_gif/")]
    print(f"{len(files)} files listed ({len(gifs)} GIFs, {sum(e['size'] for e in gifs) / 1e9:.3f} GB)", flush=True)
    t0, res = time.time(), {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(fetch, e) for e in files]
        for i, f in enumerate(as_completed(futs), 1):
            p, st, sz = f.result()
            res[p] = (st, sz)
            if i % 50 == 0:
                print(f"  {i}/{len(files)} {time.time() - t0:.0f}s", flush=True)
    bad = [p for p, (st, _) in res.items() if st == "FAILED"]
    rep = {"listed_files": len(files), "gifs_listed": len(gifs), "gifs_listed_bytes": sum(e["size"] for e in gifs), "downloaded": sum(st == "downloaded" for st, _ in res.values()),
           "already_cached": sum(st == "cached" for st, _ in res.values()), "failed": bad, "total_bytes_on_disk": sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()),
           "gifs_on_disk": len(list((DEST / "videos_gif").glob("*.gif"))), "checksum": "every file's SHA-256 equals the LFS oid from the official API tree (non-LFS files: size only)",
           "seconds_this_run": round(time.time() - t0, 1), "source": "https://huggingface.co/datasets/syCen/CameraBench (CC-BY-4.0 annotations; videos remain third-party content)"}
    (ROOT / "results/phase4/camerabench_download.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if bad:
        raise SystemExit(f"{len(bad)} files failed")


if __name__ == "__main__":
    main()

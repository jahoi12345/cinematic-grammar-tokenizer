"""Keyboard-driven labeling tool (stdlib only). Blind: never shows model predictions.
  uv run python app/label/server.py cuts --film night_of_the_living_dead --threshold 12   # 50-cut check
  uv run python app/label/server.py dev | gold                                            # shot labels
  .venv/bin/python app/label/server.py gold-mini                                          # 60 NOTLD gold shots: size, angle, movement
Open http://localhost:8765 (binds 127.0.0.1 only). Labels autosave to data/**/labels*.json (gitignored); re-running resumes.
gold-mini (docs/gold_mini_eval_plan.md) writes ONLY data/gold/labels_mini_v1.json (or --labels-out); --root reads films/keyframes from another checkout."""
import argparse
import http.server
import json
import os
import sys
import urllib.parse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gold_mini as GM

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = yaml.safe_load((ROOT / "configs/schema.yaml").read_text())["attributes"]
# key, prompt, options. '?' (unsure) is always available and stored as null.
# "level"/"none" are labeler-only extras not in schema.yaml (eye-level angle; no person).
FIELDS = [
    ("size", "Shot size", SCHEMA["size"]["values"]),
    ("angle", "Camera angle", ["level"] + SCHEMA["angle"]["values"]),
    ("dof", "Depth of field", SCHEMA["dof"]["values"]),
    ("key", "Lighting key", SCHEMA["key"]["values"]),
    ("people", "People visible (4 = 4+)", ["0", "1", "2", "3", "4"]),
    ("facing", "Main person faces", ["away", "camera", "profile"]),       # skipped when people == 0
    ("light_dir", "Light on main face from", ["front", "side", "back"]),  # skipped when people == 0
]

def load_items(mode, a):
    if mode == "gold-mini":  # blind: the items carry only the shot's place in the film and its keyframe paths
        df = pd.read_parquet(ROOT / f"data/processed/{GM.FILM}_shots.parquet", columns=["shot_id", "kf_start", "kf_mid", "kf_end"]).set_index("shot_id")
        items = [{"id": f"{s['film']}:{s['shot_id']}", "film": s["film"], "shot_id": s["shot_id"], "start_s": s["start_s"], "end_s": s["end_s"],
                  "imgs": [str(df.loc[s["shot_id"], c]) for c in ("kf_start", "kf_mid", "kf_end")]} for s in GM.load_selection(ROOT)]
        out = Path(a.labels_out) if getattr(a, "labels_out", None) else ROOT / GM.LABELS_REL
        return items, out
    if mode == "cuts":
        shots = json.loads((ROOT / f"data/interim/{a.film}/cuts_t{a.threshold:g}.json").read_text())
        rng = np.random.default_rng(0)
        idx = sorted(int(i) for i in rng.choice(np.arange(1, len(shots)), size=a.n, replace=False))
        items = [{"id": f"{a.film}:{i}", "film": a.film, "t": shots[i][0], "idx": i} for i in idx]
        return items, ROOT / f"data/interim/{a.film}/spotcheck_labels_t{a.threshold:g}.json"
    sel = json.loads((ROOT / f"data/{mode}/shots.json").read_text())
    items, tables = [], {}
    for s in sel:
        df = tables.setdefault(s["film"], pd.read_parquet(ROOT / f"data/processed/{s['film']}_shots.parquet").set_index("shot_id"))
        r = df.loc[s["shot_id"]]
        items.append({"id": f"{s['film']}:{s['shot_id']}", "film": s["film"], "start_s": s["start_s"], "end_s": s["end_s"],
                      "imgs": [str(r[c]) for c in ("kf_start", "kf_mid", "kf_end")]})
    rng = np.random.default_rng(1); rng.shuffle(items)
    return items, ROOT / f"data/{mode}/labels.json"

def video_path(film):
    """Registry lookup; unknown film ids return None (callers answer 404) instead of raising."""
    films = yaml.safe_load((ROOT / "configs/films.yaml").read_text())["films"]
    f = next((f for f in films if f["id"] == film), None)
    return str(ROOT / f["path"]) if f else None

def cut_jpeg(film, t):
    cap = cv2.VideoCapture(video_path(film)); fps = cap.get(cv2.CAP_PROP_FPS)
    tiles = []
    for tt in (t - 3 / fps, t + 1 / fps):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(tt, 0) * 1000); ok, f = cap.read()
        tiles.append(cv2.resize(f, (640, 480)) if ok else np.zeros((480, 640, 3), np.uint8))
    cap.release()
    return cv2.imencode(".jpg", np.hstack(tiles))[1].tobytes()

class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def send(self, body, ctype="application/json"):
        self.send_response(200); self.send_header("Content-Type", ctype); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path == "/":
            page = "gold_mini.html" if S["mode"] == "gold-mini" else "index.html"
            return self.send((Path(__file__).parent / page).read_bytes(), "text/html")
        if u.path == "/api/state" and S["mode"] == "gold-mini":
            return self.send(json.dumps(gold_mini_state()).encode())
        if u.path == "/api/state":
            return self.send(json.dumps({"mode": S["mode"], "items": S["items"], "labels": S["labels"], "fields": FIELDS}).encode())
        if S["mode"] == "gold-mini" and u.path in ("/img", "/video", "/cut"):
            return self.get_gold_mini_media(u.path, q)
        if u.path == "/img":
            rel = os.path.normpath(q["p"][0])   # relative to ROOT; symlinked data dirs are fine
            p = ROOT / rel
            if not rel.startswith("data/interim/") or p.suffix != ".jpg": return self.send_error(403)
            return self.send(p.read_bytes(), "image/jpeg")
        if u.path in ("/video", "/cut"):
            vp = video_path(q.get("film", [""])[0])
            if vp is None or not os.path.exists(vp): return self.send_error(404)
            if u.path == "/video": return self.send_video(vp)
            try: t = float(q["t"][0])
            except (KeyError, ValueError): return self.send_error(400)
            return self.send(cut_jpeg(q["film"][0], t), "image/jpeg")
        self.send_error(404)
    def send_video(self, path):
        """Serve the film with HTTP Range support so the browser can seek (local use only)."""
        size = os.path.getsize(path); start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-"); start = int(a or 0); end = min(int(b) if b else start + (4 << 20) - 1, size - 1)
        try:
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", "video/mp4"); self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if rng: self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start); left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk: break
                    self.wfile.write(chunk); left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if S["mode"] == "gold-mini":
            return self.post_gold_mini(body)
        S["labels"][body["id"]] = body["label"]
        tmp = S["path"].with_suffix(".tmp"); tmp.write_text(json.dumps(S["labels"], indent=1)); tmp.replace(S["path"])
        self.send(b"{}")

    def get_gold_mini_media(self, path, q):
        """Blind media allowlist: only the exact keyframe paths of the selected shots and NOTLD's film file; everything else 403
        (other interim jpgs such as p3c_review/sheet_*.jpg print model predictions)."""
        if path == "/img":
            rel = q.get("p", [""])[0]
            if rel not in S["allowed_imgs"]:
                return self.send_error(403)
            return self.send((ROOT / rel).read_bytes(), "image/jpeg")
        if path == "/video" and q.get("film", [""]) == [GM.FILM]:
            vp = video_path(GM.FILM)
            if vp is None or not os.path.exists(vp): return self.send_error(404)
            return self.send_video(vp)
        return self.send_error(403)

    def post_gold_mini(self, body):
        """Validate one shot's label, stamp it, and write the whole file atomically (autosave after every shot)."""
        by_id = {it["id"]: it for it in S["items"]}
        it = by_id.get(body.get("id"))
        if it is None:
            return self.send_error(400, "unknown shot")
        if any(f not in body for f in GM.FIELDS):  # "can't tell" is an explicit null, never a missing field
            return self.send_error(400, "size, angle and movement are all required")
        try:
            rec = GM.new_record(it["shot_id"], body.get("size"), body.get("angle"), body.get("movement"), S["vocab"])
        except ValueError as e:
            return self.send_error(400, str(e))
        S["labels"][it["shot_id"]] = rec
        GM.write_labels(S["path"], S["labels"], [x["shot_id"] for x in S["items"]])
        self.send(json.dumps({"ok": True, "n": len(S["labels"])}).encode())

def gold_mini_state():
    """Everything the page needs and nothing else: no prediction, token row or table column beyond keyframe paths and the shot's place in the film."""
    labels = {f"{GM.FILM}:{k}": {f: v[f] for f in GM.FIELDS} for k, v in S["labels"].items()}
    return {"mode": "gold-mini", "items": S["items"], "labels": labels, "vocab": S["vocab"], "move_help": GM.MOVE_HELP,
            "tool_version": GM.TOOL_VERSION, "n_total": len(S["items"])}

def init_state(mode, a):
    items, path = load_items(mode, a)
    S.clear()
    if mode == "gold-mini":
        voc = GM.vocab(ROOT)
        labels = {k: GM.validate(v, voc) for k, v in GM.read_labels(path).items()}
        S.update(mode=mode, items=items, path=path, labels=labels, vocab=voc, allowed_imgs=frozenset(p for it in items for p in it["imgs"]))
    else:
        S.update(mode=mode, items=items, path=path, labels=json.loads(path.read_text()) if path.exists() else {})

S = {}
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cuts", "dev", "gold", "gold-mini"])
    ap.add_argument("--root", default=None, help="checkout whose data/ and configs/ are read (default: this checkout); gold-mini writes only the labels file")
    ap.add_argument("--labels-out", default=None, help="gold-mini only: labels file (default <root>/data/gold/labels_mini_v1.json)")
    ap.add_argument("--film", default="night_of_the_living_dead"); ap.add_argument("--threshold", type=float, default=12.0)
    ap.add_argument("--n", type=int, default=50); ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    if a.root:
        ROOT = Path(a.root).resolve()
    init_state(a.mode, a)
    print(f"{len(S['items'])} items, {len(S['labels'])} already labeled -> http://127.0.0.1:{a.port}  (labels: {S['path']})")
    http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()

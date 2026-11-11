"""Per-shot sound class (dialogue / score / silence / ambient) from an AudioSet classifier (AST)."""
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from transformers import pipeline

_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
_pipe = None


# v2 (finding F2, docs/p2_audio_drift_fix_2026-10-13.md): timestamp-aligned decode. v1 (`audio16k.npy`) concatenated decoded samples and
# dropped the timestamp gaps of the stream (NOTLD: 36 gaps of ~0.4 s, -12.4 s by 5,400 s), so the cache file is versioned and v1 is never reused.
CACHE_NAME = "audio16k_v2.npy"
# aresample async: pad timestamp gaps with silence (and trim overlaps) so that sample index = seconds * SR on the film timeline, from t = 0
ALIGN_FILTER = "aresample=async=1:first_pts=0"


def _container_seconds(video_path: str) -> float | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video_path)], capture_output=True, text=True, check=False)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def decode_audio(video_path: str, cache: Path) -> np.ndarray:
    """Whole soundtrack as 16 kHz mono float32 aligned to the film timeline (sample i = i / SR seconds), cached as .npy."""
    if cache.exists():
        return np.load(cache, mmap_mode="r")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video_path), "-vn", "-af", ALIGN_FILTER, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                       capture_output=True, check=False)
    if r.returncode != 0 and not r.stdout:
        raise subprocess.CalledProcessError(r.returncode, "ffmpeg", r.stdout, r.stderr)  # nothing decoded at all: still a failure
    a = np.frombuffer(r.stdout, dtype=np.float32)
    info = {}
    dur = _container_seconds(video_path)
    if dur is not None and len(a) > int((dur + 1.0) * SR):  # gap padding must never extend past the container (e.g. garbage timestamps in a damaged tail)
        info["truncated_from_seconds"] = len(a) / SR
        a = a[: int(dur * SR)]
    if r.returncode != 0 or info:  # damaged file (e.g. a corrupted tail): keep what decoded and record it next to the cache instead of failing the whole stage
        Path(str(cache) + ".decode.json").write_text(json.dumps({"ffmpeg_returncode": r.returncode, "seconds_decoded": len(a) / SR, **info,
                                                                 "stderr_tail": r.stderr.decode(errors="replace")[-400:]}))
    np.save(cache, a)
    return a


def _get_pipe():
    global _pipe
    if _pipe is None:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _pipe = pipeline("audio-classification", model=_MODEL, device=dev)
    return _pipe


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


# Sound grouping v2. v1 sent Music-family labels without the word
# "music" (Opera, Cello, Choir, Harp, ...) to ambient. v2 adds the explicit list below = every node of the AudioSet ontology's "Music"
# subtree (/m/04rlf and all descendants via child_ids, https://raw.githubusercontent.com/audioset/ontology/master/ontology.json) that
# is an AST label, spelled exactly as the model's id2label (150 of 189 nodes; the rest are abstract/blacklisted ontology nodes or
# Oboe, Bassoon, Birthday music, which are not AST classes). The walk is recorded in tests/fixtures/audioset_music_subtree.json.
# The lists live in this file on purpose: the p2 audio stage key hashes the source of audio.py, so any change here recomputes the stage.
# a recorded decision Amendment 1: score = Music subtree MINUS EXCLUDED_LABELS (sound-effect / tool bells, stay ambient) PLUS SINGING_LABELS
# (singing labels under "Human voice", outside the Music subtree). GROUPS_VERSION stays 2: v2 was never released nor run on a film.
GROUPS_VERSION = 2
MUSIC_SUBTREE_LABELS = (
    "Music", "Musical instrument", "Plucked string instrument", "Keyboard (musical)", "Percussion",
    "Orchestra", "Brass instrument", "Bowed string instrument",
    "Wind instrument, woodwind instrument", "Harp", "Choir", "Bell", "Harmonica", "Accordion",
    "Bagpipes", "Didgeridoo", "Shofar", "Theremin", "Singing bowl",
    "Scratching (performance technique)", "Pop music", "Hip hop music", "Rock music",
    "Rhythm and blues", "Soul music", "Reggae", "Country", "Funk", "Folk music",
    "Middle Eastern music", "Jazz", "Disco", "Classical music", "Electronic music",
    "Music of Latin America", "Blues", "Music for children", "New-age music", "Vocal music",
    "Music of Africa", "Christian music", "Music of Asia", "Ska", "Traditional music",
    "Independent music", "Song", "Background music", "Theme music", "Jingle (music)",
    "Soundtrack music", "Lullaby", "Video game music", "Christmas music", "Dance music",
    "Wedding music", "Happy music", "Funny music", "Sad music", "Tender music", "Exciting music",
    "Angry music", "Scary music", "Guitar", "Banjo", "Sitar", "Mandolin", "Zither", "Ukulele",
    "Piano", "Organ", "Synthesizer", "Harpsichord", "Drum kit", "Drum", "Cymbal", "Cowbell",
    "Wood block", "Tambourine", "Rattle (instrument)", "Gong", "Tubular bells", "Mallet percussion",
    "French horn", "Trumpet", "Trombone", "String section", "Violin, fiddle", "Cello",
    "Double bass", "Flute", "Saxophone", "Clarinet", "Church bell", "Jingle bell", "Bicycle bell",
    "Tuning fork", "Chime", "Change ringing (campanology)", "Beatboxing", "Heavy metal",
    "Punk rock", "Grunge", "Progressive rock", "Rock and roll", "Psychedelic rock", "Swing music",
    "Bluegrass", "Opera", "House music", "Techno", "Dubstep", "Drum and bass", "Electronica",
    "Electronic dance music", "Ambient music", "Trance music", "Salsa music", "Flamenco",
    "A capella", "Chant", "Afrobeat", "Gospel music", "Carnatic music", "Music of Bollywood",
    "Electric guitar", "Bass guitar", "Acoustic guitar", "Steel guitar, slide guitar",
    "Tapping (guitar technique)", "Strum", "Electric piano", "Electronic organ", "Hammond organ",
    "Sampler", "Drum machine", "Snare drum", "Bass drum", "Timpani", "Tabla", "Hi-hat", "Maraca",
    "Marimba, xylophone", "Glockenspiel", "Vibraphone", "Steelpan", "Pizzicato", "Wind chime",
    "Mantra", "Rimshot", "Drum roll",
)
# outside the Music subtree (AudioSet: Human voice > Singing, and Human voice): still score for a film's sound slot
SINGING_LABELS = ("Male singing", "Female singing", "Child singing", "Synthetic singing", "Rapping", "Humming", "Yodeling")
# inside the Music subtree but sound effects or tools: stay ambient
EXCLUDED_LABELS = ("Bicycle bell", "Church bell", "Wind chime", "Tuning fork")
_SCORE_LABELS = (frozenset(MUSIC_SUBTREE_LABELS) - frozenset(EXCLUDED_LABELS)) | frozenset(SINGING_LABELS)


def _group(label: str) -> str:
    l = label.lower()
    # dialogue first (unchanged from v1): it wins over any music rule
    if "speech" in l or l in {"conversation", "narration, monologue", "shout", "screaming", "whispering"}:
        return "dialogue"
    if "music" in l or l in {"piano", "organ", "violin, fiddle", "orchestra", "synthesizer", "singing"}:
        return "score"
    if label in _SCORE_LABELS:  # v2: AudioSet Music subtree - EXCLUDED_LABELS + SINGING_LABELS
        return "score"
    return "ambient"


def classify_clip(x: np.ndarray, silence_db: float = -55.0, max_s: float = 10.0) -> dict:
    db = rms_db(x) if len(x) else -120.0
    if len(x) < SR // 4 or db < silence_db:
        return {"sound": "silence", "sound_db": db, "sound_top": None, "sound_top_p": None,
                "sound_p_speech": None, "sound_p_music": None}
    mid = len(x) // 2
    half = int(max_s * SR / 2)
    clip = np.asarray(x[max(0, mid - half): mid + half], dtype=np.float32)
    preds = _get_pipe()({"raw": clip, "sampling_rate": SR}, top_k=10)
    score = {"dialogue": 0.0, "score": 0.0, "ambient": 0.0}
    best = {"dialogue": 0.0, "score": 0.0, "ambient": 0.0}
    for p in preds:
        g = _group(p["label"])
        score[g] += p["score"]
        best[g] = max(best[g], p["score"])
    return {"sound": max(score, key=score.get), "sound_db": db,
            "sound_top": preds[0]["label"], "sound_top_p": float(preds[0]["score"]),
            # additive evidence columns (do not feed the class decision): strongest label of each family in the top 10
            "sound_p_speech": float(best["dialogue"]), "sound_p_music": float(best["score"])}

"""Where the sibling working copies (git worktrees) of this repository live.

Some experiment configs read frozen inputs (adapters, caches, predictions) from other working copies of the same repository. Their location
is not hard-coded: `CGT_WORKTREES` overrides it, otherwise it is read from git's own worktree registry (`<common git dir>/worktrees/*/gitdir`,
the most common parent of the registered worktrees). Outside a git checkout (e.g. an exported copy) it falls back to `<repo>/worktrees`, which
normally does not exist, so the inputs are simply missing and the tests that need them skip.
"""
import os
from collections import Counter
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _common_git_dir(repo: Path) -> Path | None:
    dot = repo / ".git"
    if dot.is_dir():
        return dot
    if dot.is_file():
        line = dot.read_text().strip()
        if line.startswith("gitdir:"):
            gd = Path(line.split(":", 1)[1].strip())
            return gd.parent.parent if gd.parent.name == "worktrees" else gd
    return None


@cache
def worktrees_dir(repo: Path = REPO) -> Path:
    env = os.environ.get("CGT_WORKTREES")
    if env:
        return Path(env)
    common = _common_git_dir(Path(repo))
    parents = Counter()
    if common is not None:
        for g in (common / "worktrees").glob("*/gitdir"):
            try:
                parents[Path(g.read_text().strip()).parent.parent] += 1
            except OSError:
                continue
    if parents:
        return parents.most_common(1)[0][0]
    return Path(repo) / "worktrees"


def main_checkout(repo: Path = REPO) -> Path:
    """The main working copy (the parent of the common git dir), else the repo itself."""
    common = _common_git_dir(Path(repo))
    return common.parent if common is not None else Path(repo)


SIBLINGS_FILE = REPO / "configs/sibling_worktrees.yaml"  # private-repo only (not in the public release): key -> working-copy folder name


@cache
def _siblings() -> dict:
    try:
        import yaml
        return dict((yaml.safe_load(SIBLINGS_FILE.read_text()) or {}).get("siblings") or {})
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def sibling(key: str) -> Path:
    """The sibling working copy that holds the frozen inputs named `key` (e.g. "phase2", "stepc_4b"). Env CGT_SIBLING_<KEY> overrides;
    otherwise configs/sibling_worktrees.yaml names its folder under worktrees_dir(). Without either (an exported copy) the location is the
    repo-relative data/external/siblings/<key>, which normally does not exist, so the inputs are missing and dependent tests skip."""
    env = os.environ.get(f"CGT_SIBLING_{key.upper()}")
    if env:
        return Path(env)
    name = _siblings().get(key)
    return worktrees_dir() / name if name else REPO / "data/external/siblings" / key

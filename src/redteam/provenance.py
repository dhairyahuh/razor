"""What produced this number, stamped onto everything that leaves the system.

A run report is a claim, and a claim without provenance cannot be checked. Six months from
now - or during judging, which is the case that actually matters - somebody will hold two
artefacts with different numbers and need to know whether the difference is the seed, the
config, a change to the attack library, or a code change. Without a stamp that question is
unanswerable and both artefacts become worthless.

So every run carries:

* a **run id** derived from the content, not from the clock, so two identical runs collide by
  design and a reviewer can tell at a glance that a rerun reproduced;
* the **git SHA** and whether the tree was dirty, because a number produced from uncommitted
  code is not reproducible and should say so out loud rather than being quietly presented
  beside numbers that are;
* a **config hash** over the resolved configuration, which catches the common case of two
  runs differing by one field nobody remembers changing;
* a **library hash** over the attack taxonomy, because the vector set is an input to every
  downstream metric and it changes more often than the code does;
* the **environment** - Python and the numeric stack - since a boosting library's version
  changes tie-breaking and therefore the third decimal place.

The stamp goes into the run report, into ``run_summary.json``, and into every API response.
The API case is the one people forget: a score returned without the identity of the model
that produced it cannot be audited after the fact, which is precisely when someone asks.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[2]


def _run_git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                             timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def git_sha() -> str:
    return _run_git("rev-parse", "--short=12", "HEAD") or "unknown"


def git_dirty() -> bool:
    """True when tracked files differ from HEAD.

    Reported rather than ignored. A metric produced from a modified working tree is not
    reproducible from the recorded SHA, and presenting it beside ones that are - without
    saying so - is the quiet way a results table stops being trustworthy.
    """
    if _run_git("rev-parse", "--git-dir") == "":
        return False
    return bool(_run_git("status", "--porcelain", "--untracked-files=no"))


def hash_object(payload: Any) -> str:
    """A stable short hash of any JSON-serialisable structure."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_path(path: Path) -> str:
    """Hash a file or every file under a directory, in sorted order."""
    digest = hashlib.sha256()
    if not path.exists():
        return "absent"
    files = sorted(path.rglob("*")) if path.is_dir() else [path]
    for item in files:
        if item.is_file():
            digest.update(item.name.encode("utf-8"))
            digest.update(item.read_bytes())
    return digest.hexdigest()[:16]


def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("numpy", "pandas", "sklearn"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except ImportError:  # pragma: no cover - all three are hard dependencies
            out[name] = "absent"
    return out


@dataclass
class Provenance:
    """The identity of one run, stamped on everything it produces."""

    run_name: str
    seed: int
    git_sha: str
    git_dirty: bool
    config_hash: str
    library_hash: str
    code_version: str
    environment: Dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    run_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        dirty = " (working tree dirty - not reproducible from this SHA)" if self.git_dirty else ""
        return (f"run `{self.run_id}` | seed {self.seed} | code `{self.git_sha}`{dirty} | "
                f"config `{self.config_hash}` | taxonomy `{self.library_hash}`")


def stamp(cfg, *, library_path: Optional[Path] = None) -> Provenance:
    """Build the provenance record for a configuration.

    The run id is content-derived rather than a timestamp or a uuid. Two runs of the same
    code, config, seed and taxonomy get the same id, which turns "did this reproduce?" from
    a diff of two reports into a string comparison. The creation time is recorded separately
    for exactly that reason: it is metadata about the run, not part of its identity.
    """
    config_payload = cfg.to_dict() if hasattr(cfg, "to_dict") else str(cfg)
    config_hash = hash_object(config_payload)
    lib = library_path or (ROOT / "src" / "redteam" / "identify" / "attack_library.yaml")
    library_hash = hash_path(lib)
    sha = git_sha()

    return Provenance(
        run_name=getattr(cfg, "run_name", "run"),
        seed=int(getattr(cfg, "seed", 0)),
        git_sha=sha,
        git_dirty=git_dirty(),
        config_hash=config_hash,
        library_hash=library_hash,
        code_version=_package_version(),
        environment=_versions(),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        run_id=hash_object({"sha": sha, "config": config_hash, "library": library_hash,
                            "seed": int(getattr(cfg, "seed", 0))}),
    )


def _package_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return "unknown"
    try:
        return version("redteam")
    except Exception:  # PackageNotFoundError, and anything a broken install raises
        return "editable"

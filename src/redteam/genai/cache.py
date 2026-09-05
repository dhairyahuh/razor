"""A content-addressed LLM cache, so a generative pipeline is still reproducible.

The problem this solves
-----------------------
A red-team system for GenAI-era fraud that runs no generative model has an obvious hole in
it. A red-team system whose results change every time it runs, because a hosted model was
sampled afresh, has a worse one: nothing it reports can be checked. Both are unacceptable
and they appear to be mutually exclusive.

They are not. Every completion is keyed by a hash of everything that determines it - the
prompt, the model name, the temperature, the task - and written to ``cache/llm/``, which is
meant to be committed alongside the code. So:

* **No API key.** Every lookup hits the cache. The run is byte-identical to the committed
  one, on any machine, offline, forever. Cache misses raise rather than returning something
  plausible, because a run that quietly substituted its own text for a model's and reported
  anyway would be a fabrication.
* **With an API key and ``refresh=True``.** Misses go to the provider and are written back.
  This is how the cache is built and extended, and it is the path a live demo uses.

The cache is therefore the artefact, and the API key is a build-time dependency rather than
a runtime one. That is the same arrangement as a lockfile, and it is the only way to have
generative content in a pipeline whose numbers are meant to be falsifiable.

What happens when the cache is empty
------------------------------------
``cache/llm/`` ships empty in this repository: populating it requires an API key, and the
alternative - hand-writing completions and labelling them as a model's - is the one thing
this module exists to prevent. Every caller therefore treats ``CacheMiss`` as "this
generative layer is unavailable", falls back to its template-composed path, and *says so* in
the run log and in the report. So a default run is fully reproducible and honestly labelled,
with the LLM-written content absent rather than faked. ``--refresh-llm`` with a key turns it
on, and everything it writes is reviewable because the prompt is stored beside the answer.

Determinism, honestly stated
----------------------------
``temperature=0`` does not make a hosted LLM deterministic; batching and kernel
non-determinism mean the same prompt can produce different tokens. That is precisely why
the cache exists rather than being an optimisation on top of a reproducible call. What is
guaranteed here is that *a given cache* produces a given run, and that the cache is under
version control where a reviewer can read it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

#: Where completions live. Committed, not generated on demand.
CACHE_DIR = Path(__file__).resolve().parents[3] / "cache" / "llm"

#: Default model. Recorded in the cache key, so changing it invalidates entries rather than
#: silently reusing another model's output.
DEFAULT_MODEL = "gpt-4o-mini"

#: OpenAI-compatible endpoint. Overridable so the same client works against any provider
#: exposing the chat-completions shape, including a local server.
DEFAULT_BASE_URL = os.environ.get("REDTEAM_LLM_BASE_URL", "https://api.openai.com/v1")

API_KEY_ENV = "REDTEAM_LLM_API_KEY"


class CacheMiss(KeyError):
    """A completion was needed, was not cached, and no live path was available.

    Deliberately fatal. The alternative - falling back to a template and carrying on - is
    how a pipeline ends up reporting numbers for generative content it never generated.
    """


@dataclass
class LLMCall:
    """Everything that determines a completion, and therefore its cache key."""

    task: str
    prompt: str
    system: str = ""
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 1200

    def key(self) -> str:
        payload = json.dumps(
            {
                "task": self.task,
                "prompt": self.prompt,
                "system": self.system,
                "model": self.model,
                "temperature": round(float(self.temperature), 4),
                "max_tokens": int(self.max_tokens),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    def path(self, cache_dir: Path) -> Path:
        # Grouped by task so the cache is browsable. A reviewer should be able to read the
        # prompts and the model's answers without running anything, and a flat directory of
        # hashes makes that needlessly hard.
        return cache_dir / self.task / f"{self.key()}.json"


@dataclass
class LLMResult:
    text: str
    cached: bool
    key: str
    model: str
    task: str


@dataclass
class CacheStats:
    """Counters, reported at the end of a run.

    Published rather than kept internal because "the generative layer ran" and "the
    generative layer was served entirely from cache" are different claims about a run, and a
    reader is entitled to know which one they are looking at.
    """

    hits: int = 0
    misses: int = 0
    live_calls: int = 0
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "live_calls": self.live_calls,
            "failures": len(self.failures),
            "served_from_cache_pct": round(
                100.0 * self.hits / max(1, self.hits + self.live_calls), 1
            ),
        }


class LLMCache:
    """Reads completions from disk; optionally refreshes them from a provider."""

    def __init__(self, cache_dir: Path | str = CACHE_DIR, *, refresh: bool = False,
                 api_key: Optional[str] = None, base_url: str = DEFAULT_BASE_URL,
                 transport: Optional[Callable[[LLMCall, str, str], str]] = None):
        self.cache_dir = Path(cache_dir)
        self.api_key = api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")
        # Refreshing needs a key. Asking for it without one is a configuration error worth
        # surfacing at construction rather than at the first miss.
        self.refresh = bool(refresh)
        if self.refresh and not self.api_key and transport is None:
            raise ValueError(
                f"refresh=True needs an API key in ${API_KEY_ENV} (or an injected transport)"
            )
        self.base_url = base_url
        self._transport = transport or _http_chat_completion
        self.stats = CacheStats()

    @property
    def live_available(self) -> bool:
        return self.refresh and (bool(self.api_key) or self._transport is not _http_chat_completion)

    def complete(self, call: LLMCall) -> LLMResult:
        path = call.path(self.cache_dir)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.stats.hits += 1
            return LLMResult(text=payload["completion"], cached=True, key=call.key(),
                             model=payload.get("model", call.model), task=call.task)

        self.stats.misses += 1
        if not self.live_available:
            raise CacheMiss(
                f"no cached completion for task={call.task!r} key={call.key()}. "
                f"Set ${API_KEY_ENV} and pass refresh=True to generate it, or run a stage "
                f"that does not need this completion."
            )

        text = self._transport(call, self.api_key, self.base_url)
        self.stats.live_calls += 1
        self._write(call, text)
        return LLMResult(text=text, cached=False, key=call.key(), model=call.model,
                         task=call.task)

    def _write(self, call: LLMCall, text: str) -> None:
        path = call.path(self.cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The prompt is stored alongside the completion. A cache of answers without the
        # questions cannot be reviewed, and reviewability is most of the point of committing
        # it.
        path.write_text(
            json.dumps(
                {
                    "task": call.task,
                    "model": call.model,
                    "temperature": call.temperature,
                    "system": call.system,
                    "prompt": call.prompt,
                    "completion": text,
                    "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def entries(self) -> List[Dict[str, Any]]:
        """Every cached completion, for the run report's provenance section."""
        rows = []
        if not self.cache_dir.exists():
            return rows
        for path in sorted(self.cache_dir.rglob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            rows.append(
                {
                    "task": payload.get("task", path.parent.name),
                    "model": payload.get("model", ""),
                    "key": path.stem,
                    "prompt_chars": len(payload.get("prompt", "")),
                    "completion_chars": len(payload.get("completion", "")),
                    "retrieved_at": payload.get("retrieved_at", ""),
                }
            )
        return rows


def _http_chat_completion(call: LLMCall, api_key: str, base_url: str) -> str:
    """Minimal OpenAI-compatible chat call over the standard library.

    Deliberately not a provider SDK. This is one POST against a shape every major provider
    implements, and adding a versioned dependency for it would make the offline path - the
    one that actually matters for reproducibility - carry a package it never calls.
    """
    messages = []
    if call.system:
        messages.append({"role": "system", "content": call.system})
    messages.append({"role": "user", "content": call.prompt})

    body = json.dumps(
        {
            "model": call.model,
            "messages": messages,
            "temperature": call.temperature,
            "max_tokens": call.max_tokens,
        }
    ).encode("utf-8")

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network path
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"LLM provider returned {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network path
        raise RuntimeError(f"could not reach LLM provider: {exc.reason}") from exc
    return payload["choices"][0]["message"]["content"]


def extract_json(text: str) -> Any:
    """Pull the first JSON object or array out of a completion.

    Models wrap structured output in prose and fenced code blocks however often they are
    asked not to. Failing the whole run because a model said "Here you go:" first would
    make the generative layer far more fragile than the thing it is generating.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [ln for ln in stripped.splitlines() if not ln.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    for opener, closer in (("[", "]"), ("{", "}")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON found in completion: {text[:200]!r}")

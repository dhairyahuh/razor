"""An LLM attack-ideation agent, gated by the library's own validator.

The Identify pillar was a hand-authored YAML file. That is a perfectly respectable artefact
and it is not an AI system, which is what the brief asks for. This module makes ideation a
loop the machine runs: the model is shown the schema, the existing taxonomy and the
families with thin coverage, and asked for vectors that are not already there.

The important part is not the prompt, it is the gate
----------------------------------------------------
Anything the model proposes is parsed into an :class:`AttackVector` and put through
``AttackLibrary.validate()`` - the same check the committed library passes. A proposal is
accepted only if it is well-formed, novel, and references *signals that exist in the
schema*. That last condition does most of the work: it is the difference between a vector
this system can actually simulate and measure, and a paragraph of plausible-sounding threat
prose. A model that invents ``deepfake_confidence_score`` gets rejected, because nothing in
the payment message carries it.

Rejections are logged and reported rather than hidden. The accept rate is the interesting
number: it says how much of what a frontier model proposes about payment fraud is
grounded enough to be executable, and quietly dropping the failures would turn a real
measurement into a marketing one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..identify.library import AttackLibrary, AttackVector, _clean
from ..schema import observable_columns
from .cache import LLMCache, LLMCall, extract_json

SYSTEM = (
    "You are a payments fraud strategist working for a card network's red team. You "
    "understand real payment rails, authorisation flows, settlement timing and the "
    "delegated-authority protocols used by AI shopping agents. You propose attacks that "
    "are mechanically plausible on real infrastructure, not generic cybercrime. You answer "
    "with JSON only."
)

#: The reply shape. Constrained tightly because every field maps onto an ``AttackVector``
#: attribute, and a free-form answer would need an LLM to parse it.
RESPONSE_CONTRACT = """
Reply with a JSON array of objects. Each object must have exactly these keys:

  "id"                    UPPER-KEBAB, prefixed with the family, e.g. "APP-COURIER-REDIRECT"
  "name"                  short human title
  "family"                one of: {families}
  "description"           2-3 sentences on the mechanism, concretely
  "genai_enablers"        list of strings: what generative AI specifically adds
  "kill_chain"            list, subset of: {kill_chain}
  "rails"                 list, subset of: {rails}
  "channels"              list, subset of: {channels}
  "victim"                one of: consumer, merchant, business, psp, issuer
  "signals"               list of column names from the schema list below. THIS IS THE
                          HARD CONSTRAINT: only names in that list are permitted, because
                          a signal the payment message does not carry cannot be simulated
                          or detected. Choose 4-8.
  "controls"              list of strings: plausible mitigations
  "severity"              integer 1-5
  "prevalence"            integer 1-5
  "detection_difficulty"  integer 1-5
  "liability_note"        one sentence on who bears the loss, or ""
"""


@dataclass
class Proposal:
    """One proposed vector and the verdict on it."""

    raw: Dict[str, Any]
    vector: Optional[AttackVector]
    accepted: bool
    problems: List[str] = field(default_factory=list)

    @property
    def vector_id(self) -> str:
        return str(self.raw.get("id", "(no id)"))


@dataclass
class IdeationResult:
    proposals: List[Proposal]
    served_from_cache: bool

    @property
    def accepted(self) -> List[AttackVector]:
        return [p.vector for p in self.proposals if p.accepted and p.vector is not None]

    def table(self) -> pd.DataFrame:
        """One row per proposal, accepted or not.

        The rejected rows are the point. An accept rate reported without them is an
        assertion; with them it is a measurement a reader can audit.
        """
        rows = []
        for p in self.proposals:
            rows.append(
                {
                    "vector_id": p.vector_id,
                    "family": str(p.raw.get("family", "")),
                    "accepted": int(p.accepted),
                    "n_signals": len(p.raw.get("signals", []) or []),
                    "rejection_reason": "; ".join(p.problems)[:200],
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> Dict[str, Any]:
        n = len(self.proposals)
        accepted = len(self.accepted)
        return {
            "proposed": n,
            "accepted": accepted,
            "rejected": n - accepted,
            "accept_rate": round(accepted / n, 4) if n else 0.0,
            "served_from_cache": self.served_from_cache,
        }


def build_prompt(library: AttackLibrary, *, n: int = 6,
                 threat_intel: str = "") -> str:
    """Assemble the ideation prompt from the live schema and taxonomy.

    Built from the objects rather than pasted in, so the prompt cannot drift out of step
    with the schema it constrains proposals against. A stale column list would produce
    proposals that fail validation for a reason that is the prompt's fault.
    """
    families = ", ".join(sorted(library.families))
    counts = {f: len(library.by_family(f)) for f in sorted(library.families)}
    thin = sorted(counts, key=lambda f: counts[f])[:3]

    existing = "\n".join(
        f"  {v.id}: {v.description.splitlines()[0][:110]}" for v in library.vectors
    )
    columns = ", ".join(sorted(observable_columns()))

    contract = RESPONSE_CONTRACT.format(
        families=families,
        kill_chain=", ".join(library.kill_chain),
        rails="CARD_CNP, CARD_CP, UPI_P2P, UPI_P2M, IMPS, NEFT, RTP_FEDNOW, SEPA_INST, WALLET",
        channels="mobile_app, web, ivr, agent_api, branch, pos, call_centre",
    )

    intel = f"\nRecent threat intelligence to draw on:\n{threat_intel}\n" if threat_intel else ""

    return f"""We maintain a taxonomy of {len(library.vectors)} payment fraud vectors for a
red-team simulator. Propose {n} NEW vectors that are not already covered.

The families with the thinnest coverage are: {', '.join(thin)}. Weight your proposals
toward those, but do not force it - a strong vector in a well-covered family beats a weak
one in a thin family.

Existing vectors (id: first line of description):
{existing}
{intel}
The schema columns available as signals - and the ONLY permitted values for the "signals"
field - are:
{columns}
{contract}
Rules:
- Do not restate an existing vector under a new name. Novelty means a different mechanism,
  not a different wording.
- Every signal must be a column from the list above, spelled exactly.
- The mechanism in "description" must be something that could actually happen on the rail
  you name. If it needs a capability the rail does not have, it is not a valid proposal.
"""


def propose_vectors(library: AttackLibrary, cache: LLMCache, *, n: int = 6,
                    threat_intel: str = "", temperature: float = 0.8) -> IdeationResult:
    """Ask for new vectors, then judge them with the library's own validator."""
    call = LLMCall(
        task="ideate_vectors",
        prompt=build_prompt(library, n=n, threat_intel=threat_intel),
        system=SYSTEM,
        temperature=temperature,
        max_tokens=3000,
    )
    result = cache.complete(call)
    payload = extract_json(result.text)
    if isinstance(payload, dict):
        payload = payload.get("vectors", [payload])

    known_ids = {v.id for v in library.vectors}
    known_names = {v.name.strip().lower() for v in library.vectors}
    proposals = [
        _judge(raw, library, known_ids, known_names) for raw in payload if isinstance(raw, dict)
    ]
    return IdeationResult(proposals=proposals, served_from_cache=result.cached)


def _judge(raw: Dict[str, Any], library: AttackLibrary, known_ids: set,
           known_names: set) -> Proposal:
    """Accept or reject one proposal, with the reason recorded either way."""
    problems: List[str] = []

    vector_id = str(raw.get("id", "")).strip()
    if not vector_id:
        return Proposal(raw=raw, vector=None, accepted=False, problems=["no id"])
    if vector_id in known_ids:
        problems.append(f"duplicate of existing vector {vector_id}")
    if str(raw.get("name", "")).strip().lower() in known_names:
        problems.append("name duplicates an existing vector")

    # Construct through the library's own loader so the proposal has to satisfy exactly the
    # constraints the committed file does - including rejecting unknown keys, which is how a
    # model's invented field gets caught.
    vector: Optional[AttackVector] = None
    try:
        vector = AttackVector(**_clean(dict(raw)))
    except (TypeError, ValueError) as exc:
        problems.append(f"malformed: {exc}")
        return Proposal(raw=raw, vector=None, accepted=False, problems=problems)

    # Validate in the context of the real library, so family names and kill-chain stages are
    # checked against the taxonomy rather than against a copy of it.
    probe = AttackLibrary(version=library.version, updated=library.updated,
                          kill_chain=library.kill_chain, families=library.families,
                          vectors=[vector])
    problems.extend(probe.validate())

    if not vector.signals:
        problems.append("no signals: nothing to simulate or detect")

    return Proposal(raw=raw, vector=vector, accepted=not problems, problems=problems)


def to_yaml_block(vectors: List[AttackVector]) -> str:
    """Render accepted proposals as a YAML block ready to paste into the library.

    Emitted as text for a human to review and merge rather than appended automatically. An
    agent that edits the taxonomy it is measured against, unsupervised, is a way to
    manufacture a rising vector count rather than a rising quality of coverage.
    """
    lines: List[str] = []
    for v in vectors:
        lines.append(f"- id: {v.id}")
        lines.append(f"  name: {json.dumps(v.name)}")
        lines.append(f"  family: {v.family}")
        lines.append(f"  description: {json.dumps(v.description)}")
        for key in ("genai_enablers", "kill_chain", "rails", "channels", "signals",
                    "controls"):
            value = getattr(v, key)
            if value:
                lines.append(f"  {key}: [{', '.join(json.dumps(x) for x in value)}]")
        lines.append(f"  victim: {v.victim}")
        lines.append("  simulated: false")
        lines.append(f"  severity: {v.severity}")
        lines.append(f"  prevalence: {v.prevalence}")
        lines.append(f"  detection_difficulty: {v.detection_difficulty}")
        if v.liability_note:
            lines.append(f"  liability_note: {json.dumps(v.liability_note)}")
        lines.append("")
    return "\n".join(lines)

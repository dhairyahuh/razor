"""Turning reason codes into something an analyst can act on in ten seconds.

Why this is not decoration
--------------------------
The model already emits reason codes - the features that moved a given alert, from
permutation importance. They look like ``f_payer_amount_sum_1h, payee_account_age_days,
f_payer_new_payees_7d``, and they are correct, auditable, and almost unreadable at the rate
a real queue arrives. A reviewer working a few hundred alerts a shift does not decode feature
names; they want to know what story the alert is telling and what to check first.

That gap is where fraud-team productivity actually lives, and it is the difference between a
model output and a product. It is also the one place in this system where a language model's
weakness is irrelevant: the narrative is *downstream* of the decision. If it hallucinates,
the alert is unaffected, the score is unaffected, and the auditable reason codes sit next to
it unchanged. Nothing here can move a decision, which is why it is safe to generate and why
it is generated last.

Grounding
---------
The prompt is given only the reason codes, a small set of numeric facts about the payment,
and the guard verdicts. It is told explicitly not to assert anything not present in those
facts, and the rendered narrative is stored beside the reason codes rather than replacing
them, so an auditor always has the machine-checkable version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .cache import CacheMiss, LLMCache, LLMCall, extract_json

SYSTEM = (
    "You are a senior fraud analyst writing the one-line summary that sits at the top of an "
    "alert in a bank's case-management queue. You are terse, concrete and never speculate "
    "beyond the evidence you are given. You answer with JSON only."
)

#: Facts offered to the narrator. Kept small and raw on purpose: the narrative should be a
#: restatement of evidence, and a model given the whole row will start inferring.
NARRATE_FIELDS: List[str] = [
    "amount", "rail", "channel", "payee_is_first_time", "payee_account_age_days",
    "device_is_new", "call_in_progress", "screen_share_active", "initiated_by_agent",
    "is_cross_border",
]


@dataclass
class Narrative:
    txn_id: str
    headline: str
    detail: str
    next_check: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "txn_id": self.txn_id,
            "alert_headline": self.headline,
            "alert_detail": self.detail,
            "suggested_next_check": self.next_check,
        }


@dataclass
class NarrativeReport:
    narratives: List[Narrative] = field(default_factory=list)
    served_from_cache: bool = True
    requested: int = 0

    def __bool__(self) -> bool:
        return bool(self.narratives)

    def to_dict(self) -> Dict[str, object]:
        return {
            "alerts_narrated": len(self.narratives),
            "alerts_requested": self.requested,
            "served_from_cache": self.served_from_cache,
        }

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([n.to_dict() for n in self.narratives])


def build_prompt(alerts: List[Dict[str, object]]) -> str:
    listing = "\n".join(
        f"  {i}. " + json.dumps(a, default=str, separators=(", ", ": "))
        for i, a in enumerate(alerts)
    )
    return f"""Below are {len(alerts)} fraud alerts from a real-time payment monitoring
system. Each carries the payment's facts, the model's score, and the reason codes - the
features that most moved this particular decision.

For each alert write the summary an analyst sees first.

{listing}

Rules:
  - Say what the pattern is, in plain words, using only the facts given. Do not invent a
    scam type, a victim profile, a location or a history that is not in the data.
  - If the facts do not support a story, say so - "elevated score with no single dominant
    signal" is a valid and useful headline.
  - The headline is at most twelve words. The detail is one or two sentences. The next check
    is one concrete action an analyst can take in the case system.
  - Never state a conclusion the reason codes do not support. An analyst who learns these
    summaries overreach will stop reading them, and then the alert queue is worse than it
    was before.

Reply with a JSON array, one object per alert:
[{{"row": 0, "headline": "...", "detail": "...", "next_check": "..."}}]
No commentary outside the JSON.
"""


def narrate_alerts(scored: pd.DataFrame, featured: pd.DataFrame, cache: LLMCache, *,
                   n: int = 12, temperature: float = 0.3) -> NarrativeReport:
    """Narrate the highest-scoring alerts.

    Only the top of the queue, and for the ordinary reason that this is where a reviewer
    starts. Narrating every alert would multiply the cache for no benefit; the value is in
    demonstrating the mechanism on the cases a human actually opens.
    """
    alerts = scored[scored["model_alert"] == 1] if "model_alert" in scored.columns \
        else scored.iloc[:0]
    if alerts.empty:
        return NarrativeReport()

    top = alerts.sort_values("score", ascending=False).head(n)
    facts = featured.set_index("txn_id")
    columns = [c for c in NARRATE_FIELDS if c in facts.columns]

    payload: List[Dict[str, object]] = []
    for row in top.itertuples():
        if row.txn_id not in facts.index:
            continue
        record = facts.loc[row.txn_id, columns]
        if isinstance(record, pd.DataFrame):
            record = record.iloc[0]
        payload.append({
            "score": round(float(row.score), 3),
            "reason_codes": str(getattr(row, "reason_codes", "") or "none recorded"),
            "intent_guard_blocked": int(getattr(row, "intent_block", 0) or 0),
            "agent_control_blocked": int(getattr(row, "control_block", 0) or 0),
            "injection_flagged": int(getattr(row, "injection_flag", 0) or 0),
            **{k: (round(float(v), 2) if isinstance(v, float) else v)
               for k, v in record.to_dict().items()},
        })

    if not payload:
        return NarrativeReport()

    call = LLMCall(task="alert_narratives", prompt=build_prompt(payload), system=SYSTEM,
                   temperature=temperature, max_tokens=2600)
    try:
        result = cache.complete(call)
        parsed = extract_json(result.text)
    except (CacheMiss, ValueError):
        return NarrativeReport(requested=len(payload))

    if isinstance(parsed, dict):
        parsed = parsed.get("narratives") or parsed.get("alerts") or []
    if not isinstance(parsed, list):
        return NarrativeReport(requested=len(payload))

    ids = list(top["txn_id"])
    out: List[Narrative] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("row", -1))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(ids):
            continue
        out.append(Narrative(
            txn_id=str(ids[index]),
            headline=str(item.get("headline", ""))[:120],
            detail=str(item.get("detail", ""))[:400],
            next_check=str(item.get("next_check", ""))[:200],
        ))
    return NarrativeReport(narratives=out, served_from_cache=result.cached,
                           requested=len(payload))

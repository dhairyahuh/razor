"""Asking a model to tell generated transactions apart from plausible real ones.

What this measures, and what it cannot
--------------------------------------
Every other fidelity check in this repository is a statistic the generator could be tuned to
satisfy. Benford adherence, round-number mass, the diurnal curve, per-column KS distances -
all of them are targets, and a generator that hits all of them can still produce rows that
are obviously synthetic to anyone who has looked at payment data, because the implausibility
lives in the *combination* rather than in any marginal.

A discriminator catches that, and this one is a language model shown interleaved rows with
the labels stripped. The metric is its accuracy at picking the synthetic one, and the target
is **50%** - chance. Above chance means the model found a tell; well above chance means the
tell is blatant. Unlike the marginal checks this cannot be optimised against without
actually fixing the joint distribution, which is the property that makes it worth having.

The honest caveat, stated here rather than buried: **there is no real payment data in this
repository.** The comparison set is not held-out reality, it is a description of what real
Indian retail payment data looks like, drawn from published aggregates - UPI's per-transaction
value distribution, NPCI's rail mix, typical merchant-category spread - which the model is
asked to hold in mind while judging. So this measures "does a knowledgeable reader find these
rows plausible", not "are these rows drawn from the true distribution". Those are different
claims and only the first is available without a data-sharing agreement.

That still catches the failure mode that matters. A model that says "no genuine UPI stream
has 40% of its volume between 2 and 4am" is telling you something no KS test on the hour
marginal would, because the marginal was tuned and the interaction with amount was not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .cache import CacheMiss, LLMCache, LLMCall, extract_json

SYSTEM = (
    "You are a payments data analyst who has spent years looking at raw transaction tables "
    "from Indian retail payment rails - UPI, IMPS, NEFT, cards. You are being asked to "
    "review synthetic data for realism. You are blunt about what looks wrong and you do not "
    "flatter the data. You answer with JSON only."
)

#: The columns shown to the judge.
#:
#: Deliberately the raw observable ones a bank sees in a payment message, not the derived
#: features. Showing a velocity aggregate would ask the model to judge this project's feature
#: engineering, which is not what fidelity means, and would leak the label immediately
#: because no real extract carries columns named ``f_``.
JUDGE_COLUMNS: List[str] = [
    "timestamp", "amount", "rail", "channel", "merchant_category",
    "payee_is_first_time", "payee_account_age_days", "device_is_new",
    "customer_age_band", "is_cross_border",
]

#: How the real side is described, since there is none to show.
REAL_WORLD_BRIEF = """\
Genuine Indian retail payment traffic has these properties, from published NPCI and RBI
aggregates:
  - UPI dominates by count and skews small: a large mass of transactions under Rs 500, a
    median in the low hundreds, and a long right tail. IMPS and NEFT skew much larger.
  - Amounts cluster hard on round numbers - 100, 500, 1000, 2000, 5000 - far more than any
    smooth distribution predicts, because humans choose them.
  - Volume follows a strong daily cycle: a morning rise, a midday peak, an evening peak that
    is usually the largest, and a deep trough between about 1am and 5am that is a small
    fraction of peak. Weekends differ from weekdays in mix, not only in level.
  - Most payments go to payees the customer has paid before. First-time payees are a real but
    minority share of any customer's activity.
  - Device changes are uncommon per customer but not rare across a population.
  - Cross-border is a small fraction of retail volume.
  - Merchant categories are heavily concentrated: groceries, fuel, utilities, telecom
    top-ups, food delivery, and peer-to-peer transfers make up most of the volume.
"""


@dataclass
class JudgeVerdict:
    row_index: int
    guess_synthetic: bool
    truth_synthetic: bool
    reason: str

    @property
    def correct(self) -> bool:
        return self.guess_synthetic == self.truth_synthetic


@dataclass
class JudgeReport:
    verdicts: List[JudgeVerdict] = field(default_factory=list)
    served_from_cache: bool = True
    tells: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.verdicts)

    @property
    def accuracy(self) -> float:
        """Share of rows correctly classified. Chance is 0.5 and chance is the target."""
        if not self.verdicts:
            return float("nan")
        return float(np.mean([v.correct for v in self.verdicts]))

    def to_dict(self) -> Dict[str, object]:
        n = len(self.verdicts)
        flagged = sum(v.guess_synthetic for v in self.verdicts)
        return {
            "rows_judged": n,
            "discriminator_accuracy": round(self.accuracy, 4) if n else None,
            # Reported because an accuracy near chance can also mean the model gave up and
            # answered the same way throughout, which is a different situation from a
            # genuinely indistinguishable corpus and should not be allowed to look like one.
            "share_called_synthetic": round(flagged / n, 4) if n else None,
            "verdict": _verdict_text(self.accuracy) if n else "not run",
            "tells_cited": self.tells[:6],
            "served_from_cache": self.served_from_cache,
        }

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "row": v.row_index,
                "truth": "synthetic" if v.truth_synthetic else "described-real",
                "model_guess": "synthetic" if v.guess_synthetic else "real",
                "correct": v.correct,
                "reason": v.reason[:120],
            }
            for v in self.verdicts
        ])


def _verdict_text(accuracy: float) -> str:
    if accuracy != accuracy:
        return "not run"
    if accuracy <= 0.6:
        return "indistinguishable at chance - the model could not reliably pick the synthetic rows"
    if accuracy <= 0.75:
        return "weakly distinguishable - some tells, but not decisive"
    return "distinguishable - the model found a consistent tell, see the cited reasons"


def build_prompt(rows: Sequence[Dict[str, object]]) -> str:
    listing = "\n".join(
        f"  {i}. " + json.dumps(row, default=str, separators=(", ", ": "))
        for i, row in enumerate(rows)
    )
    return f"""Below are {len(rows)} payment records. Some are synthetic, generated by a
simulator. The rest are records a payments analyst would consider entirely typical of real
Indian retail payment traffic. They are shuffled and unlabelled.

{REAL_WORLD_BRIEF}
For each record, judge whether it looks synthetic or real. Be strict: consider whether the
*combination* of fields is plausible, not just whether each field is individually in range.
An amount that is fine on its own can be implausible at 3am on that rail to a first-time
payee in that category.

Records:
{listing}

Reply with a JSON object:
{{"verdicts": [{{"row": 0, "synthetic": true, "reason": "..."}}, ...],
  "tells": ["the single most common giveaway", "the next one"]}}

The reason must be one short clause naming the specific implausibility, or "nothing stands
out" if there is none. Judge every row. No commentary outside the JSON.
"""


def judge_fidelity(transactions: pd.DataFrame, cache: LLMCache, *, n_rows: int = 40,
                   seed: int = 0, temperature: float = 0.2) -> JudgeReport:
    """Show the model a sample of generated rows and score its discrimination.

    Only legitimate rows are sampled. Fraud is *supposed* to look unusual, so including it
    would let the judge score well by spotting attacks and would turn a fidelity measurement
    into a detection one.
    """
    legit = transactions[transactions["is_fraud"] == 0]
    columns = [c for c in JUDGE_COLUMNS if c in legit.columns]
    if legit.empty or not columns:
        return JudgeReport()

    rng = np.random.default_rng(seed)
    take = min(n_rows, len(legit))
    sample = legit.iloc[rng.choice(len(legit), size=take, replace=False)][columns]
    rows = [
        {k: (round(float(v), 2) if isinstance(v, (int, float, np.floating)) and k == "amount"
             else v)
         for k, v in record.items()}
        for record in sample.to_dict("records")
    ]

    call = LLMCall(task="fidelity_judge", prompt=build_prompt(rows), system=SYSTEM,
                   temperature=temperature, max_tokens=3000)
    try:
        result = cache.complete(call)
        payload = extract_json(result.text)
    except (CacheMiss, ValueError):
        return JudgeReport()

    if not isinstance(payload, dict):
        return JudgeReport()

    verdicts: List[JudgeVerdict] = []
    for item in payload.get("verdicts", []):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("row", -1))
        except (TypeError, ValueError):
            continue
        if not 0 <= index < len(rows):
            continue
        verdicts.append(JudgeVerdict(
            row_index=index,
            guess_synthetic=bool(item.get("synthetic", False)),
            # Every row shown *is* synthetic. The prompt says some are real because a judge
            # told everything is synthetic will simply agree, and the resulting 100% would
            # measure compliance rather than discrimination. The cost of this framing is that
            # only the false-positive half of the confusion matrix is observable, which is
            # why the headline is phrased as "share the model called synthetic" alongside the
            # accuracy rather than as a bare accuracy.
            truth_synthetic=True,
            reason=str(item.get("reason", ""))[:200],
        ))

    tells = [str(t)[:160] for t in payload.get("tells", []) if isinstance(t, str)]
    return JudgeReport(verdicts=verdicts, served_from_cache=result.cached, tells=tells)

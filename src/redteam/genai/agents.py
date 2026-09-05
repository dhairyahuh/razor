"""LLM red-team and blue-team agents, both of them scored rather than believed.

What these add over the search that already exists
--------------------------------------------------
The closed loop already has a red team: a random-mutation search over
:class:`~redteam.loop.coevolution.AttackGenome`. It works, and it is blind. It cannot read
the per-vector recall table and notice that the vectors getting through are the ones routing
to aged payee accounts, so it rediscovers that from scratch every round by sampling.

These agents get the tables. The red team is shown per-vector recall, the moves available
and what previous rounds achieved, and asked which levers to pull on which vector and why.
The blue team is shown the worst-performing slices and asked for a feature that would
separate them, expressed as an expression over columns that already exist.

Why the reasoning is not the deliverable
----------------------------------------
An LLM asked why an attack works will produce a fluent answer whether or not it is right,
so nothing here is accepted on the strength of its explanation. Every red-team proposal is
converted into a genome and *executed*, and its fitness is measured the same way the random
search's is. Every blue-team proposal is compiled, evaluated on held-out data, and accepted
only if it improves recall at a fixed false-positive budget.

The number worth reporting is therefore the accept rate, not the best proposal: how often
does a model reasoning over real evaluation tables beat random search on the same budget.
The rejections are published for the same reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cache import LLMCache, LLMCall, extract_json

RED_SYSTEM = (
    "You are the red team on a payment fraud detection project. You are given the "
    "defence's own per-vector recall table and the mutation levers available to you. You "
    "reason about which attack vectors are closest to evading detection and which levers "
    "would push them over. You answer with JSON only."
)

BLUE_SYSTEM = (
    "You are the blue team on a payment fraud detection project. You are given the slices "
    "of traffic where the current model is weakest, and the columns available in the "
    "transaction schema. You propose derived features that would separate fraud from "
    "legitimate traffic inside those slices. You answer with JSON only."
)


# --------------------------------------------------------------------------------------
# Red team
# --------------------------------------------------------------------------------------

@dataclass
class TacticProposal:
    """One proposed attack configuration, plus the model's stated reasoning.

    ``rationale`` is carried through to the report because it is genuinely interesting
    reading, and marked clearly as unverified because it is. The fitness column beside it
    is the part that was measured.
    """

    vector_id: str
    moves: Tuple[str, ...]
    strength: float
    share: float
    aged_payee_share: float
    rationale: str
    valid: bool = True
    problems: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "moves": ", ".join(sorted(self.moves)),
            "strength": round(self.strength, 3),
            "share": round(self.share, 3),
            "aged_payee_share": round(self.aged_payee_share, 3),
            "valid": int(self.valid),
            "problems": "; ".join(self.problems)[:160],
            "unverified_rationale": self.rationale[:300],
        }


def build_red_prompt(per_vector: pd.DataFrame, move_names: Sequence[str], *,
                     n: int = 5, history: Optional[pd.DataFrame] = None) -> str:
    """Show the red team what the defence is actually weak against."""
    cols = [c for c in ("attack_vector_id", "fraud_rows", "recall", "recall_lo95")
            if c in per_vector.columns]
    weakest = per_vector.sort_values("recall").head(12)[cols]
    table = weakest.to_string(index=False)

    prior = ""
    if history is not None and not history.empty:
        keep = [c for c in ("vector_id", "moves", "strength", "aged_payee_share",
                            "recall_after") if c in history.columns]
        if keep:
            prior = (
                "\nTactics already tried, and the recall they left the defence with "
                "(lower is better for you):\n" + history[keep].to_string(index=False) + "\n"
            )

    return f"""The defence's per-vector recall on held-out data, weakest first. Recall is the
share of that vector's fraud the model alerts on at a fixed false-positive budget, so low
recall means the vector is already getting through.

{table}
{prior}
The levers you control, and only these:

  moves: any subset of {list(move_names)}
    Each move shifts a group of observable features toward the legitimate distribution.
  strength: 0.2 to 0.98 - how far each move shifts. High strength evades more but costs
    the attack its potency, because a payment mutated to look ordinary often is ordinary.
  share: 0.3 to 1.0 - the fraction of the campaign's payments that get mutated.
  aged_payee_share: 0.0 to 1.0 - the fraction rerouted to accounts with months of genuine
    inbound history, bought from a mule broker. This is the only lever that changes where
    the money goes rather than how the payment looks, so it is the only one that reaches
    the defence's counterparty-novelty features.

Propose {n} tactics. Reply with a JSON array of objects with keys:
  "vector_id"         one of the vector ids in the table above
  "moves"             array of move names
  "strength"          number in [0.2, 0.98]
  "share"             number in [0.3, 1.0]
  "aged_payee_share"  number in [0.0, 1.0]
  "rationale"         one or two sentences: why this combination, for this vector

Reason from the table. A vector already at low recall needs a small push; one at high
recall needs a lever that reaches whatever feature is catching it. Do not propose the same
combination twice.
"""


def propose_tactics(per_vector: pd.DataFrame, move_names: Sequence[str], cache: LLMCache,
                    *, n: int = 5, history: Optional[pd.DataFrame] = None,
                    temperature: float = 0.9) -> Tuple[List[TacticProposal], bool]:
    """Ask for tactics and validate them against the search space's real bounds."""
    call = LLMCall(
        task="red_team_tactics",
        prompt=build_red_prompt(per_vector, move_names, n=n, history=history),
        system=RED_SYSTEM,
        temperature=temperature,
        max_tokens=2000,
    )
    result = cache.complete(call)
    payload = extract_json(result.text)
    if isinstance(payload, dict):
        payload = payload.get("tactics", [payload])

    known_vectors = set(per_vector["attack_vector_id"].astype(str))
    valid_moves = set(move_names)
    proposals = [
        _judge_tactic(raw, known_vectors, valid_moves)
        for raw in payload if isinstance(raw, dict)
    ]
    return proposals, result.cached


def _judge_tactic(raw: Dict[str, Any], known_vectors: set,
                  valid_moves: set) -> TacticProposal:
    """Clamp and check. The model may not widen its own action space."""
    problems: List[str] = []

    vector_id = str(raw.get("vector_id", "")).strip()
    if vector_id not in known_vectors:
        problems.append(f"unknown vector {vector_id!r}")

    raw_moves = raw.get("moves") or []
    moves = tuple(sorted({str(m) for m in raw_moves if str(m) in valid_moves}))
    invented = sorted({str(m) for m in raw_moves} - valid_moves)
    if invented:
        # Recorded rather than silently dropped: a model inventing levers is a real and
        # interesting failure mode, and it is exactly what a reader would want to know.
        problems.append(f"invented moves: {invented}")
    if not moves and not float(raw.get("aged_payee_share", 0.0) or 0.0) > 0.01:
        problems.append("no usable lever")

    def _clamp(key: str, lo: float, hi: float, default: float) -> float:
        try:
            value = float(raw.get(key, default))
        except (TypeError, ValueError):
            problems.append(f"{key} not a number")
            return default
        if not lo <= value <= hi:
            problems.append(f"{key}={value} outside [{lo}, {hi}], clamped")
        return float(np.clip(value, lo, hi))

    return TacticProposal(
        vector_id=vector_id,
        moves=moves,
        strength=_clamp("strength", 0.2, 0.98, 0.6),
        share=_clamp("share", 0.3, 1.0, 0.7),
        aged_payee_share=_clamp("aged_payee_share", 0.0, 1.0, 0.0),
        rationale=str(raw.get("rationale", "")).strip(),
        valid=not [p for p in problems if "clamped" not in p],
        problems=problems,
    )


# --------------------------------------------------------------------------------------
# Blue team
# --------------------------------------------------------------------------------------

#: Operators a proposed feature expression may use. Anything else is rejected.
#:
#: This is an allowlist rather than a sanitiser because the input is text from a language
#: model and the output is evaluated against a dataframe. A denylist of dangerous names is
#: the wrong shape of defence for that: it has to anticipate every route to attribute
#: access, and `eval` offers a great many.
ALLOWED_EXPR = re.compile(r"^[A-Za-z0-9_\s\.\+\-\*/\(\)\&\|\<\>\=\!,]+$")

#: Names permitted inside an expression beyond the frame's own columns.
ALLOWED_NAMES = frozenset({"abs", "minimum", "maximum", "log1p", "where", "np"})


@dataclass
class FeatureProposal:
    """One proposed derived feature, and what happened when it was tried."""

    name: str
    expression: str
    rationale: str
    compiled: bool = False
    accepted: bool = False
    recall_before: float = float("nan")
    recall_after: float = float("nan")
    problems: List[str] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.recall_after - self.recall_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.name,
            "expression": self.expression[:120],
            "compiled": int(self.compiled),
            "accepted": int(self.accepted),
            "recall_before": round(self.recall_before, 4),
            "recall_after": round(self.recall_after, 4),
            "delta_recall": round(self.delta, 4) if self.compiled else np.nan,
            "problems": "; ".join(self.problems)[:160],
            "unverified_rationale": self.rationale[:300],
        }


def build_blue_prompt(slices: pd.DataFrame, columns: Sequence[str], *, n: int = 5) -> str:
    """Show the blue team where the model is weakest, and what it may build from."""
    keep = [c for c in ("slice", "dimension", "value", "rows", "fraud_rows", "recall")
            if c in slices.columns]
    table = slices.head(10)[keep].to_string(index=False) if keep else slices.head(10).to_string()

    return f"""The slices of traffic where the current fraud model performs worst, measured on
held-out data at a fixed false-positive budget:

{table}

Columns available to build from, all of which are already computed for every payment:
{', '.join(sorted(columns))}

Propose {n} derived features that would help inside those weak slices. Reply with a JSON
array of objects with keys:
  "name"        snake_case identifier, prefixed "llm_"
  "expression"  a single pandas expression over the columns above, using the frame name
                `df`, e.g. "df['amount'] / (df['f_amt_mean_30d'] + 1)"
  "rationale"   one or two sentences on what fraud behaviour this separates

Constraints:
- The expression must be one line, must reference only the columns listed above, and must
  return a numeric Series aligned to `df`.
- Permitted functions: abs, np.minimum, np.maximum, np.log1p, np.where. Nothing else.
- No comparisons to a hard-coded amount in a single currency; this data is multi-currency.
- The feature must be computable from a single payment plus the trailing-window columns
  already present. It may not require future information.
"""


def propose_features(slices: pd.DataFrame, columns: Sequence[str], cache: LLMCache, *,
                     n: int = 5, temperature: float = 0.7) -> Tuple[List[FeatureProposal], bool]:
    call = LLMCall(
        task="blue_team_features",
        prompt=build_blue_prompt(slices, columns, n=n),
        system=BLUE_SYSTEM,
        temperature=temperature,
        max_tokens=2000,
    )
    result = cache.complete(call)
    payload = extract_json(result.text)
    if isinstance(payload, dict):
        payload = payload.get("features", [payload])

    proposals = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        proposals.append(
            FeatureProposal(
                name=str(raw.get("name", "")).strip() or "llm_unnamed",
                expression=str(raw.get("expression", "")).strip(),
                rationale=str(raw.get("rationale", "")).strip(),
            )
        )
    return proposals, result.cached


def compile_feature(proposal: FeatureProposal, frame: pd.DataFrame) -> Optional[np.ndarray]:
    """Evaluate a proposed expression against a frame, or reject it with a reason.

    The allowlist is checked before evaluation and the namespace is empty apart from the
    frame and the permitted numpy helpers. Text from a language model reaching `eval` is a
    code-execution path, and the fact that it is our own model on our own machine does not
    change what it is - a poisoned tool description upstream is precisely the attack this
    project simulates.
    """
    expr = proposal.expression
    if not expr:
        proposal.problems.append("empty expression")
        return None
    if len(expr) > 400:
        proposal.problems.append("expression too long")
        return None
    if not ALLOWED_EXPR.match(expr.replace("'", "").replace('"', "").replace("[", "").replace("]", "")):
        proposal.problems.append("expression contains disallowed characters")
        return None
    if "__" in expr or "import" in expr or "lambda" in expr:
        proposal.problems.append("expression contains a disallowed construct")
        return None

    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    quoted = set(re.findall(r"['\"]([^'\"]+)['\"]", expr))
    unknown_cols = quoted - set(frame.columns)
    if unknown_cols:
        proposal.problems.append(f"unknown columns: {sorted(unknown_cols)[:4]}")
        return None
    stray = identifiers - ALLOWED_NAMES - {"df"} - set(frame.columns)
    if stray:
        proposal.problems.append(f"disallowed names: {sorted(stray)[:4]}")
        return None

    try:
        value = eval(  # noqa: S307 - allowlisted expression, empty builtins, see above
            expr, {"__builtins__": {}, "np": np, "abs": abs}, {"df": frame}
        )
        array = np.asarray(pd.Series(value, index=frame.index), dtype=float)
    except Exception as exc:  # noqa: BLE001 - any failure is a rejection, not a crash
        proposal.problems.append(f"evaluation failed: {type(exc).__name__}: {exc}")
        return None

    if array.shape != (len(frame),):
        proposal.problems.append(f"wrong shape {array.shape}")
        return None
    if not np.isfinite(array).any():
        proposal.problems.append("all values non-finite")
        return None
    if np.nanstd(array) == 0:
        proposal.problems.append("constant feature")
        return None

    proposal.compiled = True
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def evaluate_features(proposals: Sequence[FeatureProposal], split,
                      detector_factory: Callable[..., Any], target_fpr: float,
                      *, min_gain: float = 0.005) -> pd.DataFrame:
    """Compile each proposal, retrain with it, and keep only the ones that help.

    One feature at a time against the same baseline. Accepting a batch would make it
    impossible to say which of them did anything, and a batch that helps on net can easily
    contain a feature that hurts.
    """
    from ..defend.thresholds import threshold_for_budget

    train, calibration, test = split.train, split.calibration, split.test
    if int(train["is_fraud"].sum()) < 20 or int(test["is_fraud"].sum()) < 20:
        return pd.DataFrame()

    baseline = detector_factory().fit(train, calibration)
    y = test["is_fraud"].to_numpy().astype(int)
    scores = baseline.predict_proba(test)
    thr = threshold_for_budget(scores[y == 0], target_fpr)
    recall_before = float((scores[y == 1] >= thr).mean())

    for proposal in proposals:
        proposal.recall_before = recall_before
        # Compiled against every partition, because a feature that only exists in training
        # is not a feature. The expression is evaluated per-frame rather than once on the
        # concatenation so a mistake cannot leak information across the split boundary.
        columns = {}
        ok = True
        for name, part in (("train", train), ("cal", calibration), ("test", test)):
            values = compile_feature(proposal, part) if ok else None
            if values is None:
                ok = False
                continue
            columns[name] = values
        if not ok:
            continue
        # compile_feature appends a problem per partition on failure; dedupe the success path
        proposal.problems = [p for p in proposal.problems if p]

        augmented = {}
        for name, part in (("train", train), ("cal", calibration), ("test", test)):
            frame = part.copy()
            frame[f"f_{proposal.name}"] = columns[name]
            augmented[name] = frame

        try:
            detector = detector_factory().fit(augmented["train"], augmented["cal"])
            new_scores = detector.predict_proba(augmented["test"])
        except Exception as exc:  # noqa: BLE001
            proposal.problems.append(f"retrain failed: {type(exc).__name__}")
            continue

        new_thr = threshold_for_budget(new_scores[y == 0], target_fpr)
        proposal.recall_after = float((new_scores[y == 1] >= new_thr).mean())
        proposal.accepted = proposal.delta >= min_gain

    return pd.DataFrame([p.to_dict() for p in proposals])


def accept_rate(frame: pd.DataFrame, column: str = "accepted") -> Dict[str, Any]:
    """The headline for both agents: how often does the proposal survive contact."""
    if frame.empty or column not in frame.columns:
        return {"proposed": 0, "accepted": 0, "accept_rate": float("nan")}
    n = len(frame)
    accepted = int(frame[column].sum())
    return {
        "proposed": n,
        "accepted": accepted,
        "accept_rate": round(accepted / n, 4) if n else float("nan"),
    }

"""Feeding the loop's output back into Identify and into the control library.

The claim this module has to earn
---------------------------------
The whole submission rests on the three pillars being one loop rather than three stages in a
row. Generation feeding the defence is easy and was already true. The return path - the
defence's failures changing what the red team believes is worth attacking, and changing the
taxonomy itself - was not. Tactics were discovered, scored, printed in a table, and then
thrown away at the end of the run. That is an open loop with a diagram of a closed one.

Two functions close it.

**Back to the taxonomy.** A surviving tactic is a new attack vector: a named combination of
rail, channel, victim cohort, counterparty structure and evasion levers that beat the
deployed model. It is emitted as a library entry, put through the same
:meth:`AttackLibrary.validate` gate every hand-authored vector passes, and written to
``discovered_vectors.yaml`` with provenance - which round found it, which parent vector it
mutated from, what recall it achieved and what it cost to run. A future run can load that
file and start where this one finished.

**Back to the controls.** Every vector in the library declares the controls that ought to
stop it. For each surviving tactic this module reports which of those controls the tactic
actually defeated and which held, which converts a co-evolution log into something a bank
can act on: a ranked list of controls that a machine-discovered attacker walked through. A
control gap is more useful than a recall number, because a recall number tells you that you
have a problem and a control gap tells you where the money should go.

On the mapping
--------------
:data:`CONTROL_DEFEATED_BY` is a hand-authored map from a lever to the named controls it
neutralises, and it is deliberately conservative. A control not listed against any lever the
tactic pulled is reported as **intact**, never as defeated. Over-claiming here would be the
worst kind of error in this report - a red team marking its own homework - so the bias runs
the other way and the tactic has to earn each entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

from ..identify.library import AttackLibrary, AttackVector

#: Which named controls each attacker lever defeats. Keys are evasion lever names from
#: ``EVASION_MOVES`` and structural gene names from :class:`~redteam.loop.strategy.StrategyGenes`.
CONTROL_DEFEATED_BY: Dict[str, Tuple[str, ...]] = {
    # --- cosmetic levers ---------------------------------------------------------------
    "payee_name_match_score": (
        "confirmation_of_payee", "strict_legal_name_match", "unicode_normalisation",
        "full_address_verification_ui",
    ),
    "_payee_reg_lag_s": (
        "payee_cooling_off", "payment_cooling_off", "first_payment_hold",
        "address_book_whitelisting",
    ),
    "screen_share_active": (
        "screen_share_payment_block", "remote_access_detection", "overlay_detection",
    ),
    "remote_access_app_detected": (
        "remote_access_detection", "screen_share_payment_block",
        "accessibility_payment_block",
    ),
    "call_in_progress": (
        "call_in_progress_hold", "scam_interstitial", "in_app_scam_interstitial",
        "directional_language_ui",
    ),
    "ip_asn_is_hosting": ("asn_reputation", "onboarding_velocity_by_asn"),
    "vpn_or_proxy": ("asn_reputation", "merchant_geo_binding"),
    "form_fill_duration_s": (
        "second_order_behavioural_stats", "behavioural_change_point_detection",
    ),
    "iso20022_field_entropy": (
        "remittance_field_anomaly_model", "schema_semantic_validation",
    ),
    "amount": ("mandate_amount_caps", "mcc_specific_step_up"),
    "hour": ("cutoff_aware_risk_weighting", "24x7_review_coverage"),
    "is_night": ("cutoff_aware_risk_weighting",),

    # --- structural genes --------------------------------------------------------------
    "aged_payee_share": (
        "payee_reputation_graph", "inbound_mule_scoring", "thin_file_velocity_model",
        "dormant_to_active_monitoring", "federated_mule_intelligence",
    ),
    "seasoning_depth": (
        "payee_reputation_graph", "inbound_mule_scoring", "thin_file_velocity_model",
        "sudden_inbound_fanin_model", "first_payment_hold",
    ),
    "n_mules": (
        "fanin_account_freeze", "payee_fanin_clustering", "sudden_inbound_fanin_model",
        "fanin_fanout_graph_features",
    ),
    "mule_reuse": ("payee_fanin_clustering", "rapid_passthrough_hold"),
    "structuring": (
        "mandate_amount_caps", "intra_session_cumulative_limits", "dual_authorisation",
        "gift_card_velocity_caps", "bin_velocity_limits", "offramp_velocity_caps",
    ),
    "spacing_s": ("intra_session_cumulative_limits", "temporal_motif_detection"),
    "rail": ("corridor_risk_scoring", "cross_border_beneficiary_screening"),
    "channel": ("verified_support_number_directory", "out_of_band_callback"),
    "cohort": ("intra_household_behavioural_profiling", "parental_step_up"),
    "payee_psp": ("consortium_graph_sharing", "federated_signals", "acquirer_kyb_refresh"),
}


def control_gap(levers: Iterable[str], vector: Optional[AttackVector]) -> pd.DataFrame:
    """For one tactic, which of its parent vector's controls held and which did not.

    Returns an empty frame when the parent vector declares no controls, rather than
    inventing a verdict about a control set that does not exist.
    """
    if vector is None or not vector.controls:
        return pd.DataFrame(columns=["control", "status", "defeated_by"])

    defeated: Dict[str, List[str]] = {}
    for lever in levers:
        for control in CONTROL_DEFEATED_BY.get(lever, ()):
            defeated.setdefault(control, []).append(lever)

    rows = []
    for control in vector.controls:
        by = defeated.get(control)
        rows.append({
            "control": control,
            "status": "defeated" if by else "intact",
            "defeated_by": ", ".join(sorted(set(by))) if by else "",
        })
    return pd.DataFrame(rows)


def control_gap_summary(tactics: Sequence[Tuple[str, Sequence[str], Optional[AttackVector]]]
                        ) -> pd.DataFrame:
    """Aggregate the per-tactic control verdicts into a ranked defensive roadmap.

    Ranked by how often a control was walked through, because a control that failed against
    six surviving tactics is a different order of problem from one that failed against one.
    """
    frames = []
    for label, levers, vector in tactics:
        gap = control_gap(levers, vector)
        if gap.empty:
            continue
        gap = gap.assign(tactic=label, vector_id=vector.id if vector else "")
        frames.append(gap)
    if not frames:
        return pd.DataFrame(columns=["control", "times_defeated", "times_intact",
                                     "defeated_by", "example_tactic"])

    combined = pd.concat(frames, ignore_index=True)
    out = (
        combined.assign(defeated=(combined["status"] == "defeated").astype(int))
        .groupby("control")
        .agg(
            times_defeated=("defeated", "sum"),
            times_tested=("defeated", "size"),
            defeated_by=("defeated_by", lambda s: ", ".join(
                sorted({x for v in s for x in v.split(", ") if x}))),
            example_tactic=("tactic", "first"),
        )
        .reset_index()
    )
    out["times_intact"] = out["times_tested"] - out["times_defeated"]
    return out.sort_values(["times_defeated", "times_tested"],
                           ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Back into the taxonomy
# --------------------------------------------------------------------------------------

#: Signals a discovered vector declares, chosen by which levers it pulled. Every entry must
#: be an observable schema column or ``AttackLibrary.validate`` rejects the vector - which
#: is exactly the gate this is meant to pass through rather than around.
_LEVER_SIGNALS: Dict[str, Tuple[str, ...]] = {
    "payee_name_match_score": ("payee_name_match_score",),
    "_payee_reg_lag_s": ("payee_added_minutes_ago", "payee_age_days"),
    "screen_share_active": ("screen_share_active",),
    "remote_access_app_detected": ("remote_access_app_detected",),
    "call_in_progress": ("call_in_progress",),
    "ip_asn_is_hosting": ("ip_asn_is_hosting",),
    "vpn_or_proxy": ("vpn_or_proxy",),
    "form_fill_duration_s": ("form_fill_duration_s", "keystroke_flight_cv"),
    "iso20022_field_entropy": ("iso20022_field_entropy",),
    "amount": ("amount", "account_balance_ratio"),
    "hour": ("hour", "is_night"),
    "is_night": ("is_night",),
    "aged_payee_share": ("payee_prior_txn_count", "payee_account_age_days"),
    "seasoning_depth": ("payee_inbound_amount_24h", "payee_prior_txn_count"),
    "n_mules": ("payee_inbound_unique_payers_24h",),
    "mule_reuse": ("payee_inbound_unique_payers_24h", "payee_outbound_ratio_24h"),
    "structuring": ("amount", "payee_inbound_amount_24h"),
    "spacing_s": ("hour",),
    "rail": ("rail", "is_irrevocable_rail"),
    "channel": ("channel",),
    "cohort": ("customer_age_band_ord", "customer_digital_literacy"),
    "payee_psp": ("payee_psp",),
}


@dataclass
class DiscoveredVector:
    """A surviving tactic, restated as a taxonomy entry with its provenance attached."""

    vector: AttackVector
    provenance: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        v = self.vector
        return {
            "id": v.id,
            "name": v.name,
            "family": v.family,
            "description": v.description,
            "genai_enablers": list(v.genai_enablers),
            "kill_chain": list(v.kill_chain),
            "rails": list(v.rails),
            "channels": list(v.channels),
            "victim": v.victim,
            "signals": [str(s) for s in v.signals],
            "controls": [str(c) for c in v.controls],
            "simulated": False,
            "severity": int(v.severity),
            "prevalence": int(v.prevalence),
            "detection_difficulty": int(v.detection_difficulty),
            "discovered_by": {
                **self.provenance,
                "levers": [str(x) for x in self.provenance.get("levers", [])],
            },
        }


def discovered_vector(*, parent: Optional[AttackVector], parent_id: str, label: str,
                      levers: Sequence[str], round_index: int, recall: float,
                      roi: float, cost: float, index: int) -> Optional[DiscoveredVector]:
    """Turn a surviving genome into a library entry, or return ``None`` if it is not one.

    A tactic only earns an entry if it beat the deployed detector on a meaningful share of
    its campaign. A genome that survived the round's ranking while still being caught nine
    times in ten is a search artefact, not a discovery, and writing it into the taxonomy
    would dilute the file that the whole Identify pillar rests on.
    """
    if parent is None or recall > 0.6:
        return None

    signals: List[str] = []
    for lever in levers:
        for signal in _LEVER_SIGNALS.get(lever, ()):
            if signal not in signals:
                signals.append(signal)
    if not signals:
        return None

    # Inherit the parent's controls: they are the controls that were supposed to stop this
    # and demonstrably did not, which is the useful thing to record against a discovery.
    vector = AttackVector(
        id=f"DISCOVERED-R{round_index}-{index:02d}-{_slug(parent_id)}",
        name=f"Evolved {parent.name} ({label})",
        family=parent.family,
        description=(
            f"Machine-discovered variant of {parent_id} found by the closed loop in round "
            f"{round_index}. Evades the deployed detector on "
            f"{(1.0 - recall) * 100:.0f}% of its payments by combining: {label}. "
            f"Attacker return on tooling spend {roi:+.2f} at a cost of INR {cost:,.0f}."
        ),
        genai_enablers=list(parent.genai_enablers) or ["automated_parameter_search"],
        kill_chain=list(parent.kill_chain),
        rails=list(parent.rails),
        channels=list(parent.channels),
        victim=parent.victim,
        signals=signals,
        controls=list(parent.controls),
        # Not marked simulated: it has no generator of its own, it is a parameterisation of
        # the parent's. Marking it simulated would fail validation, and rightly so.
        simulated=False,
        severity=min(parent.severity + 1, 5),
        prevalence=parent.prevalence,
        detection_difficulty=5,
    )
    return DiscoveredVector(
        vector=vector,
        provenance={
            "round": round_index,
            "parent_vector": parent_id,
            "levers": list(levers),
            "recall_against_deployed_model": round(float(recall), 4),
            "attacker_roi": round(float(roi), 4) if np.isfinite(roi) else None,
            "attacker_cost_inr": round(float(cost), 2),
        },
    )


def validate_and_emit(discoveries: Sequence[DiscoveredVector],
                      library: AttackLibrary) -> Tuple[List[DiscoveredVector], List[str], str]:
    """Put the discoveries through the library's own validator before they are written.

    Returns the accepted entries, the rejection messages, and the YAML text. The rejections
    are returned rather than swallowed: a generator that silently drops its own invalid
    output is indistinguishable from one that never produced any, and the accept/reject rate
    is a result worth publishing.
    """
    accepted: List[DiscoveredVector] = []
    problems: List[str] = []
    existing = set(library.by_id)

    for discovery in discoveries:
        if discovery.vector.id in existing:
            problems.append(f"{discovery.vector.id}: duplicate of an existing vector")
            continue
        probe = AttackLibrary(
            version=library.version, updated=library.updated,
            kill_chain=library.kill_chain, families=library.families,
            vectors=library.vectors + [discovery.vector],
        )
        # Only failures naming this vector matter; a pre-existing complaint about a
        # hand-authored entry is not this discovery's fault and must not reject it.
        failures = [p for p in probe.validate() if p.startswith(discovery.vector.id)]
        if failures:
            problems.extend(failures)
            continue
        accepted.append(discovery)
        existing.add(discovery.vector.id)

    document = {
        "version": f"{library.version}+discovered",
        "source": "redteam.loop.coevolution",
        "note": (
            "Vectors discovered by the closed loop, not hand authored. Each one is a "
            "parameterisation of an existing generator that evaded the deployed detector, "
            "validated against the same schema gate as the hand-authored library."
        ),
        "vectors": [d.to_dict() for d in accepted],
    }
    return accepted, problems, yaml.safe_dump(document, sort_keys=False, width=100)


def _slug(vector_id: str) -> str:
    return vector_id.replace("_", "-").upper()[:28]


# --------------------------------------------------------------------------------------
# Cross-round transfer
# --------------------------------------------------------------------------------------

def transfer_matrix(rows: Sequence[Dict[str, object]]) -> pd.DataFrame:
    """Recall of each round's tactics against each round's detector, as a square table.

    The diagonal is a tactic against the model it was evolved to beat, so it should be low.
    What matters is the column *to the right of* the diagonal: a tactic that still works
    against a detector trained three rounds later is a durable capability, and one that
    stops working is evidence the defence actually generalised rather than memorised.
    """
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return (
        frame.pivot_table(index="tactic_round", columns="detector_round",
                          values="recall", aggfunc="mean")
        .round(4)
        .rename_axis(index="tactic from round", columns="scored by detector from round")
    )

"""Identity fabrication: the fraudster is the account holder.

This family cannot be expressed as a mutation of a victim's transaction, because there is
no victim. The account itself is the weapon, so the generator mints its own synthetic
customers with their own account identifiers and their own payment history.

That history matters. A synthetic identity bust-out is not a burst of anomalous payments;
it is months of *impeccable* behaviour followed by a burst. The cultivation payments are
emitted with ``is_fraud = 0`` on purpose - they really are ordinary payments, and a
defence that flags them is generating false positives, not early warnings. They are tagged
with the vector id so evaluation can measure early-warning lift separately from the
headline detection metrics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ...identify.library import AttackVector
from .base import (
    SYNTHETIC_ID_BLOCK,
    AttackContext,
    AttackGenerator,
    finalise,
    reset_labels,
    select_victims,
    set_rail,
    template_rows,
    to_counterfeit_merchant,
    to_mule,
)
from .profiles import clamp_timestamps

CASH_LIKE_MCCS = (5947, 6051, 5732, 5815, 6211)


class SyntheticIdentityGenerator(AttackGenerator):
    """Covers the four simulated onboarding-abuse vectors."""

    name = "identity"
    vector_ids = (
        "ID-SYNTHETIC-BUSTOUT",
        "ID-GAN-DOCUMENT-FORGERY",
        "ID-EKYC-CAMERA-INJECTION",
        "ID-RPPG-LIVENESS-SPOOF",
    )

    def generate(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        if n_events <= 0:
            return pd.DataFrame()
        if vector.id == "ID-SYNTHETIC-BUSTOUT":
            return self._bustout(ctx, vector, n_events)
        return self._onboarding_bypass(ctx, vector, n_events)

    # -- long-horizon cultivation then bust-out -----------------------------------------
    def _bustout(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        rng = ctx.rng
        n_cultivation = rng.integers(5, 16, n_events)
        n_bust = rng.integers(3, 10, n_events)

        frames: List[pd.DataFrame] = []
        for k in range(n_events):
            campaign = ctx.campaign_id(vector.id)
            ident = _mint_identity(ctx, k)
            horizon = float(ctx.cfg.benign.n_days)
            opened = _account_open_day(ctx, horizon_fraction=(0.0, 0.35))
            # Split the account's remaining life into a long cultivation phase and a short
            # terminal burst, expressed relative to when the account was opened.
            life = max(horizon - opened - 0.5, 1.0)

            cult = self._phase(
                ctx, vector, ident, campaign,
                n=int(n_cultivation[k]),
                day_range=(opened + 0.5, opened + 0.65 * life),
                amount_range=(180.0, 3_500.0),
                counterparty="keep",
                fraud=False,
                opened_day=opened,
            )
            # The bust-out: every available line drawn at once, into cash equivalents.
            bust_start = opened + float(rng.uniform(0.72, 0.94)) * life
            bust = self._phase(
                ctx, vector, ident, campaign,
                n=int(n_bust[k]),
                day_range=(bust_start, min(bust_start + rng.uniform(0.05, 1.5), horizon - 0.01)),
                amount_range=(18_000.0, 420_000.0),
                counterparty="cash_like",
                fraud=True,
                opened_day=opened,
            )
            frames.extend([cult, bust])

        return pd.concat(frames, ignore_index=True)

    # -- eKYC bypass then rapid extraction ----------------------------------------------
    def _onboarding_bypass(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        rng = ctx.rng
        counts = rng.integers(2, 8, n_events)
        frames: List[pd.DataFrame] = []
        for k in range(n_events):
            campaign = ctx.campaign_id(vector.id)
            ident = _mint_identity(ctx, k)
            horizon = float(ctx.cfg.benign.n_days)
            opened = _account_open_day(ctx, horizon_fraction=(0.02, 0.85))
            frames.append(
                self._phase(
                    ctx, vector, ident, campaign,
                    n=int(counts[k]),
                    day_range=(min(opened + 0.01, horizon - 0.02),
                               min(opened + rng.uniform(0.2, 6.0), horizon - 0.01)),
                    amount_range=(4_000.0, 260_000.0),
                    counterparty="mule",
                    fraud=True,
                    opened_day=opened,
                )
            )
        out = pd.concat(frames, ignore_index=True)
        _apply_bypass_artifacts(ctx, out, vector.id)
        return out

    # -- shared row construction ----------------------------------------------------------
    def _phase(self, ctx: AttackContext, vector: AttackVector, ident: Dict[str, object],
               campaign: str, *, n: int, day_range: Tuple[float, float],
               amount_range: Tuple[float, float], counterparty: str, fraud: bool,
               opened_day: float) -> pd.DataFrame:
        if n <= 0:
            return pd.DataFrame()
        rng = ctx.rng
        # Base rows inherit realistic device, behavioural and session telemetry from real
        # traffic; only identity-bearing fields are overwritten below.
        donors = select_victims(ctx, n, by_vulnerability=0.0, corporate=False)
        df = template_rows(ctx, donors)
        df = reset_labels(df)

        set_rail(ctx, df, ["UPI_P2M", "CARD_CNP", "IMPS", "NEFT"] if not fraud
                 else ["CARD_CNP", "IMPS", "NEFT", "UPI_P2M"])

        days = np.sort(rng.uniform(day_range[0], day_range[1], n))
        ts = pd.Timestamp(ctx.cfg.benign.start_date) + pd.to_timedelta(
            (days * 86400 + rng.uniform(0, 86400, n) * 0.0).astype("int64"), unit="s"
        )
        clamp_timestamps(ctx, df, pd.DatetimeIndex(ts))

        df["customer_id"] = ident["customer_id"]
        df["payer_account_id"] = ident["account_id"]
        df["_customer_idx"] = -1
        df["payer_psp"] = ident["psp"]
        df["customer_is_corporate"] = 0
        df["customer_age_band_ord"] = ident["age_band_ord"]
        df["customer_digital_literacy"] = ident["digital_literacy"]
        df["prior_fraud_reports"] = 0
        df["kyc_level"] = ident["kyc_level"]
        # Tenure is measured from the fabricated account's own opening date.
        df["customer_tenure_days"] = np.maximum(days - opened_day, 0.0).round(2)
        df["account_balance_ratio"] = np.round(rng.uniform(0.05, 0.9, n), 4)

        amount = np.exp(rng.uniform(np.log(amount_range[0]), np.log(amount_range[1]), n))
        df["amount"] = np.round(amount, 2)

        if counterparty == "mule":
            to_mule(ctx, df)
        elif counterparty == "cash_like":
            to_counterfeit_merchant(ctx, df, mccs=CASH_LIKE_MCCS, domain_age=(30.0, 2500.0),
                                    reuse=0.4)
        # "keep" leaves the donor's real merchant in place, which is the point: during
        # cultivation the synthetic identity shops at ordinary merchants.

        df["initiated_by_agent"] = 0
        df["intent_token_present"] = 0
        df["intent_signature_valid"] = 0

        df = finalise(df, vector, campaign, "synthetic_identity")
        if not fraud:
            # Cultivation is genuinely non-fraudulent payment activity. Tagged, not labelled.
            df["is_fraud"] = 0
            df["fraud_type"] = "none"
        return df


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _mint_identity(ctx: AttackContext, k: int) -> Dict[str, object]:
    seq = ctx.identity_seq()
    rng = ctx.rng
    # Synthetic identities cluster at PSPs with the weakest onboarding controls.
    maturity = ctx.pop.psps["mule_control_maturity"].to_numpy()
    w = np.exp(-2.0 * maturity)
    psp = str(rng.choice(ctx.pop.psps["psp_id"].to_numpy(), p=w / w.sum()))
    # A synthetic identity that succeeds at onboarding is, from the bank's point of view,
    # indistinguishable from a real new customer - that is the entire point of the fraud. So
    # it is issued ordinary ids above the real population rather than ids stamped "SYN",
    # which would let the defence resolve the whole family from a string prefix.
    n = len(ctx.pop.customers) + SYNTHETIC_ID_BLOCK + seq
    return {
        "customer_id": f"C{n:07d}",
        "account_id": f"A{n:07d}",
        "psp": psp,
        "age_band_ord": int(rng.choice([1, 2, 3], p=[0.45, 0.35, 0.20])),
        "digital_literacy": float(np.clip(rng.normal(0.82, 0.10), 0.1, 0.99)),
        "kyc_level": int(rng.choice([1, 2, 3], p=[0.30, 0.55, 0.15])),
    }


def _account_open_day(ctx: AttackContext, horizon_fraction=(0.0, 0.5)) -> float:
    lo, hi = horizon_fraction
    return float(ctx.rng.uniform(lo, hi) * ctx.cfg.benign.n_days)


def _apply_bypass_artifacts(ctx: AttackContext, df: pd.DataFrame, vector_id: str) -> None:
    """Residual artefacts of the specific liveness / document bypass used at onboarding."""
    n = len(df)
    if n == 0:
        return
    rng = ctx.rng

    if vector_id == "ID-GAN-DOCUMENT-FORGERY":
        # Generated documents OCR *cleanly* - too cleanly. Template-perfect fonts and
        # spacing push confidence into a band real photographed documents rarely reach.
        m = ctx.tell(n, 0.85)
        val = df["doc_ocr_confidence"].to_numpy().astype(float)
        val[m] = rng.uniform(0.965, 0.999, int(m.sum()))
        df["doc_ocr_confidence"] = np.round(val, 4)
        m2 = ctx.tell(n, 0.5)
        liv = df["liveness_score"].to_numpy().astype(float)
        liv[m2] = rng.uniform(0.55, 0.85, int(m2.sum()))
        df["liveness_score"] = np.round(liv, 4)

    elif vector_id == "ID-EKYC-CAMERA-INJECTION":
        # Bypassing the physical sensor requires a controllable client: emulators, rooted
        # handsets, virtualised desktops. The frame itself is clean; the platform is not.
        m = ctx.tell(n, 0.72)
        df["device_is_emulator"] = np.where(m, 1, df["device_is_emulator"].to_numpy())
        m2 = ctx.tell(n, 0.45)
        df["device_is_rooted"] = np.where(m2, 1, df["device_is_rooted"].to_numpy())
        m3 = ctx.tell(n, 0.55)
        df["vpn_or_proxy"] = np.where(m3, 1, df["vpn_or_proxy"].to_numpy())
        liv = df["liveness_score"].to_numpy().astype(float)
        m4 = ctx.tell(n, 0.8)
        liv[m4] = rng.uniform(0.88, 0.99, int(m4.sum()))
        df["liveness_score"] = np.round(liv, 4)
        bio = df["biometric_match_score"].to_numpy().astype(float)
        bio[m4] = rng.uniform(0.90, 0.995, int(m4.sum()))
        df["biometric_match_score"] = np.round(bio, 4)

    elif vector_id == "ID-RPPG-LIVENESS-SPOOF":
        # The hardest case by construction: the deepfake carries a coherent pulse and
        # posts liveness scores indistinguishable from a live capture. Nothing in the
        # biometric block helps, so detection has to come from behaviour and money flow.
        liv = df["liveness_score"].to_numpy().astype(float)
        bio = df["biometric_match_score"].to_numpy().astype(float)
        m = ctx.tell(n, 0.9)
        liv[m] = rng.uniform(0.93, 0.998, int(m.sum()))
        bio[m] = rng.uniform(0.92, 0.998, int(m.sum()))
        df["liveness_score"] = np.round(liv, 4)
        df["biometric_match_score"] = np.round(bio, 4)
        doc = df["doc_ocr_confidence"].to_numpy().astype(float)
        doc[m] = rng.uniform(0.86, 0.98, int(m.sum()))
        df["doc_ocr_confidence"] = np.round(doc, 4)

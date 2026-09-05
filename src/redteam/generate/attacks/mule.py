"""Mule networks: the receiving side of every push-payment scam.

This generator runs *after* the proceeds-taking attacks and deliberately reuses the shared
:class:`~redteam.generate.attacks.base.MuleRegistry`. First-hop accounts here are the same
accounts that received APP scam, QR-tampering and agentic-hijack proceeds, so the
laundering subgraph is genuinely connected to the predicate offences rather than being a
detached synthetic blob. Without that shared state the graph features would be measuring
an artefact of the simulator.

Under the UK PSR mandatory reimbursement regime the receiving PSP carries half the loss,
so inbound detection on this family is a direct balance-sheet line rather than a courtesy
to the sending bank.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ...identify.library import AttackVector
from .base import (
    AttackContext,
    AttackGenerator,
    finalise,
    reset_labels,
    select_victims,
    set_rail,
    template_rows,
)
from .profiles import clamp_timestamps

# Structuring thresholds that laundering amounts hug from below.
REPORTING_THRESHOLDS = (50_000.0, 100_000.0, 200_000.0)


class MuleGenerator(AttackGenerator):
    name = "mule"
    vector_ids = ("MULE-LAYERING-FANOUT", "MULE-GRAPH-EVASIVE-RING")

    def generate(self, ctx: AttackContext, vector: AttackVector, n_events: int) -> pd.DataFrame:
        if n_events <= 0:
            return pd.DataFrame()
        evasive = vector.id == "MULE-GRAPH-EVASIVE-RING"
        rng = ctx.rng
        frames: List[pd.DataFrame] = []

        for _ in range(n_events):
            campaign = ctx.campaign_id(vector.id)
            first_hop = self._first_hop_account(ctx)
            ring_id = ctx.mules.ring_of[first_hop]
            ring = [a for a, r in ctx.mules.ring_of.items() if r == ring_id]
            if len(ring) < 3:
                ring = ctx.mules.new_ring(ctx.pop, rng, size=int(rng.integers(6, 20)))

            hops = self._build_hops(ctx, first_hop, ring, evasive=evasive)
            if hops.empty:
                continue
            frames.append(finalise(hops, vector, campaign, "mule_layering"))

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _first_hop_account(self, ctx: AttackContext) -> str:
        """Prefer an account that has actually received predicate-offence proceeds."""
        if ctx.mules.accounts:
            return str(ctx.rng.choice(ctx.mules.accounts))
        return ctx.mules.new_ring(ctx.pop, ctx.rng, size=int(ctx.rng.integers(6, 20)))[0]

    def _build_hops(self, ctx: AttackContext, first_hop: str, ring: List[str],
                    *, evasive: bool) -> pd.DataFrame:
        rng = ctx.rng
        n_out = int(rng.integers(3, 9) if not evasive else rng.integers(2, 5))
        depth = int(rng.integers(1, 4))
        # Keep a fan-out tree inside the remaining fraud budget: breadth^depth explodes fast.
        while depth > 1 and n_out ** depth > max(ctx.budget_hint, 4):
            depth -= 1
        n_out = ctx.cap(n_out, minimum=2)

        rows: List[pd.DataFrame] = []
        # Classic layering passes funds through in minutes. The evasive variant deliberately
        # dwells for hours and randomises hop timing to break temporal-motif detection.
        dwell_s = rng.uniform(30, 900) if not evasive else rng.uniform(3 * 3600, 40 * 3600)
        base_s = ctx.episode_start(dwell_s * depth, night_bias=0.20)

        senders = [first_hop]
        elapsed = 0.0
        for level in range(depth):
            next_senders: List[str] = []
            for sender in senders:
                k = max(1, int(rng.poisson(n_out)))
                receivers = [str(x) for x in rng.choice(ring, size=k, replace=True) if x != sender]
                if not receivers:
                    continue
                df = self._hop_frame(ctx, sender, receivers, base_s, elapsed, evasive=evasive)
                rows.append(df)
                next_senders.extend(receivers)
            elapsed += dwell_s * rng.uniform(0.6, 1.6)
            senders = list(dict.fromkeys(next_senders))[: 6 if not evasive else 3]
            if not senders:
                break

        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def _hop_frame(self, ctx: AttackContext, sender: str, receivers: List[str],
                   base_s: float, elapsed: float, *, evasive: bool) -> pd.DataFrame:
        rng = ctx.rng
        n = len(receivers)
        # Inherit realistic session/device telemetry from ordinary traffic; the mule
        # operator is using a normal banking app on a normal handset.
        donors = select_victims(ctx, n, by_vulnerability=0.0)
        df = template_rows(ctx, donors)
        df = reset_labels(df)
        set_rail(ctx, df, ["UPI_P2P", "IMPS", "RTP_FEDNOW", "SEPA_INST"])

        jitter = rng.uniform(0, 600 if not evasive else 9 * 3600, n)
        ts = (pd.Timestamp(ctx.cfg.benign.start_date)
              + pd.to_timedelta((base_s + elapsed + jitter).astype("int64"), unit="s"))
        clamp_timestamps(ctx, df, pd.DatetimeIndex(ts))

        # The operator is a person holding an ordinary account, so the customer id follows the
        # same A<n>/C<n> pairing every other row uses. A "MULEOP-" prefix would make every
        # layering row identifiable from its customer id alone - fraud with probability 1,
        # before a single behavioural feature is computed.
        df["customer_id"] = "C" + str(sender)[1:]
        df["payer_account_id"] = sender
        df["_customer_idx"] = -1
        df["_merchant_idx"] = -1
        df["_peer_idx"] = -1
        df["payer_psp"] = ctx.mules.psp_of[sender]
        df["payee_account_id"] = receivers
        df["payee_psp"] = [ctx.mules.psp_of[r] for r in receivers]
        df["mule_ring_id"] = [ctx.mules.ring_of[r] for r in receivers]
        df["payee_account_age_days"] = np.round([ctx.mules.opened_days_ago[r] for r in receivers], 1)
        df["merchant_id"] = ""
        df["mcc"] = 0
        df["payee_is_high_risk_category"] = 0
        df["merchant_domain_age_days"] = 3650.0
        df["_payee_reg_lag_s"] = np.round(rng.uniform(30, 6 * 3600, n), 1)
        df["_is_explore_payee"] = 1
        df["customer_tenure_days"] = np.round(
            [ctx.mules.opened_days_ago[sender]] * n, 1
        )
        # Per row, not per frame. Writing one scalar across a whole hop frame makes the frame
        # internally identical, and a block of rows agreeing exactly on four attributes is a
        # signature no real population produces - the model learns the block, not the pattern.
        df["customer_is_corporate"] = (rng.random(n) < 0.06).astype(int)
        df["prior_fraud_reports"] = rng.choice([0, 1, 2], size=n, p=[0.93, 0.055, 0.015])
        df["kyc_level"] = rng.choice([1, 2, 3], size=n, p=[0.5, 0.36, 0.14])
        df["initiated_by_agent"] = 0

        if evasive:
            # Amounts drawn to look like ordinary peer transfers rather than hugging a
            # threshold, and name matching cleaned up: optimised against a surrogate model.
            df["amount"] = np.round(np.exp(rng.normal(np.log(9_500), 1.05, n)), 2)
            df["payee_name_match_score"] = np.round(np.clip(rng.beta(7, 1.6, n), 0, 1), 4)
        else:
            # Structure just under a reporting threshold - the classic tell.
            threshold = np.array(rng.choice(REPORTING_THRESHOLDS, size=n))
            df["amount"] = np.round(threshold * rng.uniform(0.72, 0.985, n), 2)
            df["payee_name_match_score"] = np.round(np.clip(rng.beta(2.5, 3.0, n), 0, 1), 4)

        # A mule sweeps most of what it holds, so the ratio runs high - but U(0.55, 0.99) is a
        # band ordinary traffic almost never occupies, which turns a tendency into a tell.
        # Lognormal around 0.7 keeps the central tendency and restores the overlap.
        df["account_balance_ratio"] = np.round(
            np.clip(rng.lognormal(np.log(0.70), 0.55, n), 0.01, 3.0), 4
        )
        return df

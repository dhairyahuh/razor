"""Delegated-authority controls: agent identity, token binding, spend ceilings.

The intent guard next door verifies *what the user asked for*. These three controls verify
*who is asking* and *what the credential they hold is good for*. The distinction matters
because the intent guard's own coverage table reports 0% against three vectors, and all
three fail for the same reason: the intent chain is perfectly valid. A spoofed crawler
never presented a bad intent, it presented no intent and a borrowed name. A replayed
Shared Payment Token was minted by the real user. A stolen token spent inside its scope
contradicts nothing.

What is implemented here
------------------------
**Web Bot Auth (item 224).** RFC 9421 HTTP message signatures over a canonical set of
covered components, verified against an :class:`AgentRegistry` of enrolled Ed25519 public
keys. An impersonator can copy a user-agent string; it cannot sign for a key it does not
hold.

**Proof-of-possession token binding (item 223).** A Shared Payment Token carries the
thumbprint of the key it was issued to, in the manner of DPoP's ``jkt`` claim. Presenting
it from any other key fails, so a stolen or replayed token is worth nothing without the
private key that was bound to it at issuance.

**Per-transaction scoped tokens with a ceiling (item 225).** A token is minted for one
purchase with an explicit cap and scope rather than handed over as a standing credential.
This is the control that bounds the loss when the other two have already failed.

Why identity failure raises friction rather than declining
----------------------------------------------------------
Verification failure is not proof of impersonation. Key rotation that outpaces a verifier's
cache, a proxy that strips the Signature header and clock skew past the created-at window
all produce a genuine failure on an honest request; the benign generator carries that tail
deliberately. So an unverified assertion loses the trusted lane and earns a step-up, which
is what Web Bot Auth is actually for. The hard decline stays with the intent guard, where
the contradiction is provable.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .intent_guard import _b64u, _b64u_decode, generate_keypair, public_key_bytes

#: How far a signature's ``created`` timestamp may drift from the verifier's clock. RFC 9421
#: leaves this to the application; Web Bot Auth deployments use a small window because the
#: signature covers a single request rather than a session.
DEFAULT_SIGNATURE_WINDOW_S = 60

#: Covered components, in the order they are serialised. Fixed rather than negotiated: a
#: verifier that accepts whatever set the client nominates can be handed a signature over
#: nothing much and will happily verify it.
COVERED_COMPONENTS = ("@method", "@authority", "@path", "agent-id", "token-thumbprint")


def key_thumbprint(key: Ed25519PublicKey) -> str:
    """A short, stable identifier for a public key, as DPoP's ``jkt`` claim is."""
    return _b64u(hashlib.sha256(public_key_bytes(key)).digest())[:22]


# --------------------------------------------------------------------------------------
# Agent identity: Web Bot Auth over RFC 9421 message signatures
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RegisteredAgent:
    """An agent enrolled in the registry, with the authority it was granted."""

    agent_id: str
    public_key: Ed25519PublicKey
    allowed_scopes: Tuple[str, ...] = ("purchase",)
    per_txn_ceiling: float = 50_000.0
    reputation: float = 0.8


@dataclass
class SignedRequest:
    """The parts of a settlement request that identity verification reads."""

    method: str
    authority: str
    path: str
    agent_id: str
    token_thumbprint: str
    created: float
    signature: str

    def signing_input(self) -> bytes:
        """Canonical form of the covered components, plus the signature parameters.

        ``created`` and the component list are inside the signed bytes. If they were not,
        an attacker could replay a valid signature with a fresh timestamp, or narrow the
        covered set after the fact, and the verifier would accept both.
        """
        values = {
            "@method": self.method,
            "@authority": self.authority,
            "@path": self.path,
            "agent-id": self.agent_id,
            "token-thumbprint": self.token_thumbprint,
        }
        lines = [f'"{name}": {values[name]}' for name in COVERED_COMPONENTS]
        params = " ".join(f'"{n}"' for n in COVERED_COMPONENTS)
        lines.append(f'"@signature-params": ({params});created={int(self.created)}')
        return "\n".join(lines).encode("utf-8")


def sign_request(private_key: Ed25519PrivateKey, *, method: str = "POST",
                 authority: str = "psp.example", path: str = "/settle",
                 agent_id: str, token_thumbprint: str = "",
                 created: Optional[float] = None) -> SignedRequest:
    created = time.time() if created is None else created
    request = SignedRequest(
        method=method, authority=authority, path=path, agent_id=agent_id,
        token_thumbprint=token_thumbprint, created=created, signature="",
    )
    signature = _b64u(private_key.sign(request.signing_input()))
    return SignedRequest(
        method=method, authority=authority, path=path, agent_id=agent_id,
        token_thumbprint=token_thumbprint, created=created, signature=signature,
    )


class AgentRegistry:
    """Enrolled agents and their public keys, as a Web Bot Auth key directory."""

    def __init__(self, agents: Sequence[RegisteredAgent] = ()):
        self._agents: Dict[str, RegisteredAgent] = {a.agent_id: a for a in agents}
        self._seen: Set[Tuple[str, str]] = set()

    def register(self, agent: RegisteredAgent) -> None:
        self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> Optional[RegisteredAgent]:
        return self._agents.get(agent_id)

    def verify(self, request: SignedRequest, *, now: Optional[float] = None,
               window_s: int = DEFAULT_SIGNATURE_WINDOW_S) -> List[str]:
        """Reason codes for an identity assertion. Empty means the identity is proved."""
        now = time.time() if now is None else now
        agent = self._agents.get(request.agent_id)
        if agent is None:
            # An asserted identity that is not enrolled. Not a decline on its own; it means
            # the assertion buys nothing.
            return ["AGENT_NOT_ENROLLED"]
        if abs(now - request.created) > window_s:
            return ["AGENT_SIGNATURE_STALE"]
        try:
            agent.public_key.verify(_b64u_decode(request.signature), request.signing_input())
        except (InvalidSignature, ValueError, TypeError):
            # The identity was asserted and could not be signed for. This is the
            # impersonation fingerprint.
            return ["AGENT_SIGNATURE_INVALID"]
        replay_key = (request.agent_id, request.signature)
        if replay_key in self._seen:
            return ["AGENT_REQUEST_REPLAYED"]
        self._seen.add(replay_key)
        return []


# --------------------------------------------------------------------------------------
# Token binding and scoped tokens
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ScopedToken:
    """A Shared Payment Token minted for one purchase, bound to one key.

    ``holder_thumbprint`` is the proof-of-possession binding. ``max_amount`` and
    ``scope`` are the per-transaction ceiling: the token is authority for this purchase and
    nothing else, so a token that leaks is worth at most one bounded payment rather than
    the balance of the account.
    """

    token_id: str
    user_id: str
    holder_thumbprint: str
    max_amount: float
    currency: str
    scope: Tuple[str, ...]
    expires_at: float
    single_use: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "token_id": self.token_id, "user_id": self.user_id,
            "jkt": self.holder_thumbprint, "max_amount": self.max_amount,
            "currency": self.currency, "scope": list(self.scope),
            "expires_at": self.expires_at, "single_use": self.single_use,
        }


class TokenVault:
    """Mints scoped tokens and enforces binding, ceiling and single use on redemption."""

    def __init__(self):
        self._spent: Set[str] = set()
        self._issued: Dict[str, ScopedToken] = {}

    def mint(self, *, user_id: str, holder: Ed25519PublicKey, max_amount: float,
             currency: str = "INR", scope: Tuple[str, ...] = ("purchase",),
             ttl_s: int = 900, token_id: Optional[str] = None,
             now: Optional[float] = None) -> ScopedToken:
        now = time.time() if now is None else now
        thumb = key_thumbprint(holder)
        token = ScopedToken(
            token_id=token_id or f"SPT-{thumb[:10]}-{int(now)}",
            user_id=user_id,
            holder_thumbprint=thumb,
            max_amount=float(max_amount),
            currency=currency,
            scope=tuple(scope),
            expires_at=now + ttl_s,
        )
        self._issued[token.token_id] = token
        return token

    def redeem(self, token: ScopedToken, *, presenter: Ed25519PublicKey, amount: float,
               currency: str = "INR", requested_scope: Tuple[str, ...] = ("purchase",),
               now: Optional[float] = None) -> List[str]:
        """Reason codes for a redemption attempt. Empty means the spend is authorised."""
        now = time.time() if now is None else now
        reasons: List[str] = []

        # Proof of possession first. A token presented by a key it was not issued to is
        # stolen regardless of everything else about the request, which is the entire point
        # of binding it: theft of the credential is no longer sufficient.
        if key_thumbprint(presenter) != token.holder_thumbprint:
            reasons.append("TOKEN_NOT_BOUND_TO_PRESENTER")
        if now > token.expires_at:
            reasons.append("TOKEN_EXPIRED")
        if token.single_use and token.token_id in self._spent:
            reasons.append("TOKEN_ALREADY_SPENT")
        if amount > token.max_amount:
            reasons.append("TOKEN_CEILING_EXCEEDED")
        if currency != token.currency:
            reasons.append("TOKEN_CURRENCY_MISMATCH")
        if not set(requested_scope).issubset(set(token.scope)):
            reasons.append("TOKEN_SCOPE_VIOLATION")

        if not reasons:
            self._spent.add(token.token_id)
        return reasons


def demo_stack(seed: bytes = b"redteam-demo") -> Tuple[AgentRegistry, TokenVault,
                                                       Ed25519PrivateKey, RegisteredAgent]:
    """A registry, vault and one enrolled agent, keyed deterministically.

    Exists so the tests, the service layer and the written report all exercise the same
    configuration rather than three subtly different ones.
    """
    private, public = generate_keypair(seed=seed)
    agent = RegisteredAgent(agent_id="shopper-agent-1", public_key=public,
                            allowed_scopes=("purchase",), per_txn_ceiling=25_000.0)
    return AgentRegistry([agent]), TokenVault(), private, agent


# --------------------------------------------------------------------------------------
# Dataset-level application
# --------------------------------------------------------------------------------------

#: Reason codes the controls can raise across a transaction frame.
CONTROL_CODES = [
    "AGENT_IDENTITY_UNPROVEN", "AGENT_UNTRUSTED_LANE",
    "TOKEN_REPLAY_BLOCKED", "TOKEN_SCOPE_BLOCKED", "TOKEN_OVER_CEILING",
]

#: Codes whose violation is provable from the credential itself, so the payment is stopped.
#:
#: Token binding and scope are here because both are contradictions of a credential the
#: request itself presented: a token replayed after being marked single-use, or spent
#: outside the scope it was minted for. Neither requires an inference about intent, and
#: neither fires on honest agent traffic at all.
CONTROL_BLOCK_CODES = frozenset({"TOKEN_REPLAY_BLOCKED", "TOKEN_SCOPE_BLOCKED"})

#: Codes that cost the request its trusted lane and raise friction.
CONTROL_STEPUP_CODES = frozenset({"AGENT_IDENTITY_UNPROVEN", "AGENT_UNTRUSTED_LANE"})

#: The ceiling is deliberately in neither set above.
#:
#: A per-transaction cap sized from a population quantile is a rule dressed as a proof. By
#: construction the 99th percentile of legitimate agent payments declines one legitimate
#: agent payment in a hundred, and an amount being large is not a contradiction of anything
#: the request presented - a real scoped token is minted against the quoted cart, and a
#: settlement that overruns *that* is the amount check the intent guard already performs.
#: Adding a second one would double-count the same violation and buy a 1% false-decline
#: rate for it.
#:
#: What the ceiling genuinely buys is loss bounding once the other controls have already
#: failed, so it is reported by :func:`ceiling_loss_bound` as an exposure figure rather
#: than acted on as a verdict.
CEILING_QUANTILE = 0.99


def apply_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Evaluate the three delegated-authority controls across a transaction frame.

    Maps the schema's record of what a real deployment would observe onto the same reason
    codes the live implementations above emit, so the offline measurement and the online
    control speak one language - the same arrangement the intent guard uses.
    """
    agent = df["initiated_by_agent"].to_numpy() == 1
    n = len(df)

    asserted = agent & (df["agent_identity_asserted"].to_numpy() == 1)
    signed = agent & (df["agent_request_signature_valid"].to_numpy() == 1)

    # Web Bot Auth. An assertion that does not verify loses the trusted lane; asserting
    # nothing at all also loses it, without the implication of misconduct.
    identity_unproven = asserted & ~signed
    untrusted_lane = agent & ~asserted

    # Proof-of-possession binding. A token presented more than once cannot be a single-use
    # token bound to one purchase, so replay is visible without knowing who holds the key.
    replay = agent & (df["spt_reuse_count"].to_numpy() >= 2)
    scope = agent & (df["spt_scope_violation"].to_numpy() == 1)

    # Per-transaction ceiling, calibrated on legitimate agent traffic in this frame.
    legit_agent = df.loc[agent & (df["is_fraud"].to_numpy() == 0), "amount"]
    ceiling = float(legit_agent.quantile(CEILING_QUANTILE)) if len(legit_agent) > 50 else np.inf
    over_ceiling = agent & (df["amount"].to_numpy() > ceiling)

    flags = {
        "AGENT_IDENTITY_UNPROVEN": identity_unproven,
        "AGENT_UNTRUSTED_LANE": untrusted_lane,
        "TOKEN_REPLAY_BLOCKED": replay,
        "TOKEN_SCOPE_BLOCKED": scope,
        "TOKEN_OVER_CEILING": over_ceiling,
    }

    reasons = np.array([""] * n, dtype=object)
    for code, mask in flags.items():
        for i in np.where(mask)[0]:
            reasons[i] = f"{reasons[i]}|{code}" if reasons[i] else code

    blocked = np.zeros(n, dtype=int)
    for code in CONTROL_BLOCK_CODES:
        blocked |= flags[code].astype(int)
    step_up = np.zeros(n, dtype=int)
    for code in CONTROL_STEPUP_CODES:
        step_up |= flags[code].astype(int)

    out = df.copy()
    out["agent_control_blocked"] = blocked
    out["agent_control_stepup"] = step_up
    out["agent_control_reasons"] = reasons
    out["agent_control_ceiling"] = round(ceiling, 2) if np.isfinite(ceiling) else np.nan
    return out


def control_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vector measurement of what each control neutralises, which is item 225.

    Reported per control rather than in aggregate, because the interesting result is that
    they do not overlap: identity verification catches the impersonator and nothing else,
    binding catches the replay and nothing else, and the counterfeit storefront is
    untouched by all three because every credential it is handed is genuine.
    """
    agentic = df[df["initiated_by_agent"] == 1]
    if agentic.empty:
        return pd.DataFrame()

    def _rate(group: pd.DataFrame, code: str) -> float:
        return round(float(group["agent_control_reasons"].str.contains(code).mean()), 4)

    rows = []
    labels = agentic["attack_vector_id"].replace("", "(legitimate)")
    for vector_id, group in agentic.groupby(labels):
        rows.append(
            {
                "attack_vector_id": vector_id,
                "rows": len(group),
                "is_fraud": int(group["is_fraud"].sum()),
                "identity_unproven_rate": _rate(group, "AGENT_IDENTITY_UNPROVEN"),
                "untrusted_lane_rate": _rate(group, "AGENT_UNTRUSTED_LANE"),
                "token_binding_rate": _rate(group, "TOKEN_REPLAY_BLOCKED"),
                "scope_rate": _rate(group, "TOKEN_SCOPE_BLOCKED"),
                "over_ceiling_rate": _rate(group, "TOKEN_OVER_CEILING"),
                "any_control_rate": round(
                    float((group["agent_control_blocked"] | group["agent_control_stepup"]).mean()), 4
                ),
                "blocked_rate": round(float(group["agent_control_blocked"].mean()), 4),
            }
        )
    out = pd.DataFrame(rows).sort_values("any_control_rate", ascending=False)
    return out.reset_index(drop=True)


def ceiling_loss_bound(df: pd.DataFrame) -> pd.DataFrame:
    """What a per-transaction scoped token is worth to an attacker who already holds it.

    This is the question the other two controls cannot answer. Once an agent is genuinely
    hijacked and the credential it presents is genuinely its own, no amount of verification
    helps; all that is left is how much authority the credential carried. A standing
    delegation is worth the account, a per-purchase token with a cap is worth the cap.

    Reported as exposure rather than as a detection rate because that is what it is: no
    fraud is prevented, the loss per successful attempt is bounded. The counterfactual is
    the fraud actually generated, so the reduction is measured against what this attacker
    achieved rather than against a hypothetical worst case.
    """
    agentic = df[df["initiated_by_agent"] == 1]
    if agentic.empty:
        return pd.DataFrame()

    legit = agentic.loc[agentic["is_fraud"] == 0, "amount"]
    if len(legit) <= 50:
        return pd.DataFrame()
    ceiling = float(legit.quantile(CEILING_QUANTILE))

    fraud = agentic.loc[agentic["is_fraud"] == 1, "amount"].to_numpy().astype(float)
    if not fraud.size:
        return pd.DataFrame()

    # Standing delegation: the attacker takes what it took. Scoped token: every attempt is
    # truncated at the cap, so the same attempts realise less.
    unbounded = float(fraud.sum())
    bounded = float(np.minimum(fraud, ceiling).sum())
    return pd.DataFrame([
        {
            "ceiling": round(ceiling, 2),
            "ceiling_quantile": CEILING_QUANTILE,
            "agentic_fraud_attempts": int(fraud.size),
            "loss_standing_delegation": round(unbounded, 2),
            "loss_scoped_token": round(bounded, 2),
            "loss_reduction": round(unbounded - bounded, 2),
            "loss_reduction_pct": round(
                100.0 * (unbounded - bounded) / unbounded, 2
            ) if unbounded else np.nan,
            "legit_payments_capped_pct": round(
                100.0 * float((legit.to_numpy() > ceiling).mean()), 3
            ),
        }
    ])

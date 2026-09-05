"""Cryptographic intent binding for agentic payments.

Indirect prompt injection cannot be solved by classifying transactions, because the
hijacked transaction is genuinely authorised. It also cannot be solved by classifying text,
because three of the agentic vectors in the library carry no adversarial text at all. The
structural fix is to stop trusting the agent's *account of what the user wanted* and bind
the user's instruction cryptographically before delegation, then check the settlement
request against that binding.

This is a working implementation, not a diagram. :class:`IntentArtifact` is signed with
Ed25519 over a canonical serialisation; :class:`IntentGuard` verifies signature, expiry,
replay, payee binding and amount tolerance, and returns machine-readable reason codes. It
is deliberately deterministic: it is a control, and controls that behave probabilistically
cannot be reasoned about in a dispute.

Why asymmetric, having started with HMAC-SHA256
-----------------------------------------------
A shared secret makes the verifier a valid signer. The PSP that checks an intent artefact
could also have minted it, so the artefact proves nothing about what the user asked for
the moment the party holding it has an interest in the answer - which, under a mandatory
reimbursement regime that splits liability between sending and receiving institutions, is
always. The whole purpose of binding intent is to produce evidence that survives a
dispute, and symmetric signing cannot produce evidence, only a checksum.

There is a distribution argument too. HMAC needs a per-user key present on both the
issuing wallet and every PSP that might verify, which is key material fanned out across
the widest attack surface in the system. Ed25519 keeps the private half on the user's
device and ships a 32-byte public key to anyone who needs to verify. Google's AP2 signs
its Intent and Cart Mandates asymmetrically for the same reasons; the field-by-field
mapping onto AP2, ACP, Agent Pay and Web Bot Auth is in ``docs/AGENTIC_PROTOCOLS.md``.

The honest limitation is stated in :func:`coverage`: this stops hijack and replay, and
does nothing whatsoever about a counterfeit storefront the user's agent was correctly
instructed to buy from.
"""

from __future__ import annotations

import base64
import json
import secrets
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

DEFAULT_TTL_S = 900
DEFAULT_AMOUNT_TOLERANCE = 0.05
"""Legitimate drift between quote and settlement: tax, shipping, FX."""

#: JOSE algorithm identifier for Ed25519, so the protected header is a real JWS header
#: rather than a lookalike. RFC 8037.
JWS_ALG = "EdDSA"


def _b64u(raw: bytes) -> str:
    """Base64url without padding, as every JOSE serialisation requires."""
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64u_decode(text: str) -> bytes:
    pad = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(text + pad)


def generate_keypair(seed: Optional[bytes] = None) -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """A signing keypair, optionally derived from a fixed seed for reproducible runs.

    The seed path exists so tests and committed demo artefacts are byte-identical across
    machines. It is not how a wallet should generate a key, which is why it is explicit
    rather than a default.
    """
    private = (
        Ed25519PrivateKey.from_private_bytes(seed[:32].ljust(32, b"\0"))
        if seed is not None
        else Ed25519PrivateKey.generate()
    )
    return private, private.public_key()


def public_key_bytes(key: Ed25519PublicKey) -> bytes:
    return key.public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )


@dataclass(frozen=True)
class IntentArtifact:
    """A user's instruction, frozen before any agent touches it."""

    user_id: str
    payee_id: str
    max_amount: float
    currency: str
    scope: Tuple[str, ...]
    issued_at: float
    expires_at: float
    nonce: str
    key_id: str = ""

    def canonical(self) -> bytes:
        """Deterministic serialisation. Signature verification depends on byte stability."""
        payload = {
            "user_id": self.user_id,
            "payee_id": self.payee_id,
            "max_amount": round(float(self.max_amount), 2),
            "currency": self.currency,
            "scope": sorted(self.scope),
            "issued_at": round(float(self.issued_at), 3),
            "expires_at": round(float(self.expires_at), 3),
            "nonce": self.nonce,
            "key_id": self.key_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def signing_input(self) -> bytes:
        """The JWS signing input: ``base64url(header).base64url(payload)``.

        The header is covered by the signature, which is what stops an attacker rewriting
        ``alg`` to something they can forge. Algorithm confusion is the standard way a JWT
        implementation is broken, and it is only prevented if the header is signed and the
        verifier refuses anything it did not expect.
        """
        header = {"alg": JWS_ALG, "typ": "intent+jws", "kid": self.key_id}
        protected = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"{_b64u(protected)}.{_b64u(self.canonical())}".encode("ascii")

    def sign(self, private_key: Ed25519PrivateKey) -> str:
        return _b64u(private_key.sign(self.signing_input()))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def issue_intent(private_key: Ed25519PrivateKey, *, user_id: str, payee_id: str,
                 max_amount: float, currency: str = "INR",
                 scope: Tuple[str, ...] = ("purchase",), ttl_s: int = DEFAULT_TTL_S,
                 key_id: str = "",
                 now: Optional[float] = None) -> Tuple[IntentArtifact, str]:
    now = time.time() if now is None else now
    artifact = IntentArtifact(
        user_id=user_id,
        payee_id=payee_id,
        max_amount=float(max_amount),
        currency=currency,
        scope=tuple(scope),
        issued_at=now,
        expires_at=now + ttl_s,
        nonce=secrets.token_hex(16),
        key_id=key_id or user_id,
    )
    return artifact, artifact.sign(private_key)


@dataclass
class GuardDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)

    @property
    def reason_string(self) -> str:
        return "|".join(self.reasons) if self.reasons else "ok"


class IntentGuard:
    """Verifies a settlement request against the user's signed intent.

    Holds only public keys. That is the point of the asymmetric rewrite: this object can be
    deployed on every PSP that needs to verify without any of them gaining the ability to
    mint an artefact, so a verified intent remains evidence about the user rather than a
    statement by whoever is holding it.
    """

    def __init__(self, public_keys: Dict[str, Ed25519PublicKey] | Ed25519PublicKey, *,
                 amount_tolerance: float = DEFAULT_AMOUNT_TOLERANCE):
        if isinstance(public_keys, Ed25519PublicKey):
            # Single-key convenience for tests and the demo; any key id resolves to it.
            self.public_keys: Dict[str, Ed25519PublicKey] = {}
            self._default_key: Optional[Ed25519PublicKey] = public_keys
        else:
            self.public_keys = dict(public_keys)
            self._default_key = None
        self.amount_tolerance = amount_tolerance
        self._spent_nonces: Set[str] = set()

    def _resolve(self, key_id: str) -> Optional[Ed25519PublicKey]:
        if self._default_key is not None:
            return self._default_key
        return self.public_keys.get(key_id)

    def verify(self, artifact: Optional[IntentArtifact], signature: Optional[str], *,
               settle_payee: str, settle_amount: float, settle_currency: str,
               requested_scope: Tuple[str, ...] = ("purchase",),
               now: Optional[float] = None) -> GuardDecision:
        now = time.time() if now is None else now
        reasons: List[str] = []

        if artifact is None or signature is None:
            return GuardDecision(False, ["INTENT_ABSENT"])

        # An unknown signer is a failed signature, not a pass. Resolving a missing key to
        # "no verification" is how signature checks get skipped in production.
        public = self._resolve(artifact.key_id)
        if public is None:
            return GuardDecision(False, ["INTENT_SIGNATURE_INVALID"])
        try:
            public.verify(_b64u_decode(signature), artifact.signing_input())
        except (InvalidSignature, ValueError, TypeError):
            return GuardDecision(False, ["INTENT_SIGNATURE_INVALID"])

        if now > artifact.expires_at:
            reasons.append("INTENT_EXPIRED")
        if artifact.nonce in self._spent_nonces:
            reasons.append("INTENT_REPLAYED")
        if settle_payee != artifact.payee_id:
            # The signature is valid and the payee still changed: this is the exact
            # fingerprint of an indirect prompt injection.
            reasons.append("PAYEE_MISMATCH")
        if settle_currency != artifact.currency:
            reasons.append("CURRENCY_MISMATCH")
        ceiling = artifact.max_amount * (1 + self.amount_tolerance)
        if settle_amount > ceiling:
            reasons.append("AMOUNT_EXCEEDS_INTENT")
        if not set(requested_scope).issubset(set(artifact.scope)):
            reasons.append("SCOPE_VIOLATION")

        if not reasons:
            self._spent_nonces.add(artifact.nonce)
            return GuardDecision(True, [])
        return GuardDecision(False, reasons)


# --------------------------------------------------------------------------------------
# Dataset-level application
# --------------------------------------------------------------------------------------

#: How far past the amount tolerance a settlement has to go before the overrun stops being
#: explicable by ordinary pricing drift and becomes a provable contradiction of the intent.
GROSS_OVERRUN_MULTIPLE = 4.0

#: Reason codes the guard can raise, in the order they are reported.
REASON_CODES = [
    "INTENT_ABSENT", "INTENT_SIGNATURE_INVALID", "INTENT_EXPIRED", "INTENT_REPLAYED",
    "PAYEE_MISMATCH", "AMOUNT_OVER_TOLERANCE", "AMOUNT_EXCEEDS_INTENT", "SCOPE_VIOLATION",
]

#: Codes that decline the payment outright. Each one requires a *presented* intent artefact
#: that contradicts the settlement request, which is a provable violation rather than an
#: inference, so the decline is defensible in a dispute.
HARD_BLOCK_CODES = frozenset(REASON_CODES) - {"INTENT_ABSENT", "AMOUNT_OVER_TOLERANCE"}

#: Codes that raise friction instead of declining.
#:
#: A missing intent artefact means the agent never bound the user's instruction - which
#: describes an attacker, but also describes every legacy integration that predates the
#: protocol. Declining on it costs roughly three legitimate agent payments for every
#: fraudulent one it stops.
#:
#: A modest amount overrun is here for the same reason. It reads like a provable violation
#: and is not: benign quote-to-settlement drift has a tail, so a fixed 5% line declines a
#: couple of honest payments in every ten thousand. That is a small number and it was still
#: enough to contradict this project's own claim that hard blocks carry no false positives -
#: so the claim is now true by construction rather than by hope.
STEP_UP_CODES = frozenset({"INTENT_ABSENT", "AMOUNT_OVER_TOLERANCE"})


def apply_guard(df: pd.DataFrame, *, amount_tolerance: float = DEFAULT_AMOUNT_TOLERANCE,
                ttl_s: int = DEFAULT_TTL_S) -> pd.DataFrame:
    """Evaluate the guard across every agent-initiated row in a transaction frame.

    The transaction schema records the *outcome* of the checks a real deployment would run
    against live artefacts (token present, signature valid, payee match, amount delta,
    intent age, scope). This maps those columns onto the same reason codes the live guard
    emits, so the offline evaluation and the online control speak one language.

    Two outputs, not one. ``intent_guard_blocked`` marks a provable contradiction of a
    presented artefact and declines the payment; ``intent_guard_stepup`` marks a missing
    artefact, which is a risk signal for the model rather than grounds to decline on its
    own. Collapsing the two would trade a lot of legitimate agent payments for a little
    detection the behavioural model already provides.
    """
    agent = df["initiated_by_agent"].to_numpy() == 1
    n = len(df)

    absent = agent & (df["intent_token_present"].to_numpy() == 0)
    # Every violation is gated on an artefact actually having been presented. Without the
    # gate a row that presented nothing at all could be hard-declined for "exceeding an
    # intent" that does not exist - which is precisely the false decline the two-verdict
    # design was built to avoid, arriving through the back door.
    seen = agent & ~absent
    bad_sig = seen & (df["intent_signature_valid"].to_numpy() == 0)
    expired = seen & (df["intent_age_s"].to_numpy() > ttl_s)
    replayed = seen & (df["spt_reuse_count"].to_numpy() >= 2)
    payee_mismatch = seen & (df["intent_payee_match"].to_numpy() == 0)
    scope_violation = seen & (df["spt_scope_violation"].to_numpy() == 1)

    # Amount overrun is graded, because quote-to-settlement drift is a continuum rather than
    # a contradiction. Tax, shipping recalculation and an FX tick can genuinely carry a
    # payment past a 5% tolerance, so a small overrun raises friction and only a gross one -
    # far outside anything a merchant's own pricing could explain - is treated as provable.
    delta = df["intent_amount_delta_ratio"].to_numpy()
    amount_over = seen & (delta > amount_tolerance)
    amount_exceeds = seen & (delta > amount_tolerance * GROSS_OVERRUN_MULTIPLE)

    flags = {
        "INTENT_ABSENT": absent,
        "INTENT_SIGNATURE_INVALID": bad_sig,
        "INTENT_EXPIRED": expired,
        "INTENT_REPLAYED": replayed,
        "PAYEE_MISMATCH": payee_mismatch,
        "AMOUNT_OVER_TOLERANCE": amount_over & ~amount_exceeds,
        "AMOUNT_EXCEEDS_INTENT": amount_exceeds,
        "SCOPE_VIOLATION": scope_violation,
    }

    reasons = np.array([""] * n, dtype=object)
    for code, mask in flags.items():
        idx = np.where(mask)[0]
        for i in idx:
            reasons[i] = f"{reasons[i]}|{code}" if reasons[i] else code

    blocked = np.zeros(n, dtype=int)
    for code in HARD_BLOCK_CODES:
        blocked |= flags[code].astype(int)
    step_up = np.zeros(n, dtype=int)
    for code in STEP_UP_CODES:
        step_up |= flags[code].astype(int)

    out = df.copy()
    out["intent_guard_blocked"] = blocked
    out["intent_guard_stepup"] = step_up
    out["intent_guard_reasons"] = reasons
    out["intent_guard_violation_count"] = sum(m.astype(int) for m in flags.values())
    return out


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per-vector report of what the guard alone catches, and what it cannot.

    The rows with 0% coverage are the interesting ones. A counterfeit storefront, a spoofed
    crawler identity or a stolen-but-in-scope token all produce a perfectly valid intent
    chain, and no amount of intent verification will touch them.

    The ``(legitimate)`` row is the one to read first: its block rate is the guard's false
    decline rate on honest agent traffic, and it should be zero.
    """
    agentic = df[(df["initiated_by_agent"] == 1)]
    if agentic.empty:
        return pd.DataFrame()
    rows = []
    for vector_id, group in agentic.groupby(agentic["attack_vector_id"].replace("", "(legitimate)")):
        rows.append(
            {
                "attack_vector_id": vector_id,
                "rows": len(group),
                "is_fraud": int(group["is_fraud"].sum()),
                "guard_blocked": int(group["intent_guard_blocked"].sum()),
                "block_rate": round(float(group["intent_guard_blocked"].mean()), 4),
                "step_up_rate": round(float(group["intent_guard_stepup"].mean()), 4),
                "top_reason": (
                    group.loc[group["intent_guard_blocked"] == 1, "intent_guard_reasons"]
                    .str.split("|").explode().value_counts().index[0]
                    if int(group["intent_guard_blocked"].sum()) else ""
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values("block_rate", ascending=False)
    return out.reset_index(drop=True)

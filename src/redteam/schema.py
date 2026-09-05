"""Canonical transaction schema shared by the generator and the defence.

The schema is the contract between the red team and the blue team. Attack generators
may only write fields declared here, and the defence may only read fields marked as
observable. That separation is what stops the pipeline from silently leaking labels.

Field groups mirror what a real payment risk engine actually receives at authorisation
time: the payment instruction itself, the authentication result, device and session
telemetry, passive behavioural biometrics, counterparty (payee) intelligence and — new
for agentic commerce — the delegated-authority envelope.
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------------------

RAILS: List[str] = [
    "UPI_P2M",      # India, merchant push, instant + irrevocable
    "UPI_P2P",      # India, peer-to-peer push
    "IMPS",         # India, 24x7 immediate transfer
    "NEFT",         # India, batch-ish deferred settlement
    "RTP_FEDNOW",   # US instant rails
    "SEPA_INST",    # EU instant credit transfer
    "CARD_CNP",     # card, card-not-present
    "CARD_CP",      # card, card-present
    "WALLET",       # stored-value / PPI
]

# Rails that settle irrevocably in seconds: no clawback window, so the model has to be
# right at authorisation time rather than in an overnight batch.
IRREVOCABLE_RAILS = {"UPI_P2M", "UPI_P2P", "IMPS", "RTP_FEDNOW", "SEPA_INST"}

CHANNELS = ["mobile_app", "web", "pos", "ivr", "branch_assisted", "agent_api", "recurring"]

AUTH_METHODS = ["upi_pin", "biometric", "otp_sms", "passkey", "3ds2", "cvv_only", "none", "voice_print"]

INITIATORS = ["human", "ai_agent", "scheduled"]

FRAUD_TYPES = ["none", "app_scam", "account_takeover", "synthetic_identity", "card_fraud",
               "mule_layering", "agentic_compromise", "infrastructure_abuse", "first_party"]


# --------------------------------------------------------------------------------------
# Column groups
# --------------------------------------------------------------------------------------

#: Identity / bookkeeping columns. Never fed to the model.
META_COLUMNS: List[str] = [
    "txn_id",
    "timestamp",
    "customer_id",
    "payer_account_id",
    "payee_account_id",
    "merchant_id",
    "device_id",
]

#: The payment instruction as it appears on the wire.
INSTRUCTION_COLUMNS: List[str] = [
    "rail",
    "channel",
    "amount",
    "currency",
    # The settled amount restated in the base currency. A multi-currency portfolio needs both:
    # ``amount`` is what the customer sees and what appears on the rail, while every velocity
    # limit, reporting threshold and cross-rail comparison has to be computed on one scale.
    # Carrying only the former is how a EUR-settled payment ends up being compared against an
    # INR limit; carrying only the latter is how a CSV ends up claiming a $9,800 median.
    "amount_inr",
    "is_cross_border",
    "mcc",
    "payee_psp",
    "payer_psp",
    "is_irrevocable_rail",
    "iso20022_remittance_len",
    "iso20022_field_entropy",
    "iso20022_unstructured_ratio",
]

#: Authentication outcome and friction telemetry.
AUTH_COLUMNS: List[str] = [
    "auth_method",
    "auth_attempts",
    "auth_latency_ms",
    "step_up_triggered",
    "mfa_passed",
    "otp_delivery_delay_s",
    "recent_credential_change_h",
    "sim_swap_recency_days",
]

#: Device and session telemetry.
DEVICE_COLUMNS: List[str] = [
    "device_age_days",
    "device_is_new",
    "device_is_emulator",
    "device_is_rooted",
    "screen_share_active",
    "remote_access_app_detected",
    "accessibility_service_active",
    "call_in_progress",
    "vpn_or_proxy",
    "ip_asn_is_hosting",
    "ip_country_mismatch",
    "geo_velocity_kmh",
    "distance_from_home_km",
    "session_duration_s",
    "app_backgrounded_count",
]

#: Passive behavioural biometrics captured during the payment journey.
BEHAVIOUR_COLUMNS: List[str] = [
    "keystroke_flight_mean_ms",
    "keystroke_flight_cv",
    "typing_burstiness",
    "pointer_entropy",
    "form_fill_duration_s",
    "payee_field_pasted",
    "hesitation_events",
    "amount_field_corrections",
    "biometric_match_score",
    "voice_match_score",
    "liveness_score",
    "doc_ocr_confidence",
]

#: Counterparty intelligence — the single most predictive block for push-payment fraud.
PAYEE_COLUMNS: List[str] = [
    "payee_age_days",
    "payee_added_minutes_ago",
    "payee_is_first_time",
    "payee_account_age_days",
    "payee_inbound_unique_payers_24h",
    "payee_inbound_amount_24h",
    "payee_outbound_ratio_24h",
    "payee_name_match_score",
    "payee_name_homoglyph_flag",
    "payee_prior_txn_count",
    "payee_is_high_risk_category",
]

#: Delegated-authority envelope for agentic commerce (ACP / MCP / Agent Pay style).
AGENTIC_COLUMNS: List[str] = [
    "initiated_by_agent",
    "agent_is_registered",
    # Web Bot Auth / RFC 9421 message signatures, which are two facts and not one. An agent
    # *asserting* a registered identity (a Signature-Agent header naming a key directory) is
    # a claim anybody can make by copying a user-agent string; the request signature
    # verifying against that directory's key is the only part an impersonator cannot
    # reproduce. Collapsing them into `agent_is_registered` made a spoofed crawler
    # indistinguishable from the legacy integration that never enrolled.
    "agent_identity_asserted",
    "agent_request_signature_valid",
    "agent_reputation",
    "intent_token_present",
    "intent_signature_valid",
    "intent_amount_delta_ratio",
    "intent_payee_match",
    "intent_age_s",
    "agent_tool_calls",
    "agent_untrusted_content_tokens",
    "agent_injection_score",
    "spt_scope_violation",
    "spt_reuse_count",
    "merchant_is_agent_optimised",
    "merchant_domain_age_days",
]

#: Customer / relationship context available from the core banking system.
CUSTOMER_COLUMNS: List[str] = [
    "customer_tenure_days",
    "customer_age_band_ord",
    "customer_digital_literacy",
    "customer_is_corporate",
    "account_balance_ratio",
    "kyc_level",
    "prior_fraud_reports",
]

#: Temporal context.
TIME_COLUMNS: List[str] = ["hour", "day_of_week", "is_night", "is_salary_window"]

#: Labels and red-team provenance. Strictly forbidden as model inputs.
LABEL_COLUMNS: List[str] = [
    "is_fraud",
    "fraud_type",
    "attack_vector_id",
    "campaign_id",
    "mule_ring_id",
    "is_hard_negative",
    "evasion_applied",
]


def observable_columns() -> List[str]:
    """Columns the defence is allowed to see (before derived feature engineering)."""
    return (
        INSTRUCTION_COLUMNS
        + AUTH_COLUMNS
        + DEVICE_COLUMNS
        + BEHAVIOUR_COLUMNS
        + PAYEE_COLUMNS
        + AGENTIC_COLUMNS
        + CUSTOMER_COLUMNS
        + TIME_COLUMNS
    )


def all_columns() -> List[str]:
    return META_COLUMNS + observable_columns() + LABEL_COLUMNS


CATEGORICAL_COLUMNS: List[str] = ["rail", "channel", "auth_method", "payee_psp", "payer_psp", "currency"]

#: Sensible defaults so an attack generator only has to set the fields it actually
#: manipulates; everything else falls back to a benign-looking value.
DEFAULTS: Dict[str, object] = {
    "currency": "INR",
    "amount_inr": 0.0,
    "is_cross_border": 0,
    "mcc": 5999,
    "iso20022_remittance_len": 0,
    "iso20022_field_entropy": 0.0,
    "iso20022_unstructured_ratio": 0.0,
    "auth_attempts": 1,
    "step_up_triggered": 0,
    "mfa_passed": 1,
    "otp_delivery_delay_s": 0.0,
    "recent_credential_change_h": 9999.0,
    "sim_swap_recency_days": 9999.0,
    "screen_share_active": 0,
    "remote_access_app_detected": 0,
    "accessibility_service_active": 0,
    "call_in_progress": 0,
    "vpn_or_proxy": 0,
    "ip_asn_is_hosting": 0,
    "ip_country_mismatch": 0,
    "device_is_emulator": 0,
    "device_is_rooted": 0,
    "payee_field_pasted": 0,
    "biometric_match_score": 0.0,
    "voice_match_score": 0.0,
    "liveness_score": 0.0,
    "doc_ocr_confidence": 0.0,
    "payee_name_homoglyph_flag": 0,
    "payee_is_high_risk_category": 0,
    "initiated_by_agent": 0,
    "agent_is_registered": 0,
    "agent_identity_asserted": 0,
    "agent_request_signature_valid": 0,
    "agent_reputation": 0.0,
    "intent_token_present": 0,
    "intent_signature_valid": 0,
    "intent_amount_delta_ratio": 0.0,
    "intent_payee_match": 0,
    "intent_age_s": 0.0,
    "agent_tool_calls": 0,
    "agent_untrusted_content_tokens": 0,
    "agent_injection_score": 0.0,
    "spt_scope_violation": 0,
    "spt_reuse_count": 0,
    "merchant_is_agent_optimised": 0,
    "merchant_domain_age_days": 3650.0,
    "prior_fraud_reports": 0,
    "is_fraud": 0,
    "fraud_type": "none",
    "attack_vector_id": "",
    "campaign_id": "",
    "mule_ring_id": "",
    "is_hard_negative": 0,
    "evasion_applied": 0,
}

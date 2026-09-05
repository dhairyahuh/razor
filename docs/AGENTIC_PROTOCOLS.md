# Agentic payment protocols, and where this repository sits against them

The agentic pillar of this project is not a speculative design. Every field in the
delegated-authority section of the schema corresponds to something a named 2026 protocol
either carries on the wire or requires a verifier to check, and this document is the field
by field mapping. It exists because "we invented a plausible-looking envelope for agent
payments" and "we implemented the envelope the industry is standardising on" are very
different claims, and only the mapping distinguishes them.

## The protocols this schema is shaped for

| Protocol | Sponsor | What it standardises | Relevance here |
| --- | --- | --- | --- |
| **AP2** (Agent Payments Protocol) | Google, with ~60 payment partners | Intent Mandate and Cart Mandate as verifiable digital credentials, signed asymmetrically | The direct model for `IntentArtifact` |
| **ACP** (Agentic Commerce Protocol) | OpenAI and Stripe | Product discovery and checkout for agents, Shared Payment Tokens | The model for the `spt_*` columns |
| **Agent Pay** | Mastercard | Registered agents, Intent API binding a user's instruction to the authorisation request | The model for the intent-to-settlement comparison |
| **Intelligent Commerce** | Visa | Agent-issued payment credentials scoped to a transaction | The model for per-transaction scoped tokens |
| **Web Bot Auth** | IETF draft, deployed by Cloudflare and others | HTTP message signatures (RFC 9421) proving a bot is who it claims to be | Implemented in `agent_controls.py` |
| **MCP** (Model Context Protocol) | Anthropic, now the Agentic AI Foundation | Tool discovery and invocation by LLMs | The threat surface for tool poisoning and confused deputy |

## AP2 mandate vocabulary, mapped to this schema

AP2's central idea is that a user's instruction becomes a **verifiable credential** before
any agent acts on it, and the payment request is checked against that credential rather
than against the agent's account of it. That is exactly the structure of
`src/redteam/defend/intent_guard.py`.

AP2 distinguishes two mandates. The **Intent Mandate** captures what the user authorised
before shopping begins — useful for delegated or unattended purchases, where the user is
not present at checkout. The **Cart Mandate** captures the specific cart the user approved,
signed at the moment of purchase. This repository implements a single artefact carrying
both roles, which is the honest simplification to state: the amount ceiling and scope play
the Intent Mandate's part, and the payee binding plays the Cart Mandate's.

| AP2 concept | This repository | Notes |
| --- | --- | --- |
| Intent Mandate | `IntentArtifact` with `scope` and `max_amount` | Authority granted before the agent shops |
| Cart Mandate | `IntentArtifact.payee_id` checked against `settle_payee` | The `PAYEE_MISMATCH` reason code |
| Mandate signature (asymmetric) | Ed25519 over a JWS signing input, `alg: EdDSA` | See "Why asymmetric" below |
| Verifiable Credential envelope | `IntentArtifact.canonical()` plus a signed protected header | Canonical JSON, not JSON-LD: a deliberate simplification |
| Mandate expiry | `expires_at`, `INTENT_EXPIRED` | Default TTL 900s |
| Replay protection | `nonce` and the guard's spent-nonce set, `INTENT_REPLAYED` | |
| Human-present vs human-absent | `intent_token_present`, `initiated_by_agent` | |
| Agent identity | `agent_identity_asserted`, `agent_request_signature_valid` | Web Bot Auth, below |

### Why asymmetric, having started with HMAC

The first implementation here signed intent artefacts with HMAC-SHA256, and that was wrong
for a reason worth stating plainly: **a shared secret makes the verifier a valid signer.**
A PSP that can check an artefact could also have minted it, so the artefact proves nothing
about what the user asked for the moment the party holding it has an interest in the
answer. Under the UK PSR's mandatory reimbursement regime, which splits APP fraud liability
50/50 between sending and receiving institutions, both parties always have an interest. An
intent artefact exists to be evidence in exactly that dispute, and symmetric signing cannot
produce evidence — only a checksum.

There is a key-distribution argument too. HMAC needs per-user key material present on the
issuing wallet *and* on every PSP that might verify, which is the widest attack surface in
the system. Ed25519 keeps the private half on the user's device and ships a 32-byte public
key to anyone who needs to verify. AP2 signs its mandates asymmetrically for both reasons.

The protected header is inside the signing input, so `alg` cannot be rewritten after the
fact. Algorithm confusion is the standard way a JWT deployment is broken, and it is only
prevented if the header is covered by the signature and the verifier refuses anything it
did not expect. `test_the_signed_header_pins_the_algorithm` asserts this.

## ACP Shared Payment Tokens, mapped

| ACP concept | This repository | Notes |
| --- | --- | --- |
| Shared Payment Token | `ScopedToken` | Minted per purchase, not handed over as a standing credential |
| Token scope | `scope`, `spt_scope_violation`, `TOKEN_SCOPE_BLOCKED` | |
| Token single use | `single_use`, `spt_reuse_count`, `TOKEN_ALREADY_SPENT` | |
| Delegated spend limit | `max_amount`, `TOKEN_CEILING_EXCEEDED` | Reported as a loss bound, not a detection rate |
| Agent-facing product feed | `merchant_is_agent_optimised` | The attack surface for counterfeit storefronts |

ACP does not currently specify proof-of-possession binding for Shared Payment Tokens, which
leaves a stolen token a bearer credential. This repository implements the DPoP-style
binding anyway (`holder_thumbprint`, in the manner of DPoP's `jkt` claim) because
`AGENTIC-SPT-REPLAY` is in the attack library and nothing else in the stack stops it. That
is a place where the implementation here is deliberately ahead of the protocol rather than
behind it, and `test_a_bound_token_is_worthless_to_a_thief` is the demonstration.

## Web Bot Auth, mapped

Web Bot Auth applies RFC 9421 HTTP Message Signatures to the problem of an agent proving
its identity. The distinction it draws is the one this repository got wrong at first and had
to fix: **asserting** an identity and **proving** it are two different observations.

DataDome recorded over 16 million spoofed requests against `Meta-ExternalAgent` in a
two-month window and a 2.4% impersonation rate for `PerplexityBot`. Copying a user-agent
string is free. Producing an RFC 9421 signature over the request is not, because it requires
the private key the identity was enrolled with.

| Web Bot Auth concept | This repository |
| --- | --- |
| Signature agent / key directory | `AgentRegistry` |
| Covered components | `COVERED_COMPONENTS`, fixed rather than client-nominated |
| `created` and freshness window | `AGENT_SIGNATURE_STALE`, 60s default |
| Signature verification failure | `AGENT_SIGNATURE_INVALID` — the impersonation fingerprint |
| Unenrolled bot | `AGENT_NOT_ENROLLED` — reported separately, and deliberately so |

The covered-component list is fixed rather than negotiated per request. A verifier that
accepts whatever set the client nominates can be handed a signature over almost nothing and
will verify it happily.

The last two rows are the important pair. Conflating them made the control unusable: 6% of
legitimate agent traffic in the generator never enrolled, so treating "unenrolled" as
"impostor" declines one honest agent payment in seventeen. And verification failure is not
proof of impersonation either — key rotation outpacing a verifier's cache, a proxy stripping
the Signature header, and clock skew past the `created` window all produce genuine failures
on honest requests. So an unproven assertion costs the trusted lane and raises friction,
which is what Web Bot Auth is for. The hard decline stays with the intent guard, where the
contradiction is provable.

## What these protocols do not solve, and this repository measures

The coverage tables in every run report carry rows at or near zero, and they are the most
useful thing in the agentic section. `AGENTIC-COUNTERFEIT-STOREFRONT` is untouched by all
three controls, because the user's agent was correctly instructed to buy from a merchant
that happens to be fraudulent: the intent is honest, the identity is genuine, the token is
properly held and spent in scope. Every credential in the chain verifies.

No mandate vocabulary fixes that. It is a merchant-provenance and behavioural-detection
problem, which is why the tabular model exists alongside the guards rather than being
replaced by them — and why the ablation grid reports the guard layer's contribution
separately rather than folding it into a single number.

## References

- AP2 — <https://ap2-protocol.org>
- ACP — <https://developers.openai.com/commerce/> and Stripe's ACP documentation
- Mastercard Agent Pay — Mastercard newsroom, agentic commerce announcements
- Visa Intelligent Commerce — Visa developer documentation
- Web Bot Auth — IETF drafts `draft-meunier-web-bot-auth-architecture` and RFC 9421
- MCP — <https://modelcontextprotocol.io>

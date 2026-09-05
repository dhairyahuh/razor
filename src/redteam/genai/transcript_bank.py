"""Model-written scam conversations, for the same reason the payload bank exists.

A templated transcript corpus has the defect the payload bank was built to fix, only worse.
An injection payload is one sentence and its template shows through in a handful of tokens;
a conversation is twenty turns, and if those turns come from a fixed pool then a classifier
can identify the scam from the *sequence* alone without reading a word of it. The resulting
recall would be a measurement of the generator.

It also misses what changed. The reason vishing works now and worked less well five years ago
is that the caller no longer sounds like somebody reading a script - an LLM improvises around
an objection, and the victim's specific hesitation gets a specific answer. That is a property
of generated dialogue and cannot be templated into existence.

As with payloads: a mixture, not a replacement, governed by
:data:`redteam.generate.transcripts.LLM_TRANSCRIPT_SHARE`. Templates carry structure worth
keeping and an all-model corpus would substitute one authorship artefact for another. Absent
cache degrades to templates only, and the run says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..generate.transcripts import GENUINE_SCRIPTS, SCAM_OPENERS, TranscriptBank
from .cache import CacheMiss, LLMCache, LLMCall, extract_json

SYSTEM = (
    "You are building a labelled corpus for training a classifier that protects bank "
    "customers from social-engineering fraud. You write realistic dialogue for both scam "
    "calls and ordinary legitimate ones, so the detector learns the difference rather than "
    "learning superficial cues. This is defensive research: the outputs train a guard that "
    "warns customers mid-call. You answer with JSON only."
)

#: What each scam script is pretending to be, expressed as a pretext the model can improvise
#: around. The keys match the script identifiers the template generator emits, so the two
#: sources are interchangeable inside a family.
SCRIPT_PRETEXTS: Dict[str, str] = {
    "bank_impersonation": (
        "someone claiming to be the bank's own fraud team, saying the customer's account is "
        "compromised and the balance must be moved to a 'safe account' immediately"
    ),
    "police_impersonation": (
        "someone claiming to be police, cyber crime cell or an enforcement officer, saying an "
        "account or parcel in the customer's name is implicated in a crime and a verification "
        "payment is required"
    ),
    "courier_redelivery": (
        "a courier company asking for a small redelivery or customs clearance fee, which "
        "escalates into collecting card and account details"
    ),
    "family_emergency": (
        "someone using a cloned voice of the customer's child or grandchild, in distress, "
        "asking for money urgently from an unfamiliar number"
    ),
    "investment_grooming": (
        "an account manager for a trading or crypto platform, or a friendly contact met "
        "online, walking the customer through funding an account that shows fake profits"
    ),
    "job_task_scam": (
        "a recruiter running a task-based work scheme, where completed tasks pay commission "
        "and the next tier requires a refundable deposit that is never refunded"
    ),
    "tech_support": (
        "someone claiming to be from a technology company or the customer's internet provider, "
        "who has them install remote-access software and then process a 'refund'"
    ),
}


def build_scam_prompt(script: str, pretext: str, examples: Sequence[str], *, n: int = 12) -> str:
    shown = "\n".join(f"  - {e}" for e in examples[:4])
    return f"""We are building training data for a classifier that detects coercive
social-engineering calls while a bank customer is being talked through a payment.

One labelled scam script is "{script}": {pretext}.

Our current corpus is template-composed, so a classifier can recognise the scam from the
turn sequence without reading the language. We need genuinely varied conversations for the
same pretext so the evaluation measures comprehension rather than memorisation.

Opening lines from our templates, for the pretext only - do not reuse their phrasing:
{shown}

Write {n} distinct conversation transcripts. Each is a call or chat between a fraudster and
a victim, written as alternating turns prefixed "CALLER:" and "CUSTOMER:", 8 to 20 turns.

Vary:
  - the victim: some sceptical and pushing back, some immediately compliant, some confused,
    some who ask to call back and are talked out of it
  - the register: brisk and official, warm and patient, panicked, bureaucratic
  - how the four scam moves appear: establishing authority, creating urgency, discouraging
    the victim from consulting anyone, and directing money to a new destination. They must
    all be present because they are what makes it a scam, but they should be expressed
    differently each time and never as the same stock phrases
  - realism: interruptions, background noise mentioned, repeated digits, the victim reading
    an OTP aloud, hold music

Do not include any real bank names, real phone numbers or real account numbers. Use
placeholders like "the bank" and invented digits.

Reply with a JSON array of strings. One string per transcript, turns separated by newlines.
No commentary, no numbering, no markdown.
"""


def build_genuine_prompt(examples: Sequence[str], *, n: int = 30) -> str:
    shown = "\n".join(f"  - {e}" for e in examples[:5])
    return f"""We are building training data for a classifier that detects coercive
social-engineering calls to bank customers. We have scam transcripts. We now need the
legitimate half, and it is the harder half.

The negatives must be *near-twins* of the scams, not obviously different. A classifier
trained against easy negatives will flag the bank's own outbound fraud-prevention calls, and
the cost of that is customers who stop answering the telephone.

Situations from our templates, for the scenario only - do not reuse their phrasing:
{shown}

Write {n} distinct legitimate conversation transcripts, alternating "CALLER:" and
"CUSTOMER:" turns, 6 to 18 turns. These are real conversations that happen to involve money
moving, urgency, or an unfamiliar payee. Cover:
  - a genuine bank fraud team calling about a card block, who correctly refuse to ask for
    credentials and tell the customer to call the number on their card
  - a real builder or supplier discussing an invoice and a first payment
  - a family member genuinely asking for help with a bill, from their usual number
  - a legitimate courier arranging redelivery without asking for payment details
  - a customer calling their own bank to authorise a large transfer to a new payee
  - a genuine investment or pension adviser the customer already has a relationship with
  - a friend settling a shared cost, informally

Some should contain urgency and some should involve a first-time payee, because those are
present in genuine traffic and a classifier that treats either as decisive is useless. What
must be absent is the coercion: nobody discourages the customer from verifying, nobody
manufactures a consequence for stopping, nobody asks for a code.

Reply with a JSON array of strings. One transcript per string, turns separated by newlines.
No commentary, no numbering, no markdown.
"""


def generate_bank(cache: LLMCache, *, n_per_script: int = 12, n_genuine: int = 30,
                  temperature: float = 1.0,
                  scripts: Optional[Sequence[str]] = None) -> TranscriptBank:
    """Fetch or generate the transcript bank.

    Per-script cache misses are skipped rather than fatal, matching the payload bank: a
    partially populated cache should yield a smaller bank, not a failed run.
    """
    wanted = list(scripts) if scripts else sorted(SCRIPT_PRETEXTS)
    by_script: Dict[str, List[str]] = {}
    all_cached = True

    for script in wanted:
        pretext = SCRIPT_PRETEXTS.get(script)
        if not pretext:
            continue
        call = LLMCall(
            task="vishing_transcripts",
            prompt=build_scam_prompt(script, pretext, SCAM_OPENERS.get(script, []),
                                     n=n_per_script),
            system=SYSTEM,
            temperature=temperature,
            max_tokens=6000,
        )
        try:
            result = cache.complete(call)
        except CacheMiss:
            continue
        all_cached = all_cached and result.cached
        try:
            cleaned = _clean(extract_json(result.text))
        except ValueError:
            continue
        if cleaned:
            by_script[script] = cleaned

    genuine: List[str] = []
    call = LLMCall(
        task="genuine_transcripts",
        prompt=build_genuine_prompt(
            [turns[0] for scripts in GENUINE_SCRIPTS.values() for turns in scripts if turns],
            n=n_genuine,
        ),
        system=SYSTEM,
        temperature=temperature,
        max_tokens=7000,
    )
    try:
        result = cache.complete(call)
        all_cached = all_cached and result.cached
        genuine = _clean(extract_json(result.text))
    except (CacheMiss, ValueError):
        genuine = []

    return TranscriptBank(by_script=by_script, genuine=genuine, served_from_cache=all_cached)


def _clean(payload: object) -> List[str]:
    """Filter model output into usable transcripts.

    Stricter than the payload cleaner on one axis: a transcript with no turn markers is not
    a transcript, it is a summary, and letting one into the coercive class would teach the
    guard that third-person narration is a fraud signal.
    """
    if isinstance(payload, dict):
        payload = (payload.get("transcripts") or payload.get("conversations")
                   or payload.get("examples") or [])
    if not isinstance(payload, (list, tuple)):
        return []

    out: List[str] = []
    seen = set()
    for item in payload:
        if isinstance(item, (list, tuple)):
            item = "\n".join(str(t) for t in item)
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not 200 <= len(text) <= 12_000:
            continue
        if len(re.findall(r"(?im)^\s*(?:caller|customer|agent|victim)\s*:", text)) < 4:
            continue
        lowered = text.lower()
        if any(m in lowered for m in ("i cannot", "i can't", "i'm sorry", "as an ai",
                                      "cannot assist", "here are", "sure, here")):
            continue
        key = lowered[:200]
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out

"""The third data plane: what was said to the victim while the payment was made.

Why a transcript plane exists
-----------------------------
The tabular plane sees a payment. The agent-context plane sees what a shopping agent read.
Neither sees the thing that actually causes an authorised push payment: a person on a
telephone, talking a customer through their own banking app. Every feature the tabular model
has for this - ``call_in_progress``, ``screen_share_active``, the hesitation counters - is a
*proxy* for a conversation, and each one is a single bit an operator can choose not to leave
behind. The closed loop demonstrates exactly that: suppressing the coercion telemetry is one
of the cheapest levers in the search space.

The conversation itself is much harder to fake into innocence, because the scam has to say
what it needs to say. It has to establish authority, create urgency, isolate the victim from
anyone who would talk them out of it, and direct money to a new destination. Those four moves
are the scam; an operator who drops them has no scam left. Voice-channel telemetry is
increasingly available to banks - in-app call detection, IVR recordings, contact-centre
transcription - and it is where GenAI has changed the threat most sharply, because a cloned
voice and a fluent LLM remove the two things that used to make these calls detectable.

What makes this hard rather than a keyword list
-----------------------------------------------
The legitimate transcripts are the point of this module. A genuine bank fraud-team callback
also asserts authority, also conveys urgency, also asks about a payment, and also reads out
account digits. A relative helping a parent with a transfer also gives step-by-step
instructions over the phone. If benign traffic were small talk, a bag of words would score
perfectly and the number would mean nothing - which is precisely the criticism the injection
guard's in-distribution AUC already earns.

So each scam script has a legitimate near-twin that shares its surface: the impersonated
fraud-team call is paired with a real fraud-team call, the courier redelivery scam with a
real delivery-fee message, the family-emergency scam with a real family transfer. What
separates them is structural - who is being asked to move money where, and whether the caller
tries to prevent verification - and that is what the guard has to learn.

Generative layer
----------------
Templates give the structure and run with no API key. When the cached LLM layer has content,
a share of transcripts are model-written instead, for the same reason as the injection
payloads: a classifier trained purely on template output learns the templates. Both are mixed
so the corpus has real variation without becoming a study of one model's cadence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: Share of transcripts drawn from the model-written bank rather than composed from
#: templates, when a bank is available. Mirrors ``LLM_PAYLOAD_SHARE`` and for the same
#: reason: an all-template corpus teaches templates, an all-LLM corpus teaches one model's
#: prose style, and neither is what a deployed classifier meets.
LLM_TRANSCRIPT_SHARE = 0.45

#: Fraudulent conversation scripts, keyed by the family of vector they accompany. Each is a
#: sequence of turn generators; the structure is the attack and the wording is decoration.
SCAM_OPENERS: Dict[str, List[str]] = {
    "bank_impersonation": [
        "Good afternoon, this is {name} calling from the fraud prevention team at your bank. "
        "Am I speaking with the account holder?",
        "Hello, I'm calling from the security department. We've flagged unusual activity on "
        "your account ending {digits} and I need to verify some details urgently.",
    ],
    "police_impersonation": [
        "This is Inspector {name} from the cyber crime cell. Your account has been linked to "
        "a money laundering investigation.",
        "Sir, I'm calling from the enforcement directorate. There is a case registered "
        "against your PAN and your accounts are being frozen this evening.",
    ],
    "courier_redelivery": [
        "Hello, this is {name} from the courier service. Your parcel is held at customs and "
        "there's a small clearance fee outstanding.",
        "Ma'am, your delivery was returned. There's a redelivery charge of a few hundred "
        "rupees, otherwise the package goes back to the sender tonight.",
    ],
    "family_emergency": [
        "Mum, it's me. I'm using a friend's phone, mine broke. I'm in a bit of trouble and "
        "I need you to send some money quickly.",
        "Dad, please don't be angry. I've had an accident with the car and I need to settle "
        "it right now before they involve the police.",
    ],
    "investment_grooming": [
        "So glad you messaged back. I've been trading on this platform for two years now, my "
        "returns last quarter were about forty percent.",
        "You remember I mentioned my uncle's fund? He's opened a small window for friends "
        "this month. Minimum is quite low to start.",
    ],
    "job_task_scam": [
        "Congratulations on being shortlisted. Your first task set is ready - you complete "
        "the orders, and your commission is credited the same day.",
        "You've completed four tasks now. The next tier needs a refundable deposit to unlock, "
        "and then both the deposit and the commission come back together.",
    ],
    "tech_support": [
        "I'm calling from the technical support team, we've detected a virus transmitting "
        "from your router. I'll need to guide you through securing it.",
        "Your subscription auto-renewed at a much higher amount by mistake. I can process the "
        "refund but I'll need to walk you through the screens.",
    ],
}

#: The moves that make a conversation a scam. Every scam transcript contains authority,
#: urgency and a redirection; most contain isolation. That is the structure the classifier
#: has to find, and it is the structure an operator cannot drop and still succeed.
COERCION_TURNS: Dict[str, List[str]] = {
    "urgency": [
        "We have about eleven minutes before the transfer window closes, so please stay with me.",
        "If we don't complete this in the next few minutes the funds will be gone permanently.",
        "The case officer signs off at six and after that I can't help you until Monday.",
        "This has to be done now, tonight, or the freeze goes through automatically.",
    ],
    "isolation": [
        "Please don't discuss this with anyone at the branch, the investigation is confidential.",
        "Do not mention this call to family members, we have reason to believe someone close "
        "to you is involved.",
        "If the bank calls you back, don't answer - the fraudsters are spoofing their number.",
        "Keep this between us for now. I'll explain everything once the transfer clears.",
    ],
    "authority": [
        "You can verify my badge number afterwards, it's on the official website.",
        "I'm reading from your account record right now, so you know this is genuine.",
        "My employee reference is {digits}, note it down for your records.",
    ],
    "redirection": [
        "I need you to move the balance to the safe account I'm about to read out. It's held "
        "in your name, it's just a protected holding account.",
        "Add a new payee - the name will show as a holding account, that's normal, please "
        "approve it anyway.",
        "The account number is {digits}. Type it carefully and confirm even if the name "
        "doesn't match exactly, our system displays the department name.",
        "Use the immediate transfer option, not the scheduled one. Scheduled payments can be "
        "intercepted.",
    ],
    "coaching": [
        "Ignore the warning message that comes up, that's a generic notice we're aware of.",
        "If it asks you the reason for the payment, select 'paying family or friends'.",
        "Now install the support application so I can see your screen and check the details.",
        "Read me the code that just arrived on your phone so I can confirm the session.",
    ],
}

#: The customer's side of the call.
#:
#: Used by *both* classes, which is not incidental. When these appeared only in scam
#: transcripts the guard learned "sounds like somebody being walked through their banking
#: app" as a free label, and it transferred across held-out vocabulary as a register rather
#: than as content - a perfect score that was really a corpus artefact. A customer saying
#: "okay, I'm opening the app now" or "let me find my glasses" is ordinary on any call about
#: money, and the guard has to earn its separation somewhere else.
VICTIM_TURNS: List[str] = [
    "Sorry, I'm not sure I understand. Is this definitely from the bank?",
    "Okay. Okay, I'm opening the app now.",
    "It's showing me a warning about scams, should I be worried?",
    "How much did you say?",
    "I've never had to do anything like this before.",
    "Let me just get my glasses. One moment.",
    "Alright, it's asking me to confirm.",
    "My son usually helps me with these things.",
]

#: Legitimate near-twins. Each shares the surface of a scam script - authority, urgency,
#: account digits, a discussion about money - and differs in structure: nobody is being
#: directed to move funds to a new destination, and verification is encouraged rather than
#: discouraged. These are what stop the guard from being a keyword detector.
#:
#: Composed from parts, exactly as the scam side is, and that symmetry is not cosmetic. If
#: the positives were assembled from a combinatorial pool and the negatives were drawn from a
#: handful of fixed scripts, the classifier could separate them on repetition alone: the
#: negative class would contain a few dozen near-duplicate documents and TF-IDF would find
#: them instantly. The reported false-positive rate would then be a fact about the corpus
#: construction rather than about the guard, which is the failure mode the whole generative
#: layer exists to avoid.
GENUINE_SCRIPTS: Dict[str, List[List[str]]] = {
    "bank_impersonation": [
        [
            "Good afternoon, this is the fraud team at your bank. We've blocked a card "
            "payment we weren't sure about and I need to confirm whether it was you.",
            "I'm not going to ask you for any passwords or codes, and I won't ask you to "
            "move any money.",
            "Was there a transaction for around four thousand rupees this morning?",
            "Yes, that was me, I was paying for the flights.",
            "That's all I needed. I've released the block. If you'd prefer, hang up and call "
            "the number on the back of your card to confirm this call was genuine.",
        ],
    ],
    "police_impersonation": [
        [
            "This is the branch manager. There's been a notice on your account regarding a "
            "documentation update required by the regulator.",
            "There's no urgency and no payment involved. You can come into any branch in the "
            "next sixty days with your identification.",
            "Do I need to do anything online?",
            "Nothing at all. And please don't act on anyone who calls asking you to transfer "
            "funds because of this - that is not how it works.",
        ],
    ],
    "courier_redelivery": [
        [
            "Hello, this is the delivery service. Your parcel needs a signature and nobody "
            "was home. Can I confirm a redelivery slot for tomorrow?",
            "Is there anything to pay?",
            "No, nothing to pay, it's already prepaid. Just somebody at the address between "
            "ten and two.",
        ],
    ],
    "family_emergency": [
        [
            "Hi Mum, it's me. Are you free this evening? I wanted to sort out the money for "
            "the deposit.",
            "Of course. Is it the same account as last time?",
            "Same one, yes. And there's no rush at all, whenever suits you this week. Ring me "
            "back on my number if you want to double check it's me.",
        ],
    ],
    "investment_grooming": [
        [
            "I've been reading about the index funds you mentioned. The fees look reasonable.",
            "They're the boring ones, which is rather the point. Nothing to do quickly.",
            "I'll speak to the advisor at the branch before I move anything.",
            "Sensible. Take your time over it.",
        ],
    ],
    "job_task_scam": [
        [
            "Following up on your application - the interview panel is confirmed for Thursday "
            "at eleven, over video.",
            "Do I need to prepare anything?",
            "Just the portfolio you sent. There's no fee or deposit at any stage of our "
            "process, so please treat anyone who asks for one as fraudulent.",
        ],
    ],
    "tech_support": [
        [
            "You raised a ticket about the app not loading. I can see the error on our side, "
            "it's affecting a number of customers this morning.",
            "So it's not just me.",
            "Not at all. We're deploying a fix within the hour. I won't need access to your "
            "device and I won't ask for any codes.",
        ],
    ],
}

#: Share of genuine conversations that open with the *scam* opener for their scenario.
#:
#: This is not a trick to depress the score. A real bank fraud team does say "we've flagged
#: unusual activity on your account ending 1234"; a real courier does say a parcel is held.
#: The opener is genuinely ambiguous, and that ambiguity is the entire reason these scams
#: work - if the first sentence gave it away nobody would fall for them. A corpus whose two
#: classes have disjoint openers lets a bag-of-n-grams model classify on the first line and
#: never read the conversation, which produces a perfect AUC that measures nothing.
GENUINE_SHARES_SCAM_OPENER = 0.4

#: Share of scam conversations that make only the minimum moves - authority and a redirection,
#: with no isolation, no coaching and softened urgency.
#:
#: The tabular generator already insists that a large share of fraud emits no loud tell, and
#: the transcript plane has to obey the same rule for the same reason. Not every operator is
#: a shouting caricature; a competent one on a well-groomed victim sounds like an
#: administrator. Without these, every positive carries an identical five-part skeleton and
#: the classifier learns the skeleton.
QUIET_SCAM_SHARE = 0.3

#: Turns that legitimately follow the opener, shared across scenarios. Several are the exact
#: surface features a naive keyword rule would treat as fraud - a deadline, a first-time
#: payee, an unfamiliar account, a large sum - occurring in conversations that are entirely
#: ordinary. That overlap is the point.
GENUINE_MIDDLES: List[str] = [
    "It does need doing before Friday, otherwise the quote expires and we'd have to redo it.",
    "This would be the first time I've paid them, so I want to be careful about the details.",
    "The amount is quite a bit larger than I usually send, I hope that isn't a problem.",
    "I'll read the account number back to you so we're both sure. {digits}, is that right?",
    "My reference on the invoice is {digits} if you need to look it up.",
    "Can I check who I'm speaking to? I'd rather be cautious about this sort of thing.",
    "Of course, that's completely reasonable. My name is {name} and my extension is {digits}.",
    "Take as long as you need. Nothing has to be decided on this call.",
    "I'd rather sleep on it and ring you back tomorrow if that's alright.",
    "That's absolutely fine. There's no deadline on our side.",
    "The bank might send you a verification message - that's normal, it's their check, not ours.",
    "If you'd rather do it in branch, that works just as well.",
    "I've already got the payee saved from last year, so it should come up automatically.",
    "Sorry, the line broke up. Could you say the last part again?",
    "No rush at all. Have a think and let me know either way.",
]

#: Genuine turns that direct money to a destination the customer has not paid before.
#:
#: A new payee is not a fraud signal. Builders change banks, children move flat, a supplier
#: is used for the first time, and every one of those conversations contains an account
#: number read down a telephone. What separates them from the scam's redirection move is
#: what surrounds it: nobody is told to override a name-check warning, to choose a payment
#: reason that suppresses a prompt, or to keep it to themselves. Without these turns the
#: guard would learn that "account number" means fraud, and it would be right on this corpus
#: and useless on a real one.
GENUINE_NEW_PAYEE: List[str] = [
    "It'll be a new account for you - we changed banks in the spring. Shall I read it out?",
    "The account is {digits}. Please do check the name matches before you send anything.",
    "You'll get a warning that it's a first payment to this payee, which is normal. Read it "
    "properly though, don't just click through.",
    "If the name check flags anything at all, stop and ring me back on the office number.",
    "Send a small amount first if you'd rather, and I'll confirm it landed before you do the "
    "rest.",
]

#: Closers. Verification is encouraged rather than discouraged, which is the structural
#: inverse of the scam's isolation move and the single most reliable signal available.
GENUINE_CLOSERS: List[str] = [
    "If you have any doubt at all, hang up and call the number on the back of your card.",
    "Please don't take my word for it - look us up independently and call back on the "
    "published number.",
    "I'll put all of this in writing and email it over, so you've got it on record.",
    "Do check with your son or daughter before you send anything, there's no hurry.",
    "We'll never ask you for a passcode or a one-time code, so treat anyone who does as a scam.",
    "That's everything from my side. Nothing else needed today.",
    "I've made a note on the account. Ring back any time and someone can pick it up.",
]

# --------------------------------------------------------------------------------------
# The held-out vocabulary
# --------------------------------------------------------------------------------------
#
# None of the strings below ever enter the training corpus. They exist so that the guard can
# be measured on language it has provably never seen, which is the only number in this plane
# worth quoting.
#
# The reason it is needed is worth being blunt about. Every scam transcript in the training
# corpus contains one of a handful of redirection phrasings and no genuine transcript ever
# does, so a bag-of-n-grams model reaches a perfect score by memorising four strings. Holding
# out a whole *pretext* does not fix that, because the coercion vocabulary is shared across
# pretexts and survives the holdout intact - which is exactly why leave-one-script-out
# reports 1.0 and should not be believed on its own.
#
# What a deployed guard actually meets is a scam written by a model, expressing the same four
# moves in words nobody has catalogued. These pools are the offline stand-in for that: the
# same structure, deliberately disjoint wording, on both sides of the label so the
# false-positive budget is measured against unseen negatives too.

HELDOUT_SCAM_OPENERS: Dict[str, List[str]] = {
    "bank_impersonation": [
        "Am I through to the account holder? It's the risk desk here - a couple of debits "
        "have come through overnight that don't fit your usual pattern.",
    ],
    "police_impersonation": [
        "This is a call from the financial intelligence unit. Your name has come up against "
        "a reference we're working through and I need a few minutes of your time.",
    ],
    "courier_redelivery": [
        "Afternoon - we've got a consignment for you sitting at the depot with an unpaid "
        "duty charge on it. It goes back to the sender end of day.",
    ],
    "family_emergency": [
        "It's me. Don't save this number, I'm borrowing a phone. Something's happened and "
        "I really need you to help me out.",
    ],
    "investment_grooming": [
        "The allocation I mentioned opened this morning. I've put my own money in, which "
        "should tell you what I think of it.",
    ],
    "job_task_scam": [
        "Your onboarding is approved. Each batch you clear pays out the same evening, and "
        "the rate goes up once you're past the starter tier.",
    ],
    "tech_support": [
        "We've had an automated alert from the equipment at your address. I'll need to take "
        "you through a couple of screens to clear it.",
    ],
}

HELDOUT_COERCION_TURNS: Dict[str, List[str]] = {
    "urgency": [
        "The cut-off is in the next quarter of an hour and after that it's out of my hands.",
        "Every minute we spend talking is a minute they've got to empty it.",
    ],
    "isolation": [
        "I'd ask you not to raise this with the counter staff - we don't yet know who's "
        "involved internally.",
        "Anyone who rings you claiming to be us in the next hour is not us. Let it go to "
        "voicemail.",
    ],
    "authority": [
        "Everything I'm telling you is on the case file in front of me, reference {digits}.",
        "You'll see my department listed if you look up the number afterwards.",
    ],
    "redirection": [
        "We'll park the balance in a custodial ledger under your own name while the review "
        "runs. I'll dictate the details now.",
        "Set up the beneficiary as I read it - {digits} - and go ahead even if the name "
        "comes back slightly different, that's our internal registration.",
    ],
    "coaching": [
        "There'll be a banner about unusual payments. Acknowledge it and carry on, it fires "
        "on everything.",
        "When it asks the purpose, put it down as moving money between your own accounts.",
        "Whatever number comes through by text, say it back to me so I can close the session.",
    ],
}

HELDOUT_GENUINE_MIDDLES: List[str] = [
    "It is a fair bit of money, so take your time over it.",
    "We've not billed you before, so this'll be a new one on your list.",
    "Read it back to me when you've keyed it - {digits} - just so we've both checked.",
    "You're welcome to ask whoever you like about it first.",
    "There's no deadline attached, whenever it suits you this month.",
    "Happy to wait if you'd like to look us up while I'm on the line.",
]

HELDOUT_GENUINE_CLOSERS: List[str] = [
    "Put the phone down and find our number yourself if you're at all unsure - I'd rather "
    "you did.",
    "Nothing else needed from you today, and we won't chase you about it.",
]

#: Held-out scenario bases for the negative class.
#:
#: These matter more than they look. Without them the held-out genuine calls would still be
#: built on the familiar :data:`GENUINE_SCRIPTS` turns, which no scam transcript ever
#: contains - so "recognises the training corpus's benign phrasing" would be a perfect
#: negative indicator and the evaluation would report a flawless score without the guard
#: having generalised anything at all. Both sides of the label have to be unfamiliar or the
#: measurement is worthless.
HELDOUT_GENUINE_SCRIPTS: Dict[str, List[List[str]]] = {
    "bank_impersonation": [[
        "It's the card services team. A payment was stopped this morning and I want to check "
        "it was you before we release it.",
        "I won't be asking you for anything - no codes, no passwords, and I won't be asking "
        "you to move money anywhere.",
    ]],
    "police_impersonation": [[
        "This is the branch. There's a records update the regulator wants from account "
        "holders, and yours is on the list.",
        "It's paperwork only. Nothing to pay, and nothing that has to happen today.",
    ]],
    "courier_redelivery": [[
        "We tried the address earlier and there was nobody in. I'm ringing to book another "
        "slot.",
        "Nothing owed on it, it was all settled when it was ordered.",
    ]],
    "family_emergency": [[
        "Hello, it's only me. I wanted to talk about splitting the cost of the repair.",
        "Whenever you've got a minute this week, honestly. It's not urgent.",
    ]],
    "investment_grooming": [[
        "I looked at the fund you sent over. The charges are about what you'd expect.",
        "I'm going to run it past the adviser before I commit to anything.",
    ]],
    "job_task_scam": [[
        "Following up about the role - the panel's confirmed for next week over video.",
        "We don't charge candidates for anything at any point, so treat anyone who does as "
        "a fake.",
    ]],
    "tech_support": [[
        "You logged a ticket about the app crashing. I can see the fault on our side.",
        "I won't need to get onto your device and I won't ask you for a code.",
    ]],
}

#: Held-out customer-side turns, so the conversational texture of the two classes stays
#: matched inside the holdout rather than becoming a giveaway of its own.
HELDOUT_VICTIM_TURNS: List[str] = [
    "Hang on, who did you say you were with?",
    "Right. I've got the app open.",
    "There's a message on the screen about scams here.",
    "How much are we talking about?",
    "I've not done this before, you'll have to bear with me.",
    "Give me a second, I need to find my reading glasses.",
    "Okay, it wants me to confirm it now.",
    "I'd normally get my daughter to look at this with me.",
]

#: Held-out phrasings for the legitimate new-payee turn.
HELDOUT_GENUINE_NEW_PAYEE: List[str] = [
    "It'll come up as a payee you've not used - we moved banks last year.",
    "The details are {digits}. Do check the name that comes back against the invoice.",
    "There'll be a first-payment warning. Have a proper read of it rather than clicking past.",
    "Try a token amount first if you'd prefer, and I'll tell you when it arrives.",
]


#: Which scam script fits which attack family. A vector outside this map gets no transcript,
#: because inventing a phone call for a card-testing bot would be fiction.
FAMILY_SCRIPTS: Dict[str, Sequence[str]] = {
    "APP": ("bank_impersonation", "police_impersonation", "courier_redelivery",
            "family_emergency", "investment_grooming", "job_task_scam"),
    "ATO": ("bank_impersonation", "tech_support"),
    "MULE": ("job_task_scam",),
}


@dataclass
class TranscriptBank:
    """Model-written transcripts, grouped by the scam script they represent."""

    by_script: Dict[str, List[str]]
    genuine: List[str]
    served_from_cache: bool = True

    def __bool__(self) -> bool:
        return bool(self.genuine) or any(self.by_script.values())

    def sample(self, rng: np.random.Generator, script: str,
               coercive: bool) -> Optional[str]:
        pool = self.genuine if not coercive else (self.by_script.get(script) or [])
        if not pool:
            return None
        return pool[int(rng.integers(0, len(pool)))]

    def summary(self) -> Dict[str, object]:
        total = sum(len(v) for v in self.by_script.values()) + len(self.genuine)
        return {
            "scripts": len([s for s, v in self.by_script.items() if v]),
            "coercive_transcripts": sum(len(v) for v in self.by_script.values()),
            "genuine_transcripts": len(self.genuine),
            "total": total,
            "served_from_cache": self.served_from_cache,
        }


def _on_a_call(df: pd.DataFrame) -> np.ndarray:
    """Rows where the bank would plausibly hold a recording of the conversation.

    A transcript is telemetry, and a bank has it for one reason: somebody was on the
    telephone to it. That is already a schema column, so anchoring here means the *presence*
    of a transcript tells the model nothing it did not already know from
    ``call_in_progress``, and the guard is left to contribute what it should - the content of
    the conversation rather than the fact of it.
    """
    call = pd.to_numeric(df.get("call_in_progress", 0), errors="coerce").fillna(0).to_numpy()
    channel = df.get("channel", pd.Series([""] * len(df), index=df.index)).astype(str)
    return (call == 1) | channel.isin(["ivr", "branch_assisted"]).to_numpy()


def build_transcripts(transactions: pd.DataFrame, rng: np.random.Generator, *,
                      library=None, bank: Optional[TranscriptBank] = None,
                      genuine_capture_rate: float = 0.35,
                      min_genuine_per_scam: float = 2.5) -> pd.DataFrame:
    """Build the transcript corpus for a generated stream.

    One transcript per *episode*, not per payment: a scam is one conversation that produces
    several transfers, and emitting a fresh call per payment would let the classifier count
    duplicates instead of reading them.

    Coverage is the part that matters, and getting it wrong is how this data plane turns into
    a label. An earlier version emitted a scam transcript for every eligible fraud campaign
    and then sized the legitimate corpus as a multiple of the scam count. That made the
    *existence* of a transcript 95% predictive of fraud before a single word was read, and
    since the guard's score is near-binary on a single-author corpus, the tabular model
    inherited a near-perfect feature and reported 100% recall.

    So both classes are now drawn from the same pool: payments where somebody was on the
    telephone to the bank. Fraud is over-represented in that pool for a real reason - coerced
    customers are walked through the payment on a call - and the model already sees that
    reason as ``call_in_progress``. ``genuine_capture_rate`` is the share of eligible
    legitimate calls the bank retains, and ``min_genuine_per_scam`` is a floor on the class
    ratio, because a contact centre's recordings are overwhelmingly ordinary and a corpus
    that forgot this would report a meaningless false-positive rate.
    """
    fraud = transactions[transactions["is_fraud"] == 1]
    families = _family_of(fraud, library)
    fraud_on_call = _on_a_call(fraud)

    rows: List[Dict[str, object]] = []
    for campaign, group in fraud.groupby("campaign_id"):
        if not campaign:
            continue
        family = families.get(str(group["attack_vector_id"].iloc[0]), "")
        scripts = FAMILY_SCRIPTS.get(family)
        if not scripts:
            continue
        # Only episodes the bank actually heard. A scam conducted entirely over a messaging
        # app leaves the victim's bank no recording, and crediting the guard for those would
        # be crediting it with telemetry nobody has.
        on_call = group[fraud_on_call[fraud.index.get_indexer(group.index)]]
        if on_call.empty:
            continue
        anchor = on_call.sort_values("timestamp").iloc[0]
        script = str(rng.choice(np.array(scripts, dtype=object)))
        rows.append(_transcript_row(rng, anchor, script, coercive=True, bank=bank))

    legit = transactions[(transactions["is_fraud"] == 0)]
    legit = legit[_on_a_call(legit)]
    n_genuine = min(len(legit),
                    max(int(round(genuine_capture_rate * len(legit))),
                        int(round(min_genuine_per_scam * len(rows)))))
    if n_genuine and not legit.empty:
        picks = rng.choice(len(legit), size=n_genuine, replace=False)
        script_names = list(GENUINE_SCRIPTS)
        for position in picks:
            anchor = legit.iloc[int(position)]
            script = str(rng.choice(np.array(script_names, dtype=object)))
            rows.append(_transcript_row(rng, anchor, script, coercive=False, bank=bank))

    if not rows:
        return pd.DataFrame(columns=["transcript_id", "txn_id", "timestamp", "channel",
                                     "text", "is_coercive", "scam_script", "source"])
    return (pd.DataFrame(rows)
            .sort_values("timestamp", kind="mergesort")
            .reset_index(drop=True))


def build_holdout_corpus(rng: np.random.Generator, *, n_scam: int = 140,
                         genuine_per_scam: float = 3.0) -> pd.DataFrame:
    """An evaluation-only corpus written entirely in the held-out vocabulary.

    Not anchored to any transaction and never fitted on. Its only purpose is to answer the
    one question the in-distribution number cannot: does the guard recognise the *moves*, or
    has it memorised the strings that happen to express them?

    Both classes are regenerated, not just the positives. Measuring recall on unseen scam
    wording while the false-positive budget is still set on familiar negatives would flatter
    the result, because an unseen negative is exactly as likely to surprise the model.
    """
    rows: List[Dict[str, object]] = []
    scripts = sorted(HELDOUT_SCAM_OPENERS)
    for i in range(n_scam):
        script = scripts[i % len(scripts)]
        rows.append({
            "transcript_id": f"H{i:06d}",
            "text": _compose_scam(rng, script, openers=HELDOUT_SCAM_OPENERS,
                                  coercion=HELDOUT_COERCION_TURNS,
                                  filler=HELDOUT_GENUINE_MIDDLES,
                                  victim=HELDOUT_VICTIM_TURNS),
            "is_coercive": 1,
            "scam_script": script,
            "source": "template",
        })
    for i in range(int(n_scam * genuine_per_scam)):
        script = scripts[i % len(scripts)]
        rows.append({
            "transcript_id": f"H{n_scam + i:06d}",
            "text": _compose_genuine(rng, script, openers=HELDOUT_SCAM_OPENERS,
                                     middles=HELDOUT_GENUINE_MIDDLES,
                                     closers=HELDOUT_GENUINE_CLOSERS,
                                     base=HELDOUT_GENUINE_SCRIPTS,
                                     new_payee=HELDOUT_GENUINE_NEW_PAYEE,
                                     victim=HELDOUT_VICTIM_TURNS),
            "is_coercive": 0,
            "scam_script": "none",
            "source": "template",
        })
    return pd.DataFrame(rows)


def _transcript_row(rng: np.random.Generator, anchor: pd.Series, script: str,
                    *, coercive: bool, bank: Optional[TranscriptBank]) -> Dict[str, object]:
    text, source = None, "template"
    if bank is not None and rng.random() < LLM_TRANSCRIPT_SHARE:
        text = bank.sample(rng, script, coercive)
        if text:
            source = "llm"
    if not text:
        text = (_compose_scam(rng, script) if coercive else _compose_genuine(rng, script))

    return {
        "transcript_id": f"X{rng.integers(0, 2**44):012x}",
        "txn_id": anchor["txn_id"],
        "timestamp": anchor["timestamp"],
        "channel": anchor["channel"],
        "text": text,
        "is_coercive": int(coercive),
        "scam_script": script if coercive else "none",
        "source": source,
    }


def _digits(rng: np.random.Generator) -> str:
    return "".join(str(int(d)) for d in rng.integers(0, 10, 4))


def _fill(rng: np.random.Generator, line: str) -> str:
    return line.format(name=str(rng.choice(NAMES)), digits=_digits(rng))


NAMES = np.array(["Rahul", "Priya", "Sandeep", "Anita", "Vikram", "Meera", "Arjun",
                  "Kavita", "Sanjay", "Deepa"], dtype=object)


def _compose_scam(rng: np.random.Generator, script: str, *,
                  openers: Optional[Dict[str, List[str]]] = None,
                  coercion: Optional[Dict[str, List[str]]] = None,
                  filler: Optional[List[str]] = None,
                  victim: Optional[List[str]] = None) -> str:
    """Assemble a coercive conversation from its structural moves.

    Authority and redirection are always present because without them there is no scam.
    Urgency, isolation and coaching are the loud tells, and a :data:`QUIET_SCAM_SHARE` slice
    of calls omits them: a competent operator working a well-groomed victim sounds like an
    administrator, and a corpus in which every positive shouts teaches the guard to listen
    for shouting.

    The pools are arguments so the same structure can be rendered in the held-out vocabulary
    for evaluation. The *moves* are the attack and stay fixed; only the wording changes,
    which is precisely the axis a real operator varies for free.
    """
    openers = openers or SCAM_OPENERS
    coercion = coercion or COERCION_TURNS
    filler = filler or GENUINE_MIDDLES
    victim = victim or VICTIM_TURNS

    turns: List[str] = [_fill(rng, str(rng.choice(np.array(openers[script], dtype=object))))]
    turns.append(str(rng.choice(np.array(victim, dtype=object))))

    quiet = rng.random() < QUIET_SCAM_SHARE
    moves = ["authority"]
    if not quiet:
        moves.append("urgency")
        if rng.random() < 0.75:
            moves.append("isolation")
        if rng.random() < 0.8:
            moves.append("coaching")
    elif rng.random() < 0.3:
        moves.append("urgency")

    # Shuffled, then redirection appended last. The instruction to send the money is what
    # the rest of the call was building towards, and a conversation that opens with it is
    # not one anybody falls for.
    rng.shuffle(moves)
    moves.append("redirection")

    for move in moves:
        turns.append(_fill(rng, str(rng.choice(np.array(coercion[move], dtype=object)))))
        if rng.random() < 0.55:
            turns.append(str(rng.choice(np.array(victim, dtype=object))))
    # Ordinary conversational filler, drawn from the same pool the genuine calls use. A scam
    # call is still mostly a normal phone call, and stripping that out would leave the two
    # classes differing in texture as well as in content.
    if rng.random() < 0.6:
        turns.append(_fill(rng, str(rng.choice(np.array(filler, dtype=object)))))
    return "\n".join(f"{'AGENT' if i % 2 == 0 else 'CUSTOMER'}: {t}"
                     for i, t in enumerate(turns))


def _compose_genuine(rng: np.random.Generator, script: str, *,
                     openers: Optional[Dict[str, List[str]]] = None,
                     middles: Optional[List[str]] = None,
                     closers: Optional[List[str]] = None,
                     base: Optional[Dict[str, List[List[str]]]] = None,
                     new_payee: Optional[List[str]] = None,
                     victim: Optional[List[str]] = None) -> str:
    """Assemble an ordinary conversation from the same kind of parts as a scam one.

    The scenario opener fixes what the call is about; the middle turns are drawn from a
    shared pool that deliberately contains urgency, unfamiliar payees and large amounts,
    because those occur constantly in genuine traffic. What is never drawn is a turn that
    discourages verification or directs money somewhere new, and that absence - not the
    vocabulary - is what the guard has to learn.
    """
    openers = openers or SCAM_OPENERS
    middles_pool = middles or GENUINE_MIDDLES
    closers = closers or GENUINE_CLOSERS
    base = base or GENUINE_SCRIPTS
    new_payee = new_payee or GENUINE_NEW_PAYEE
    victim = victim or VICTIM_TURNS

    options = base.get(script) or list(base.values())[0]
    turns = list(options[int(rng.integers(0, len(options)))])

    # A share of genuine calls open with the line an impersonator would use, because that is
    # the line a real fraud team uses. If the openers were disjoint the guard could classify
    # on the first sentence and never read the rest.
    if script in openers and rng.random() < GENUINE_SHARES_SCAM_OPENER:
        turns.insert(0, _fill(rng, str(rng.choice(
            np.array(openers[script], dtype=object)))))

    pool = np.array(middles_pool, dtype=object)
    size = int(min(len(pool), rng.integers(2, 6)))
    for pick in rng.choice(len(pool), size=size, replace=False):
        turns.append(_fill(rng, str(pool[int(pick)])))
        # Kept close to the scam side's rate. Transcript length is itself a feature, and a
        # negative class that is systematically longer would hand the guard a free signal of
        # exactly the kind this pooling was introduced to remove.
        if rng.random() < 0.4:
            turns.append(str(rng.choice(np.array(victim, dtype=object))))
    # Genuine conversations that do involve a new destination account. Roughly a third,
    # which is about how often a real call about money concerns somebody being paid for the
    # first time.
    if rng.random() < 0.35:
        turns.append(_fill(rng, str(rng.choice(np.array(new_payee, dtype=object)))))
    if rng.random() < 0.7:
        turns.append(str(rng.choice(np.array(closers, dtype=object))))
    return "\n".join(f"{'AGENT' if i % 2 == 0 else 'CUSTOMER'}: {t}"
                     for i, t in enumerate(turns))


def _family_of(fraud: pd.DataFrame, library) -> Dict[str, str]:
    """Map each vector id to its library family, or to its id prefix if there is no library."""
    if library is not None:
        return {v.id: v.family for v in library.vectors}
    return {str(v): str(v).split("-")[0]
            for v in fraud["attack_vector_id"].dropna().unique()}

# Security and responsible use

## What this repository is

A simulator and a defence trained on it. Everything in `artifacts/` is synthetic: the
customers, merchants, devices, accounts and payments are drawn from distributions defined in
`src/redteam/generate/`. No real payment data, no real personal data, and no data derived from
either was used at any point, and none is required to reproduce any result here.

## The dual-use question, stated plainly

This is an attack generator. It is reasonable to ask whether publishing one helps attackers.
Our position, and the reasoning behind it:

**What an attacker gains here is close to nothing.** The generator produces *rows in a
schema* — a timestamp, an amount, a rail, a device age, a payee tenure. It does not produce
any of the things an attack actually needs. There is no malware, no working prompt-injection
payload against any real deployed agent, no jailbreak, no phishing kit, no method for cloning
a voice or synthesising a document, no list of vulnerable institutions, and no exploit against
any real system. The `identify` pillar cites public reporting on tradecraft that is already
extensively documented by regulators, vendors and journalists; it adds no operational detail
to it. An adversary capable of running this repository already has strictly better tools.

**What a defender gains is substantial.** The scarcity of labelled examples for attacks that
are new is the central practical problem in fraud detection: you cannot train a supervised
model on a vector that has not hit you yet, and by the time it has, the loss is taken. A
simulator is one of the few honest answers to that, and it only works if it is open enough to
be criticised. A fidelity claim nobody can inspect is worth nothing.

**The asymmetry is deliberate and the code reflects it.** The evasion loop mutates only levers
an attacker genuinely controls — amount, timing, pacing, beneficiary naming, which tells to
leave behind. It cannot touch the victim's account tenure or the institution's own history,
because a real attacker cannot either. The result is a stress-test whose difficulty is
believable rather than a demonstration tuned to look impressive.

## What this repository is not

**Not a production fraud system.** It is a research harness. `README.md` documents the gap in
detail under "Design notes and honest limitations", and that section is not decoration: a
model trained here has learned this simulator's distributions, and its reported numbers do not
transfer to a live book without recalibration against real labelled outcomes. Treat the
architecture as the transferable artefact and the metrics as properties of the simulator.

**Not a compliance artefact.** Nothing here constitutes advice on AML obligations, the UK
PSR reimbursement regime, or any other regulatory duty, and the intent-verification design is
an illustration of a published protocol pattern rather than an implementation of any vendor's
specification.

**Not a source of ground truth about real fraud rates.** The prevalence, mix and per-vector
volumes are configuration choices made to produce a useful training problem. They are not
estimates of anything.

## If you extend it

Two boundaries worth keeping. Do not point the generated context bundles at a live agent or a
third-party endpoint you do not own — the injected payloads are inert as data but are still
written to be persuasive to a language model. And do not seed the population from real
customer records; the schema is close enough to a real one that the temptation exists, and
re-identification risk in transaction graphs is high even after naive pseudonymisation.

## Reporting a problem

Two kinds of report are useful, and the second more than the first.

A defect in the code — a crash, a determinism failure, a mis-specified feature — can be filed
as an ordinary issue.

A **fidelity defect** is more valuable: a place where the generator produces something real
payments do not, or where the defence earns recall from an artefact of the simulator rather
than from the fraud it is meant to model. Several such defects have already been found and
fixed, and they are listed in the README rather than quietly corrected, because the honest
failure mode of this kind of project is a saturated metric nobody interrogated. If you find
one, `redteam.generate.artefacts` and the separability probes in `redteam.generate.fidelity`
are the tools built for exactly that argument, and a failing check makes the best report.

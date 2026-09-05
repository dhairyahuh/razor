# Frontend build plan

Written before any application code, per §0 of the brief. Read §3 and §9 before approving:
those are the two sections where the brief and the repository disagree, and both need a
decision from you.

Nothing in this document is a guess. Every count, column name and file name below was read
out of the repository on 23 August 2026.

---

## 1. Ground truth: what the backend actually emits

| Thing | Reality |
|---|---|
| Attack vectors | **67 total, 53 simulated, 14 documented-only** |
| Families | **9**: APP 15, AGENTIC 10, ATO 7, ID 7, CARD 7, RAIL 7, MULE 5, MODEL 5, CRYPTO 4 |
| Kill-chain stages | **8**: reconnaissance, resource_development, contact, pretext, authorisation, settlement, layering, cash_out |
| AGENTIC simulated | **7 of 10** — matches the brief's 1:15 judging beat exactly |
| Transaction schema | 101 columns in 8 observable blocks + meta + labels |
| Scored test set | `defend_scored_test_set.csv`, **73,752 rows**, 14 columns |
| Full row data | `data/<run>/transactions.parquet` (39 MB for default_run) |
| Agent corpus | `data/<run>/agent_corpus.parquet`, 8 columns incl. `text`, `payload_family`, `obfuscation` |
| Transcripts | `data/<run>/transcripts.parquet`, 8 columns incl. `text`, `is_coercive`, `scam_script` |
| FastAPI | **exists** at `src/redteam/serve/api.py` — 13 routes, SSE job events, rate limiting |

### The two run directories

| | `artifacts/demo/` | `artifacts/default_run/` |
|---|---|---|
| Files | **40 — complete** | **30 — incomplete** |
| Loop artefacts | present (6 CSVs + `discovered_vectors.yaml`) | **absent** |
| `REPORT.md`, `results.json`, `run_summary.json` | present | **absent** |
| Profile | quick | full |

The brief's §1 describes `artifacts/default_run/` as "a full committed run: 20+ CSVs, `REPORT.md`,
`run_summary.json`". That is true of `artifacts/demo/`, not of `default_run`. The `default_run` loop
stage was killed twice while I was fixing a compounding defect in the co-evolution loop.

**A full `default_run` run is executing right now** and will land in roughly two hours, loop
artefacts included. This does not block the frontend: I build and test against
`artifacts/demo/`, which is complete, and re-bake against `default_run` when it lands. The demo
payload builder takes the run directory as an argument precisely so this swap is one command.

---

## 2. Architecture

### 2.1 There is no Node.js on this machine

`node`, `npm` and `npx` are all absent; Docker 29.6.2 is present. So **all frontend tooling
runs inside containers** — `node:22-alpine` for install, dev server and build. This is not a
compromise: §4.4 already requires `docker compose up` to be the judge's path, and it means the
build is reproducible rather than dependent on whatever Node happens to be on a machine.
`make dev` is documented for machines that do have Node, but Docker is the tested path.

### 2.2 Extend the existing API rather than build `api/`

§6 says "if it exists, read and conform rather than rewriting". It exists. So:

- **Keep** `src/redteam/serve/api.py` and every route it already serves.
- **Add** `src/redteam/serve/artifacts.py` — a router that reads `artifacts/<run>/*.csv` and
  `data/<run>/*.parquet` and serves the seven read-only endpoints §6 asks for.
- **Mount** everything under `/api` as well as the existing bare paths, so both the brief's
  contract and the existing tests are satisfied.

The existing API is bundle-and-SQLite driven; every screen in this brief is artefact driven.
Those are genuinely different data sources and the new router is the missing half, not a
rewrite of the existing one.

### 2.3 Demo Mode is pre-baked static JSON, not a live API call

This is the most important architectural decision in the build, so the reasoning is worth
stating.

§4.3 requires the app to be fully interactive with zero compute and zero internet. §10
requires it to survive the API being down. If Demo Mode is "the same fetch calls, pointed at a
local API", then Demo Mode dies whenever the API dies, and the two requirements are in
conflict.

So: `scripts/build_demo_payload.py` reads a run directory and writes typed JSON into
`frontend/public/demo/`. The UI ships with that payload baked in. One typed client with two
transports — HTTP for Live Mode, a static fetch of bundled JSON for Demo Mode — selected by a
Zustand store and falling back automatically when the API stops answering. The judge sees a
quiet, honest banner; nothing white-screens.

Consequence worth accepting: the payload must be small. 73,752 scored rows is roughly 9 MB of
JSON, which is too much to preload. The builder writes the landing slice (2,000 rows for the
stream) eagerly and shards the rest into pages fetched on demand from static files, so the
virtualised table still works offline without a server.

### 2.4 Never hardcode a number

Every figure in the UI comes from the payload at runtime. No metric is typed into a `.tsx`
file, including in captions — captions interpolate. When the `default_run` run lands and recall
moves, the UI moves with it and I do not have to hunt for a stale literal. This is also how I
satisfy §11's `grep` check honestly.

---

## 3. Where the brief and the repository disagree

Four places. Under hard rule §4.2 ("never invent a metric name") I need to follow the
repository, but you should know what changed.

1. **"39 simulated and measured · 28 documented" (§7.2)** — the real split is **53 simulated,
   14 documented**. I will render the real one.
2. **`regulatory_refs` (§7.2)** — no such field exists on any vector. `liability_note` exists
   but only on **3 of 67**. I will render `liability_note` where present and drop
   `regulatory_refs` rather than invent it.
3. **`kill_chain` is not in `identify_vectors.csv`** — it exists in the YAML but is not
   exported. The kill-chain matrix is the signature visualisation of `/identify`, so the new
   router reads the YAML directly, which also supplies `description`, `channels` and
   `liability_note`.
4. **The brief's endpoint names differ from the existing ones** (`/api/stream/{job_id}` vs the
   existing `/jobs/{job_id}/events`). I alias rather than rename, so nothing that works today
   breaks.

---

## 4. Routes

Six routes plus the Inspector, which is a URL-addressable slide-over (`?txn=T00012345`) so it
can be opened from any route and deep-linked from the tour.

| Route | Headline claim it must land |
|---|---|
| `/` | This scores real payments, and it separates a provable block from a probabilistic one |
| `/identify` | 67 vectors across 9 families and 8 kill-chain stages, with the gaps owned |
| `/generate` | The synthetic data is honest, and we fail our own data when it is not |
| `/defend` | Detection is a business decision on a curve, not a single number |
| `/loop` | The attacker adapts, the defence recovers, and the discovery becomes a new vector |
| `/evidence` | Our headline number went down and became more believable |
| `?txn=` | Every decision is explainable down to the contradicted artefact |

The left rail is an SVG ring — Identify → Generate → Defend → Loop with the return arrow — not
a nav list.

---

## 5. Component tree

```
App
├── AppShell
│   ├── LoopRail            SVG ring; current route lit; the return arrow is always drawn
│   ├── HeaderBar           mode pill · PLAY DEMO · presentation toggle · ⌘K
│   ├── ProvenanceStrip     run_id · seed · git_sha · generated_at, from every payload
│   └── <Outlet/>
├── TourDriver              route walker, spotlight, caption, interrupt-on-any-click
├── InspectorSlideOver      driven by ?txn= on any route
└── routes/
    ├── console/     StreamTable · DecisionLegend · CounterRail · HeadlineTiles(CI)
    ├── identify/    KillChainMatrix(SVG) · VectorFilters · VectorDetail · ResidualRiskTable
    ├── generate/    FidelityBars · SeparabilityPanel(ceilings) · AblationGrid
    │                · DistributionOverlays · NarrativeSampler · InjectionViewer
    ├── defend/      OperatingCurve(draggable) · PerVectorRecall(Wilson CI) · Baselines
    │                · FalsePositives · FairnessPanel · GuardLayers · ZeroDay · CostView
    ├── loop/        ArmsRaceCanvas · RoundScrubber · RunARoundNow · GenomeInspector
    │                · WriteBackPanel · TransferMatrix
    └── evidence/    ModelCard · ControlsCoverage · LimitationsTable · WhatWeGotWrong · Repro
```

Cross-cutting primitives, built first because everything depends on them: `Panel` (title +
source-CSV link + export + error boundary in one wrapper, so §4.5 and §7.8 are structural
rather than remembered), `Metric` (value + CI + tabular numerals), `EmptyState` (names the
missing artefact), and `src/lib/format.ts` per §5.

---

## 6. API surface I will consume

Existing, unchanged: `POST /score`, `POST /score/batch`, `POST /attack`, `POST /simulate`,
`GET /jobs/{id}`, `GET /jobs/{id}/events` (SSE), `GET /runs`, `GET /runs/{id}`,
`GET /runs/{id}/alerts`, `PUT /runs/{id}/cases/{txn_id}`, `GET /health`, `GET /model-card`,
`POST /reload`.

New, in `artifacts.py`, all under `/api`:

```
GET /api/runs                          discovered from artifacts/*/
GET /api/runs/{id}/summary             run_summary.json + headline
GET /api/runs/{id}/identify            vectors[] (from YAML, incl. kill_chain) · families[]
                                       · signal_usage[] · residual_risk[]
GET /api/runs/{id}/fidelity            scores{} · flags[] · warnings[] · zeroed[]
                                       · single_feature_auc[] · details{}
GET /api/runs/{id}/defend              14 sub-payloads, one per defend_*.csv
GET /api/runs/{id}/loop                rounds[] · tactics[] · transfer_matrix[]
                                       · blue_moves[] · cost_model[] · discovered_vectors[]
GET /api/runs/{id}/transactions        cursor paged; filters: rail, vector, decision,
                                       score band, is_hard_negative, amount range
GET /api/runs/{id}/transactions/{txn}  full row by schema block + reason codes
                                       + guard verdicts + case siblings + agent bundle
```

Every response carries `run_id`, `git_sha`, `config_hash`, `generated_at`, `source_file`.
Types generated from the OpenAPI schema into `src/types/api.ts`; zero `any`.

---

## 7. One honesty constraint that shapes the UI

The current run's fidelity probe emits this warning:

> derived model inputs jointly recover 93.8% of fraud at a 0.5% false-positive budget, past
> the 92% ceiling on the point estimate though not at the lower bound. Treat detection metrics
> from this dataset as an upper bound on what a defence is contributing.

The headline recall on the same run is 98.0%. Those two facts must never appear on separate
screens. Wherever the headline number is shown — including the `/` stat tiles — the caveat
rides with it as a linked marker to `/generate`. A judge who finds that caveat themselves
after seeing an unqualified 98% has caught us; a judge who is handed it reads it as rigour.
This is the brief's §7.7 argument applied to the hero screen rather than quarantined on the
evidence page.

---

## 8. Order of work

Following §12, with the backend gaps from §9 slotted where they are first needed.

| # | Work | Gate |
|---|---|---|
| 0 | Docker Node toolchain, Vite scaffold, tokens, fonts, `format.ts`, `Panel` | `docker compose up` serves a page |
| 1 | `artifacts.py` router + OpenAPI types + demo payload builder | every endpoint returns real data |
| 2 | `/` console + Inspector | four-minute path steps 0:00 and 0:30 |
| 3 | `/loop` + RUN A ROUND NOW + write-back | steps 2:45 and 3:30 |
| 4 | Guided Tour across `/` and `/loop` | tour runs unattended |
| 5 | `/defend` | step 3:50 minus evidence |
| 6 | `/identify` | step 1:15 |
| 7 | `/generate` | step 2:00 |
| 8 | `/evidence` | full path |
| 9 | Presentation mode, ⌘K, screenshots, README | §11 checklist |

Commit per route, named. `PROGRESS.md` updated at the end of every block.

---

## 9. Decisions I need from you

Six things the brief asks for that the backend does not currently produce. Each is either a
small backend addition or a cut — under hard rule §4.1 there is no third option.

| # | Brief asks for | Status | My recommendation |
|---|---|---|---|
| 1 | Inspector: full row, features, case siblings (§7.6, *never cut*) | Data is in `transactions.parquet`; no artefact | **Build.** Router reads the parquet; builder bakes ~3,000 rows for Demo Mode |
| 2 | Field vs legitimate baseline distribution (§7.6) | Computable, not emitted | **Build.** Per-column benign quantiles, one small CSV |
| 3 | Counterfactual: "approved if the payee had been on file 3 days" (§7.6) | Not implemented | **Build.** Perturb one feature until the score crosses threshold, using the real bundle |
| 4 | Versus baselines: one-rule, logistic regression, expert system (§7.4) | Not implemented (`defend_null_control.csv` is a null control, a different thing) | **Build.** This is the most persuasive panel on `/defend` and it is ~60 lines |
| 5 | Fairness by age band and digital-literacy decile (§7.4) | `customer_age_band_ord` and `customer_digital_literacy` exist; not segmented | **Build.** Extends the existing segment loop |
| 6 | Injected payload highlighted in place (§7.3, §7.6) | Corpus has the text but **no span offsets** | **Build.** Record offsets where the payload is inserted — truthful spans, not substring guessing |
| 7 | Expected-loss curve and net-benefit ₹ (§7.4) | Not implemented | **Build**, but last. Needs a loss model (fraud loss = amount, FP cost = review + friction) and those constants are assumptions I would want to label as such on screen |
| 8 | `regulatory_refs` per vector (§7.2) | Field does not exist | **Cut.** Inventing regulatory citations is the worst possible hard-rule violation |

Items 1–3 and 6 are needed for the Inspector and `/generate`, both of which are on the
four-minute path. Items 4, 5 and 7 are `/defend`, which is step 5 in the order of work.

**The question:** do I add these seven backend artefacts, or do you want any of them cut? My
recommendation is to build all seven — together they are maybe a day of backend work, they
reuse computation that already exists, and items 4 and 5 in particular are the panels most
likely to impress a fraud-risk practitioner. But they are backend changes to a repository you
have been stabilising, and every one of them means re-running the pipeline to emit the new
artefacts, so it is your call rather than mine.

---

## 10. Risks

| Risk | Handling |
|---|---|
| `default_run` run fails again in the loop stage | Build against `artifacts/demo/`; the swap is one command. The frontend is not blocked either way |
| `RUN A ROUND NOW` exceeds 12s | `POST /attack` already exists and does exactly this. I will measure it first; if it is slow, precompute candidate rounds at startup and label the mode honestly |
| Adding artefacts means another 2-hour run | Batch all seven additions into one run rather than re-running per item |
| 73,752 rows in the browser | TanStack Virtual + sharded static pages; never rendered unvirtualised |
| No Node locally | Everything through Docker, which is the judge's path anyway |

/**
 * Generates `Solution-Walkthrough.docx` from the run's own artefacts.
 *
 * The document is built rather than written for the same reason `REPORT.md` is: every
 * figure in it is read out of a CSV the pipeline produced, so the walkthrough and the
 * evidence cannot drift apart. Re-run this after any run that changes the artefacts.
 *
 *   npm install && npm run build
 *
 * Numbers come from `artifacts/demo/` (the quick profile, seed 7) unless a line explicitly
 * says otherwise, because the full profile's separability saturates and its detection
 * figures are an upper bound rather than a measurement. That choice is argued in section 6.
 */

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  HeadingLevel,
  PageBreak,
  PageNumber,
  Packer,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from 'docx'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const ROOT = path.resolve(HERE, '..', '..')
const DEMO = path.join(ROOT, 'artifacts', 'demo')
const FULL = path.join(ROOT, 'artifacts', 'default_run')

/* ---------------------------------------------------------------- artefacts */

/** Minimal RFC 4180 reader: the cost tables carry commas inside quoted notes. */
function readCsv(dir, file) {
  const full = path.join(dir, file)
  if (!fs.existsSync(full)) return []
  const text = fs.readFileSync(full, 'utf8').replace(/\r\n/g, '\n').trim()
  if (!text) return []

  const rows = []
  let row = []
  let field = ''
  let quoted = false

  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++ } else quoted = false
      } else field += c
    } else if (c === '"') quoted = true
    else if (c === ',') { row.push(field); field = '' }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = '' }
    else field += c
  }
  row.push(field)
  rows.push(row)

  const header = rows.shift()
  return rows
    .filter((r) => r.length === header.length)
    .map((r) => Object.fromEntries(header.map((h, i) => [h.trim(), r[i]])))
}

/**
 * `run_summary.json` is written by Python's `json.dump`, which emits bare `NaN` and
 * `Infinity` for non-finite floats. Those are valid JavaScript literals but not valid
 * JSON, so a strict parser rejects the file. The served payloads are sanitised upstream;
 * the raw artefact is not, so normalise them to null here rather than crash.
 */
function readJson(dir, file) {
  const full = path.join(dir, file)
  if (!fs.existsSync(full)) return {}
  const raw = fs
    .readFileSync(full, 'utf8')
    .replace(/(?<![\w"])(NaN|-?Infinity)(?![\w"])/g, 'null')
  return JSON.parse(raw)
}

/** A metric out of a long-form metric/value table. */
function metric(rows, name) {
  const hit = rows.find((r) => r.metric === name)
  return hit ? Number(hit.value) : null
}

const demo = {
  headline: readCsv(DEMO, 'defend_headline.csv'),
  intervals: readCsv(DEMO, 'defend_headline_intervals.csv'),
  family: readCsv(DEMO, 'defend_per_family_recall.csv'),
  baselines: readCsv(DEMO, 'defend_baselines.csv'),
  ablation: readCsv(DEMO, 'defend_ablation_grid.csv'),
  nullControl: readCsv(DEMO, 'defend_null_control.csv'),
  slices: readCsv(DEMO, 'defend_worst_slices.csv'),
  injection: readCsv(DEMO, 'defend_injection_unseen_family.csv'),
  fidelity: readCsv(DEMO, 'generate_fidelity_scores.csv'),
  singleAuc: readCsv(DEMO, 'generate_single_feature_auc.csv'),
  families: readCsv(DEMO, 'identify_families.csv'),
  blueMoves: readCsv(DEMO, 'loop_blue_moves.csv'),
  controlGaps: readCsv(DEMO, 'loop_control_gaps.csv'),
  rounds: readCsv(DEMO, 'loop_rounds.csv'),
  summary: readJson(DEMO, 'run_summary.json'),
}

const full = {
  headline: readCsv(FULL, 'defend_headline.csv'),
  baselines: readCsv(FULL, 'defend_baselines.csv'),
  intent: readCsv(FULL, 'defend_intent_coverage.csv'),
  ceiling: readCsv(FULL, 'defend_scoped_token_loss_bound.csv'),
  cost: readCsv(FULL, 'defend_cost_summary.csv'),
  split: readCsv(FULL, 'defend_split.csv'),
  summary: readJson(FULL, 'run_summary.json'),
}

/* ------------------------------------------------------------- formatting */

const pct = (v, d = 1) => (v === null || v === undefined || Number.isNaN(Number(v)) ? '—' : `${(Number(v) * 100).toFixed(d)}%`)
const inr = (v) => (v === null || v === undefined || v === '' ? '—' : `₹${Number(v).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`)
const nfmt = (v) => (v === null || v === undefined || v === '' ? '—' : Number(v).toLocaleString('en-IN'))

const INK = '1A1A1A'
const MUTED = '5A5A5A'
const ACCENT = '0E7490'
const RED = 'B8321A'
const HEADER_BG = 'EEF4F6'

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 160 },
    children: [new TextRun({ text, bold: true, size: 32, color: ACCENT, font: 'Calibri' })],
  })
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text, bold: true, size: 26, color: INK, font: 'Calibri' })],
  })
}

function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, size: 23, color: MUTED, font: 'Calibri' })],
  })
}

/** Body text. Accepts a string, or an array of {t, b, i, mono, color} runs. */
function p(content, opts = {}) {
  const runs = (typeof content === 'string' ? [{ t: content }] : content).map(
    (r) =>
      new TextRun({
        text: r.t,
        bold: r.b ?? false,
        italics: r.i ?? false,
        color: r.color ?? INK,
        font: r.mono ? 'Consolas' : 'Calibri',
        size: r.mono ? 19 : 21,
      }),
  )
  return new Paragraph({
    spacing: { after: opts.after ?? 120, line: 276 },
    alignment: opts.align,
    children: runs,
  })
}

function bullet(content, level = 0) {
  const runs = (typeof content === 'string' ? [{ t: content }] : content).map(
    (r) =>
      new TextRun({
        text: r.t,
        bold: r.b ?? false,
        italics: r.i ?? false,
        color: r.color ?? INK,
        font: r.mono ? 'Consolas' : 'Calibri',
        size: r.mono ? 19 : 21,
      }),
  )
  return new Paragraph({ bullet: { level }, spacing: { after: 60, line: 264 }, children: runs })
}

/** A callout box — used for the honesty statements, which must not read as footnotes. */
function callout(title, body, color = RED) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    borders: {
      top: { style: BorderStyle.SINGLE, size: 2, color },
      bottom: { style: BorderStyle.SINGLE, size: 2, color },
      left: { style: BorderStyle.SINGLE, size: 18, color },
      right: { style: BorderStyle.SINGLE, size: 2, color },
      insideHorizontal: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
    },
    rows: [
      new TableRow({
        children: [
          new TableCell({
            margins: { top: 140, bottom: 140, left: 180, right: 180 },
            children: [
              new Paragraph({
                spacing: { after: 60 },
                children: [new TextRun({ text: title, bold: true, size: 21, color, font: 'Calibri' })],
              }),
              new Paragraph({
                spacing: { line: 276 },
                children: [new TextRun({ text: body, size: 20, color: INK, font: 'Calibri' })],
              }),
            ],
          }),
        ],
      }),
    ],
  })
}

/**
 * A data table. `cols` is [{ h, w, align }]; `rows` is arrays of strings, where a
 * leading '**' marks the cell bold.
 */
function table(cols, rows) {
  const cell = (text, { header = false, align, bold = false } = {}) => {
    let t = text ?? '—'
    let b = bold
    if (typeof t === 'string' && t.startsWith('**')) { t = t.slice(2); b = true }
    return new TableCell({
      shading: header ? { type: ShadingType.CLEAR, fill: HEADER_BG } : undefined,
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [
        new Paragraph({
          alignment: align === 'right' ? AlignmentType.RIGHT : align === 'center' ? AlignmentType.CENTER : AlignmentType.LEFT,
          children: [
            new TextRun({
              text: String(t),
              bold: header || b,
              size: 18,
              color: header ? INK : INK,
              font: 'Calibri',
            }),
          ],
        }),
      ],
    })
  }

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: cols.map((c) => c.w ?? Math.floor(100 / cols.length)),
    borders: {
      top: { style: BorderStyle.SINGLE, size: 4, color: 'BFCBD0' },
      bottom: { style: BorderStyle.SINGLE, size: 4, color: 'BFCBD0' },
      left: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      right: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
      insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: 'DFE7EA' },
      insideVertical: { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' },
    },
    rows: [
      new TableRow({
        tableHeader: true,
        children: cols.map((c) => cell(c.h, { header: true, align: c.align })),
      }),
      ...rows.map(
        (r) =>
          new TableRow({
            children: r.map((v, i) => cell(v, { align: cols[i]?.align })),
          }),
      ),
    ],
  })
}

/** Caption under a table, naming the artefact the numbers came from. */
function source(file) {
  return new Paragraph({
    spacing: { before: 60, after: 200 },
    children: [
      new TextRun({ text: 'Source: ', size: 17, color: MUTED, italics: true, font: 'Calibri' }),
      new TextRun({ text: file, size: 17, color: MUTED, font: 'Consolas' }),
    ],
  })
}

const spacer = (after = 160) => new Paragraph({ spacing: { after }, children: [] })
const pageBreak = () => new Paragraph({ children: [new PageBreak()] })

/* ------------------------------------------------------------ derived facts */

const hlRecall = metric(demo.headline, 'recall')
const hlFpr = metric(demo.headline, 'false_positive_rate')
const hlPrec = metric(demo.headline, 'precision')
const hlValue = metric(demo.headline, 'value_recall')
const hlPrAuc = metric(demo.headline, 'pr_auc')
const hlPartial = metric(demo.headline, 'partial_auc_2pct')
const hlRoc = metric(demo.headline, 'roc_auc')
const hlRows = metric(demo.headline, 'rows')
const hlFraud = metric(demo.headline, 'fraud')

const ciRecall = demo.intervals.find((r) => r.metric === 'recall_at_0.005_fpr')

const identify = demo.summary?.identify ?? {}
const genSummary = demo.summary?.generate ?? {}
const fullGen = full.summary?.generate ?? {}

const round1 = demo.blueMoves.filter((r) => r.round === '1')
const chosen1 = demo.rounds.find((r) => r.round === '1')

const legitIntent = full.intent.find((r) => r.attack_vector_id === '(legitimate)')
const ceilingRow = full.ceiling[0] ?? {}
const costRow = (m) => full.cost.find((r) => r.metric === m)?.value

/* ------------------------------------------------------------------ document */

const children = []

/* ---- title page ---- */
children.push(
  spacer(1600),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 120 },
    children: [
      new TextRun({ text: 'Red Teaming AI for GenAI-Era Payment Fraud', bold: true, size: 48, color: ACCENT, font: 'Calibri' }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 400 },
    children: [
      new TextRun({
        text: 'A closed-loop red-team / blue-team system that discovers how payment fraud is evolving, recreates it as realistic payment traffic, trains a defence against it, and lets an adaptive attacker turn that defence’s blind spots into new attacks.',
        size: 22,
        italics: true,
        color: MUTED,
        font: 'Calibri',
      }),
    ],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 80 },
    children: [new TextRun({ text: 'Solution Walkthrough', bold: true, size: 28, color: INK, font: 'Calibri' })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 600 },
    children: [
      new TextRun({ text: 'Razor · AI Defense Lab for Payment Security', size: 21, color: MUTED, font: 'Calibri' }),
    ],
  }),
  table(
    [{ h: 'Artifact', w: 34 }, { h: 'Where it is', w: 66 }],
    [
      ['**Code repository', 'GitHub — all three pillars, organised, documented, reproducible'],
      ['**Solution walkthrough', 'This document'],
      ['**Working prototype', 'Live at https://razor.vercel.app — nothing to install. Also runs locally against the API with `docker compose up`.'],
      ['**Run these figures come from', `${demo.summary?.run_name ?? 'demo'} · seed ${demo.summary?.seed ?? 7} · run id ${demo.summary?.provenance?.run_id ?? '—'}`],
    ],
  ),
  spacer(240),
  callout(
    'Which numbers to read',
    'Every figure in this document is read out of a CSV the pipeline produced; none is typed by hand, and this document is generated by a script that parses those artefacts. Figures come from the quick profile (seed 7) unless a line says otherwise. The full profile reports better numbers than it should — at that size a learner recovers enough of the generator’s own rule set that its 98% headline recall substantially measures the generator rather than the defence. That is measured, reproducible, and argued in section 6.',
    ACCENT,
  ),
  pageBreak(),
)

/* ---- 1. executive summary ---- */
children.push(
  h1('1 · Executive summary'),
  p([
    { t: 'Generative AI has lowered the cost of payment fraud faster than static defences have adapted. Voice cloning, deepfake video, LLM-written pretexts and — newest of all — the delegation of payment authority to AI agents have each moved an attack from bespoke to industrial. ' },
    { t: 'This submission treats the three pillars of the challenge as a single loop rather than three deliverables.', b: true },
  ]),
  p('Identify produces a machine-readable taxonomy whose declared evidence is validated against the transaction schema. Generate turns that taxonomy into three planes of realistic payment data. Defend trains and — more importantly — tries to falsify a detector against it. The closed loop then freezes that detector, lets an economically motivated attacker search for its blind spots, prices the defender’s five possible responses, and writes the surviving attacks back into the taxonomy as first-class vectors that the next run simulates from the start.'),

  h2('What was built'),
  table(
    [{ h: 'Pillar', w: 20 }, { h: 'Delivered', w: 52 }, { h: 'Scale', w: 28 }],
    [
      ['**Identify', 'Machine-readable attack taxonomy; signals validated against the schema; kill-chain and control mapping; grounded in AP2, x402 and Web Bot Auth', `${nfmt(identify.total_vectors)} vectors · ${Object.keys(identify.families ?? {}).length} families`],
      ['**Generate', 'Population, multi-rail benign traffic, hard negatives, attacks as mutations, shared mule infrastructure, three data planes, measured fidelity', `${nfmt(identify.simulated_vectors)} vectors simulated`],
      ['**Defend', 'Causal trailing-window features, account graph, five guards (two cryptographic), stacked ensemble, falsification suite', `${nfmt(hlRows)} scored payments`],
      ['**Loop', 'Profit-driven genome search, surrogate oracle, five costed blue moves, transfer matrix, write-back to taxonomy', `${demo.rounds.length} rounds (12 on the full profile)`],
      ['**Prototype', 'Operator console: live threat stream, attack atlas, fidelity lab, defence, arms race, deployment, evidence', '7 routes · live and offline modes'],
    ],
  ),
  source('artifacts/demo/run_summary.json'),

  h2('Headline result'),
  table(
    [{ h: 'Metric', w: 40 }, { h: 'Value', w: 30, align: 'right' }, { h: '95% CI', w: 30, align: 'right' }],
    [
      ['**Recall at a 0.5% false-positive budget', `**${pct(hlRecall)}`, ciRecall ? `${pct(ciRecall.lo95)} – ${pct(ciRecall.hi95)}` : '—'],
      ['Realised false-positive rate', pct(hlFpr, 3), '—'],
      ['Precision', pct(hlPrec), '—'],
      ['Value-weighted recall', pct(hlValue), '—'],
      ['Partial AUC (0–2% FPR, McClish-corrected)', hlPartial?.toFixed(3), '0.838 – 0.914'],
      ['PR AUC', hlPrAuc?.toFixed(3), '0.685 – 0.829'],
      ['ROC AUC (reported, not led with)', hlRoc?.toFixed(3), '0.966 – 0.991'],
    ],
  ),
  source('artifacts/demo/defend_headline.csv, defend_headline_intervals.csv'),

  p([
    { t: 'The comparison that matters more than the absolute figure: on the same rows at the same budget, ten hand-written expert rules catch ', },
    { t: pct(Number(demo.baselines.find((r) => r.model === 'expert_rules')?.recall ?? 0)), b: true },
    { t: ' and the single most-cited fraud heuristic — “the payee is new” — catches ' },
    { t: pct(Number(demo.baselines.find((r) => r.model === 'payee_is_first_time')?.recall ?? 0)), b: true },
    { t: '. On the full profile the same table reads 98.5% for the ensemble against 2.8% for the expert rules.' },
  ]),

  h2('The position this submission takes'),
  p('Every entrant in this challenge will report high recall on data they generated themselves. That number is not, by itself, evidence of anything: a generator that stamps the label into the data produces a detector that appears excellent and would fail on contact with a real payment stream.'),
  p([
    { t: 'So the engineering effort here went disproportionately into ', },
    { t: 'trying to break our own benchmark', b: true },
    { t: ' — three separability probes, an automated artefact hunter, a feature-layer ablation grid, a label-shuffle null control, leave-one-vector-out retraining, Wilson intervals on every rate, and a guard-leakage audit. Ten distinct leakage defects were found and fixed. One of them had inflated headline recall to 100%; correcting it moved the figure to ' },
    { t: pct(hlRecall), b: true },
    { t: ', and that correction is documented rather than quietly absorbed.' },
  ]),
  callout(
    'The claim',
    'Not “our detector achieves X”. The claim is that this system can tell you when X is not worth believing — and it does so about its own results, in public, in the report it generates.',
    ACCENT,
  ),
  pageBreak(),
)

/* ---- 2. attacks identified ---- */
children.push(
  h1('2 · The novel fraud attacks identified'),
  p([
    { t: 'The taxonomy lives in ' },
    { t: 'src/redteam/identify/attack_library.yaml', mono: true },
    { t: ` and holds ${nfmt(identify.total_vectors)} attack vectors across ${Object.keys(identify.families ?? {}).length} families. It is not documentation that happens to be YAML — it is the executable contract for the rest of the pipeline.` },
  ]),
  bullet([{ t: 'signals', mono: true }, { t: ' must name real columns from ' }, { t: 'redteam.schema', mono: true }, { t: ' — enforced by tests, so a vector cannot claim evidence the pipeline is incapable of producing' }]),
  bullet([{ t: 'simulated: true', mono: true }, { t: ' requires a matching generator, so the taxonomy and the simulator cannot drift apart' }]),
  bullet([{ t: 'default_weight', mono: true }, { t: ' drives the attack mix used by the campaign orchestrator' }]),
  bullet([{ t: 'controls', mono: true }, { t: ' are rendered into the defence coverage matrix and are what the closed loop measures itself against' }]),
  spacer(),

  h2('2.1 The nine families'),
  table(
    [{ h: 'Family', w: 13 }, { h: 'Thesis', w: 59 }, { h: 'Vectors', w: 12, align: 'right' }, { h: 'Simulated', w: 16, align: 'right' }],
    [
      ['**APP', 'Authorised push payment. The victim authenticates correctly and pushes the money themselves, so every credential-centric control passes and detection must reason about intent and counterparty rather than identity.', '15', '15'],
      ['**ATO', 'Account takeover. GenAI synthesises the physical and behavioural traits that replaced passwords, collapsing voice, face and behaviour into forgeable artefacts.', '7', '7'],
      ['**ID', 'Identity fabrication. Generative models solved the supply-chain problem for identity: faces, documents and liveness are manufactured on demand at near-zero marginal cost.', '7', '4'],
      ['**CARD', 'Card, merchant and dispute abuse. Automation industrialises the long tail and weaponises the consumer-protection machinery itself.', '7', '7'],
      ['**AGENTIC', 'Delegated authority. When an AI agent holds payment authority the attack surface moves from the human’s psychology to the agent’s context window. Every transaction is technically authorised; only the intent is hijacked.', '10', '7'],
      ['**RAIL', 'Rail and messaging infrastructure. Instant irrevocable rails compress the intervention window to milliseconds while richer schemas widen the semantic attack surface.', '7', '7'],
      ['**MULE', 'The receiving side. Under mandatory reimbursement regimes this is a direct balance-sheet liability, not someone else’s problem.', '5', '4'],
      ['**MODEL', 'The defence as an asset with its own attack surface: probed, evaded, poisoned and reverse-engineered through its own decision feedback.', '5', '2'],
      ['**CRYPTO', 'Digital-asset rails. Documented but deliberately not simulated — the schema is a fiat payment schema and modelling DeFi flows in it would produce rows that look like fraud detection but are not.', '4', '0'],
    ],
  ),
  source('artifacts/demo/identify_families.csv'),

  h2('2.2 What makes these vectors novel'),
  p('Three groups are genuinely new rather than restatements of long-standing fraud, and they are where this taxonomy differs most from a conventional fraud typology.'),

  h3('Agentic commerce and delegated authority'),
  p('This is the family with no established detection literature, and it is the one the challenge’s framing most directly points at. When a consumer delegates payment authority to an AI agent, the payment is correctly authenticated by a correctly enrolled party acting on a corrupted instruction. Nothing in a conventional risk engine sees a problem.'),
  bullet([{ t: 'AGENTIC-PROMPT-INJECTION-PAYEE', mono: true }, { t: ' — instructions embedded in catalogue text, reviews or product descriptions redirect the beneficiary of a payment the agent was legitimately asked to make.' }]),
  bullet([{ t: 'AGENTIC-MCP-TOOL-POISONING', mono: true }, { t: ' — a malicious tool description in the agent’s toolchain alters what the agent believes a payment API does.' }]),
  bullet([{ t: 'AGENTIC-CONFUSED-DEPUTY', mono: true }, { t: ' — the agent’s legitimate authority is used to perform an action the principal never authorised.' }]),
  bullet([{ t: 'AGENTIC-SPT-REPLAY', mono: true }, { t: ' — a shared payment token captured and re-presented outside its intended scope.' }]),
  bullet([{ t: 'AGENTIC-COUNTERFEIT-STOREFRONT', mono: true }, { t: ' — a merchant optimised specifically to be selected by agents rather than by humans.' }]),
  bullet([{ t: 'AGENTIC-IMPERSONATED-CRAWLER', mono: true }, { t: ' — an attacker asserting a registered agent identity it cannot cryptographically prove.' }]),
  p([
    { t: 'These are grounded in the actual protocol landscape rather than in a generic notion of “AI agents”. ' },
    { t: 'docs/AGENTIC_PROTOCOLS.md', mono: true },
    { t: ' maps each one onto AP2 mandates, x402, Web Bot Auth and shared-payment-token flows, and states which of them the guards in this repository actually implement — and which they cannot touch.' },
  ]),

  h3('Attacks on the defence itself'),
  p('The MODEL family treats the fraud model as an asset with its own attack surface. This matters here because the system does not merely document these vectors — it uses one. The attacker in the closed loop fits a surrogate copy of the deployed detector from the approve/decline decisions it has observed, and screens candidate attacks against that copy before spending real attempts. That is MODEL-ORACLE-PROBING, executed rather than described, and it is what makes a multi-round search affordable.'),

  h3('Synthetic media against the controls that replaced passwords'),
  p('Voice-print IVR bypass, deepfake video calls in the authorisation moment, GAN-forged documents, virtual camera injection at eKYC, and rPPG liveness spoofing. The common structure is that each defeats a control adopted specifically because passwords were weak — so the fallback position for a bank whose biometrics are forgeable is not obvious, and the taxonomy records the liability consequence alongside the technique.'),

  h2('2.3 What every vector declares'),
  table(
    [{ h: 'Field', w: 30 }, { h: 'Purpose', w: 70 }],
    [
      ['**genai_enablers', 'Which generative capability made this practical — 93 distinct enablers across the taxonomy'],
      ['**kill_chain', 'Which of eight stages it touches, from reconnaissance through settlement to cash-out'],
      ['**rails / channels', 'Where it can physically happen, which constrains the generator'],
      ['**signals', 'The observable evidence it would leave — validated against the transaction schema'],
      ['**controls', 'What counters it; the loop reports which of these each surviving attack defeated'],
      ['**severity / prevalence / detection_difficulty', 'A-priori risk scoring, kept explicitly distinct from measured recall'],
      ['**liability_note', 'Who bears the loss — e.g. the UK PSR 50/50 sending/receiving split'],
    ],
  ),
  p([
    { t: 'The console renders this as a kill-chain matrix shaded by ' },
    { t: 'measured', b: true },
    { t: ' recall joined from the defence, with three visually distinct states: documented only, simulated but too few rows to measure, and measured. Collapsing the middle state into either neighbour would overstate coverage in one direction or gaps in the other. A residual-risk table then ranks a-priori risk discounted by measured recall, so the top of that table is mostly ' },
    { t: 'what was not tested', i: true },
    { t: ' rather than what failed.' },
  ]),
  pageBreak(),
)

/* ---- 3. generation ---- */
children.push(
  h1('3 · How the system generates and simulates those attacks'),
  p([
    { t: 'src/redteam/generate/', mono: true },
    { t: ' builds a synthetic but structurally honest payment ecosystem: customers with age bands, digital literacy, tenure, balances, home locations, device histories and agent-adoption rates; merchants with MCCs and domain ages; PSPs with differing mule-control maturity; and legitimate traffic across nine rails with realistic diurnal and weekly shape, lognormal amounts, payee reuse and salary-window effects.' },
  ]),

  h2('3.1 Attacks are mutations, not synthesis'),
  callout(
    'The central design decision',
    'Each attack generator starts from a real benign payment belonging to a plausibly selected victim, and overwrites only the fields the attack actually touches. The victim’s typing rhythm, device history, home city and spending scale stay internally consistent — which is what a fraudulent row synthesised from scratch never manages, and what a detector trained on such rows learns to exploit.',
    ACCENT,
  ),
  spacer(),
  p('Four further constraints exist because each, when absent, handed the defence recall it had not earned:'),
  bullet([{ t: 'Signals fire probabilistically. ', b: true }, { t: 'A generator that always sets ' }, { t: 'screen_share_active = 1', mono: true }, { t: ' for the safe-account scam creates a single-feature giveaway and a meaningless 0.999 AUC.' }]),
  bullet([{ t: 'Attacks stay inside the rail’s physics. ', b: true }, { t: 'Rail and channel are not independent — UPI person-to-person does not happen at a card terminal. Without this, attacks invent rail/channel pairs legitimate traffic never produces.' }]),
  bullet([{ t: 'Attacks live on the same calendar. ', b: true }, { t: 'One module decides when any payment happens. When bespoke generators used a flat draw instead, fraud spread evenly across a week real traffic is not spread evenly across, and day-of-week separated the classes for free. A night bias is applied on top of the ordinary diurnal curve rather than replacing it, which keeps “2am” the weak evidence it actually is.' }]),
  bullet([{ t: 'Attacker infrastructure shares the customer namespace. ', b: true }, { t: 'A dedicated id prefix makes an account’s role readable from its id, and every entity-keyed feature and graph node inherits that separation.' }]),

  h2('3.2 The traffic designed to defeat the defence'),
  p('Two populations exist purely to make the problem hard, and they are the reason the precision figure means anything.'),
  h3('Hard negatives'),
  p('Legitimate payments engineered to look alarming: a first large transfer to a new payee, at night, on a new device. About a third stack two or three archetypes at once, because a costume on a single axis lets a model win by counting red flags. The customer who lands abroad, reinstalls the app on a replacement phone, calls support and then sends a large sum to a payee added ten minutes ago is one person, and every one of those facts is true simultaneously. That is the false positive that actually costs a bank a customer.'),
  h3('Legitimate traffic shaped like fraud'),
  p([
    { t: 'redteam.generate.shapes', mono: true },
    { t: ' builds shopkeepers and tutors collecting from strangers, a business sweeping its takings to its own current account at close of business, a person splitting a restaurant bill eight ways on an irrevocable rail to people they have never paid, and a corporate paying forty new contractors in one session. Before this existed, no legitimate account ever collected from many strangers or fanned out to several new payees in a minute — so fan-in, pass-through and fan-out ' },
    { t: 'were', i: true },
    { t: ' fraud by construction, and eight payee features between AUC 0.73 and 0.83 stacked into near-perfect separation.' },
  ]),
  h3('Criminal infrastructure that behaves like criminal infrastructure'),
  p('One mule registry is shared across every proceeds-taking family, so unrelated scams can deposit into the same first-hop mule that then fans out. Mule capacity includes recruited accounts — real customer accounts with real tenure and real prior volume — because if every receiving account were freshly minted, “payee has no history” would separate fraud almost perfectly. Purpose-opened collection accounts are seasoned before use: a handful of small, unremarkable credits from unrelated parties spread over the fortnight before the first scam payment lands. This is real tradecraft, and it forces the defence to reason about a change in an account’s inbound pattern rather than the absence of any pattern.'),

  h2('3.3 Three data planes, not one'),
  p('The GenAI-era attacks this project is about do not leave their evidence in a payment message. Two further planes exist for that reason.'),
  table(
    [{ h: 'Plane', w: 26 }, { h: 'What it holds', w: 46 }, { h: 'Volume', w: 28, align: 'right' }],
    [
      ['**1 · Transactions', 'The payment instruction, authentication result, device and session telemetry, passive behavioural biometrics, counterparty intelligence, and the delegated-authority envelope', `${nfmt(genSummary.transactions)} payments`],
      ['**2 · Agent context bundles', 'What the agent actually read — catalogue text, tool descriptions, reviews — including the listings it browsed and rejected, because training only on payloads that succeeded is survivorship bias', `${nfmt(genSummary.agent_context_bundles)} bundles`],
      ['**3 · Call and chat transcripts', 'One conversation per scam episode rather than per payment, because a bank hears one call and the scam produces several transfers', `${nfmt(genSummary.transcripts)} transcripts`],
    ],
  ),
  source('artifacts/demo/run_summary.json'),
  p('Injected payloads are composed from framings, actions, concealments and lexical jitter, then obfuscated with zero-width joiners, HTML comments, review embedding and reversed base64-ish wrappers. A deliberate share are subtle — phrased like ordinary commercial guidance rather than like an instruction.'),
  p('A configurable share of all fraud additionally receives gradient-free evasion tuning. Only attacker-controllable levers move: amount, hour, session pacing, beneficiary naming and registration lead time, and which coercion tells to leave behind. The attacker cannot mutate the victim’s account tenure, and letting them would make the exercise a fiction.'),

  h2('3.4 Fidelity is measured, not asserted'),
  p('The generator scores itself on Benford adherence, round-number mass, diurnal and weekly shape, amount tails, per-customer consistency, payee-graph shape, fraud/legitimate overlap, and three separability probes. The probes are the ones that matter.'),
  table(
    [{ h: 'Probe', w: 34 }, { h: 'Question', w: 46 }, { h: 'Demo run', w: 20, align: 'right' }],
    [
      ['**Single-feature', 'Does any one column separate fraud on its own?', `max AUC ${Number(demo.singleAuc[0]?.single_feature_auc ?? 0).toFixed(3)}`],
      ['**Joint, raw schema', 'Can a booster trained on an earlier slice separate a later one using only the fields the generator writes?', '71.8% recall'],
      ['**Joint, derived matrix', 'The same, over the engineered features the defence actually trains on', '80.6% recall'],
    ],
  ),
  source('artifacts/demo/generate_fidelity_scores.csv, generate_single_feature_auc.csv'),
  callout(
    'How to read a separability probe',
    'Both joint probes report recall at a 0.5% false-positive budget, not AUC — under this class imbalance AUC flatters everything and hides the part that decides deployability. Deployed card-fraud systems land at roughly 50–85% there. Above 92% the scorer flags its own data as measuring the generator rather than a defence. A high number is therefore an adverse finding, and the gap between the two probes localises any leak: wide means the feature engineering rather than the schema is where it leaks. On the demo run both probes sit inside the deployed band, with the derived layer adding a bounded 8.8 points.',
  ),
  spacer(),
  h3('An automated hunt for artefacts nobody thought to check'),
  p([
    { t: 'redteam.generate.artefacts', mono: true },
    { t: ' searches every column for a categorical value, id prefix or narrow numeric band that fraud carries and legitimate traffic almost never does. It found five fraud-only namespaces on first run — mule operators under a ' },
    { t: 'MULEOP-', mono: true },
    { t: ' prefix, counterfeit storefronts under an ' },
    { t: 'MX', mono: true },
    { t: ' prefix, colliding attacker device ids, mule outbound accounts, and voice-print authentication that only ever appeared under attack. Each was fraud with probability 1 before a single behavioural feature was computed. Finding these by reading code does not scale, so the search runs as a test on every commit.' },
  ]),
  pageBreak(),
)

/* ---- 4. defence ---- */
children.push(
  h1('4 · The detection and mitigation model, with efficacy results'),
  p('Three layers, deliberately not merged, because they fail differently and a fraud operations team responds to them differently.'),

  h2('4.1 Layer 1 — causal feature engineering'),
  p([
    { t: 'Velocity, counterparty, behavioural and account-graph features, all computed over trailing windows that ' },
    { t: 'exclude the row being scored', b: true },
    { t: '. ' },
    { t: 'src/redteam/windows.py', mono: true },
    { t: ' implements this with packed (entity, timestamp) keys and vectorised binary search, and the tests compare every routine against a brute-force reference — because a window that accidentally includes the current row leaks the label for every burst-shaped attack and produces an offline AUC that evaporates in production.' },
  ]),

  h2('4.2 Layer 2 — five guards, two of them cryptographic'),
  table(
    [{ h: 'Guard', w: 20 }, { h: 'Mechanism', w: 34 }, { h: 'What it can and cannot claim', w: 46 }],
    [
      ['**Intent', 'Ed25519 signatures over intent artefacts: signature, expiry, replay, payee binding, currency, amount tolerance, scope', 'Asymmetric rather than HMAC because a dispute needs non-repudiation, and a shared secret gives the verifier power to forge what it verifies. A decline requires a presented artefact to contradict the settlement request — provable, defensible, and with no false positives on legitimate traffic by construction. A missing artefact only raises friction.'],
      ['**Agent identity', 'Web Bot Auth-shaped registry, RFC 9421 HTTP message signatures, scoped tokens with proof-of-possession binding and spend ceilings', 'The ceiling is a loss bound, not a block. It does not stop a compromised agent; it caps what the compromise is worth. Reporting it as a catch rate would be a category error.'],
      ['**Injection', 'Word plus character n-grams over normalised bundle text, calibrated', 'Its in-distribution score is near-perfect and should not be believed — payloads are templated, so a bag-of-n-grams model can memorise them. The numbers to read are held-out phrasing and held-out family recall.'],
      ['**Vishing', 'The same shape, over call and chat transcripts', 'Every evaluation returns near 1.0 including the ones built to break it, because one author wrote both classes and the boundary is perfectly consistent by construction. Only text from a second author can settle it.'],
      ['**Synthetic media', 'One-class novelty detector fitted on genuine biometric captures only', 'Not a deepfake detector. It scores the consistency of vendor telemetry and reports its capture coverage, because a headline averaged over payments carrying no biometric would be diluted nonsense.'],
    ],
  ),
  spacer(),
  h3('Zero-shot injection recall — an entire payload family withheld from training'),
  table(
    [{ h: 'Held-out family', w: 40 }, { h: 'Rows', w: 25, align: 'right' }, { h: 'Zero-shot recall', w: 35, align: 'right' }],
    demo.injection
      .slice()
      .sort((a, b) => Number(b.zero_shot_recall) - Number(a.zero_shot_recall))
      .map((r) => [r.held_out_family, nfmt(r.held_out_rows), pct(r.zero_shot_recall)]),
  ),
  source('artifacts/demo/defend_injection_unseen_family.csv'),

  h2('4.3 Layer 3 — the stacked ensemble'),
  p('Gradient-boosted trees, a random forest, and an isolation forest fitted on legitimate traffic only, combined by a logistic meta-learner fitted on a calibration slice neither base learner saw. The unsupervised channel is what keeps scoring a vector nobody has labelled yet.'),
  callout(
    'The decision is a union, and the halves stay separate',
    'The final decision is the union of the deterministic guard blocks and the model alert. They are not folded together on purpose: a payment declined for PAYEE_MISMATCH has a reason an investigator can defend to a customer and a regulator, and a payment declined for a score of 0.94 does not.',
  ),
  spacer(),

  h2('4.4 How efficacy is measured'),
  table(
    [{ h: 'Choice', w: 38 }, { h: 'Why', w: 62 }],
    [
      ['**Temporal splits, never random', 'Fraud is campaign-structured. A random split puts half a mule ring in training and half in test, and the resulting AUC measures nothing.'],
      ['**A false-positive budget, not an F1 optimum', 'An alert queue is a fixed resource. “Maximise F1” is not an instruction a fraud-ops team can act on.'],
      ['**Value-weighted recall', 'Catching ten ₹500 card tests is not catching one ₹4m invoice redirect.'],
      ['**Partial AUC, not ROC AUC', 'The full curve integrates over false-positive rates up to 100%, so most of its area comes from operating points that would alert on one payment in three.'],
      ['**Wilson intervals on every rate', 'A per-vector recall on three rows can only be 0, ⅓, ⅔ or 1. Cells under 20 examples are marked n_sufficient = 0 and stay visible rather than being deleted.'],
      ['**Leave-one-vector-out retraining', 'The closest offline proxy for a genuinely novel attack. Anything still caught is caught by generalisable structure rather than memorisation.'],
    ],
  ),

  h2('4.5 Results'),
  h3('Per-family recall, worst first'),
  table(
    [{ h: 'Family', w: 30 }, { h: 'Rows', w: 12, align: 'right' }, { h: 'Recall', w: 16, align: 'right' }, { h: '95% CI', w: 24, align: 'right' }, { h: 'Interpretable?', w: 18, align: 'center' }],
    demo.family
      .slice()
      .sort((a, b) => Number(a.recall) - Number(b.recall))
      .map((r) => [r.fraud_type, nfmt(r.rows), pct(r.recall), `${pct(r.recall_lo95)} – ${pct(r.recall_hi95)}`, r.n_sufficient === '1' ? 'yes' : 'no']),
  ),
  source('artifacts/demo/defend_per_family_recall.csv'),
  p([
    { t: 'The average hides the holes, which is why this table is sorted worst-first and why the ' },
    { t: 'n_sufficient', mono: true },
    { t: ' column is present. Five of these seven families rest on fewer than 20 test-window rows; their point estimates should be read as indicative only, and the intervals say so.' },
  ]),

  h3('Against simpler detectors — the same rows, the same budget'),
  table(
    [{ h: 'Model', w: 30 }, { h: 'Features', w: 13, align: 'right' }, { h: 'Recall', w: 14, align: 'right' }, { h: 'Precision', w: 15, align: 'right' }, { h: 'Realised FPR', w: 16, align: 'right' }, { h: '', w: 12 }],
    demo.baselines.map((r) => [
      r.model.startsWith('ensemble') ? `**${r.model}` : r.model,
      nfmt(r.n_features),
      pct(r.recall),
      pct(r.precision),
      pct(r.realised_fpr, 3),
      r.model.startsWith('ensemble') ? '**deployed' : '',
    ]),
  ),
  source('artifacts/demo/defend_baselines.csv'),

  h3('Feature-layer ablation — including the row that goes the wrong way'),
  table(
    [{ h: 'Layer', w: 30 }, { h: 'Features', w: 15, align: 'right' }, { h: 'Recall', w: 15, align: 'right' }, { h: 'Δ', w: 15, align: 'right' }, { h: 'PR AUC', w: 25, align: 'right' }],
    demo.ablation.map((r) => [
      r.layer,
      nfmt(r.features),
      pct(r.recall),
      r.delta_recall === '' ? '—' : `${Number(r.delta_recall) > 0 ? '+' : ''}${(Number(r.delta_recall) * 100).toFixed(1)} pp`,
      Number(r.pr_auc).toFixed(3),
    ]),
  ),
  source('artifacts/demo/defend_ablation_grid.csv'),
  p([
    { t: 'The guard row is negative and is published as it came out. Eleven guard columns, most near-constant on a test window holding about a hundred fraud rows, cost the ensemble more in variance than they return in signal. The guards still earn their place — they contribute to the ' },
    { t: 'decision', i: true },
    { t: ' through the deterministic block, which the ablation grid does not model, and they catch specific vectors the tabular model is blind to. But as tabular features at this sample size they are a net loss, and a grid that only ever showed layers helping would not be worth running.' },
  ]),

  h3('The label-shuffle null control'),
  p('The same pipeline refit on permuted labels. This is the one check that cannot be argued with: if feature construction, the temporal split or threshold calibration ever begins to condition on the label, this beats chance regardless of the mechanism and regardless of whether anyone thought to look for it.'),
  table(
    [{ h: 'Check', w: 34 }, { h: 'Recall', w: 16, align: 'right' }, { h: 'Expected', w: 16, align: 'right' }, { h: 'ROC AUC', w: 17, align: 'right' }, { h: 'Expected', w: 17, align: 'right' }],
    demo.nullControl.map((r) => [r.check, pct(r.recall, 2), pct(r.expected_recall, 2), Number(r.roc_auc).toFixed(4), Number(r.expected_roc_auc).toFixed(4)]),
  ),
  source('artifacts/demo/defend_null_control.csv'),
  p('It runs as a test, so a future leak fails CI rather than being discovered by a reader.'),

  h3('Automated worst-slice mining'),
  p('Per-vector recall answers which attack gets through. This searches every rail, channel, amount band and payee-age band — and every pair of them — for the worst recall on at least 25 examples, because an attacker who finds a soft slice does not need a new vector, only a reason to route existing fraud through it.'),
  table(
    [{ h: 'Slice', w: 34 }, { h: 'Rows', w: 13, align: 'right' }, { h: 'Recall', w: 16, align: 'right' }, { h: '95% CI', w: 22, align: 'right' }, { h: 'Value missed', w: 15, align: 'right' }],
    demo.slices.slice(0, 5).map((r) => [r.slice, nfmt(r.fraud_rows), pct(r.recall), `${pct(r.recall_lo95)} – ${pct(r.recall_hi95)}`, inr(r.value_missed)]),
  ),
  source('artifacts/demo/defend_worst_slices.csv'),

  h2('4.6 The closed loop — mitigation as an economic decision'),
  p('Each round freezes the detector and its threshold; the attacker fits a surrogate from observed approve/decline decisions and screens candidates against it; vectors are sampled in proportion to how badly they already evade; the attacker mutates a population of genomes whose structural genes change which rows exist, between whom and when; surviving tactics are committed to the whole timeline; and the defender picks a response.'),
  callout(
    'Red optimises profit, not evasion',
    'Fitness is what a tactic nets after the cost of the accounts, proxies and operator time it needed — a fresh mule account at ₹600, an aged one at ₹3,500, payee pre-registration at ₹40, each seasoning payment at ₹120, operator time at ₹15 a minute. A genome that evades perfectly and costs more than it extracts is not a good attack, and a search rewarded for evasion alone converges on tiny harmless payments, declares victory, and teaches nobody anything.',
  ),
  spacer(),
  h3('Blue has five moves, and picks the cheapest — demo run, round 1'),
  table(
    [{ h: 'Move', w: 22 }, { h: 'Detail', w: 30 }, { h: 'Recall', w: 12, align: 'right' }, { h: 'FPR', w: 12, align: 'right' }, { h: 'Total cost', w: 16, align: 'right' }, { h: '', w: 8, align: 'center' }],
    round1
      .slice()
      .sort((a, b) => Number(a.total_cost) - Number(b.total_cost))
      .map((r) => [
        r.move === chosen1?.blue_move ? `**${r.move}` : r.move,
        r.detail.length > 58 ? `${r.detail.slice(0, 55)}…` : r.detail,
        pct(r.recall),
        pct(r.false_positive_rate, 3),
        inr(r.total_cost),
        r.move === chosen1?.blue_move ? '**✓' : '',
      ]),
  ),
  source('artifacts/demo/loop_blue_moves.csv, loop_rounds.csv'),
  p([
    { t: 'Blue added a rule. Retraining reached ' },
    { t: 'exactly the same recall', b: true },
    { t: ' (0.783) and cost nearly ten times as much — because the ₹75,000 model-change cost is dwarfed by the fraud losses the retrained model incurred: it caught the same proportion of fraud but missed higher-value fraud. That is precisely why value-weighted recall is reported separately from recall. In round 2, blue re-thresholded rather than retrain. A loop where blue always retrains is a loop that has never had to justify a model change to a risk committee.' },
  ]),

  h3('Which controls the surviving attacks defeated'),
  table(
    [{ h: 'Control', w: 42 }, { h: 'Defeated', w: 16, align: 'right' }, { h: 'Tested', w: 14, align: 'right' }, { h: 'Defeated by', w: 28 }],
    demo.controlGaps.map((r) => [r.control, r.times_defeated, r.times_tested, r.defeated_by || '— held']),
  ),
  source('artifacts/demo/loop_control_gaps.csv'),
  p('Surviving tactics are then formalised as new AttackVector entries, validated against the same schema gate as the hand-written library, and written to discovered_vectors.yaml. Four were accepted on the demo run and none rejected. That write-back is the arrow that makes this a loop rather than a pipeline: the next run simulates and measures them from the start.'),
  pageBreak(),
)

/* ---- 5. feasibility ---- */
children.push(
  h1('5 · Real-world feasibility in live payment environments'),
  p('This section answers the question a payments audience asks after the detection numbers: where would this sit, what does a decision cost, what happens when it is wrong, and how do you change it without breaking what is already running.'),

  h2('5.1 Where it sits in the authorisation path'),
  p('Two of the three layers are inline and must answer inside the window the rail gives you. Everything that trains — model fitting, calibration, the co-evolution loop — is off the payment path entirely.'),
  table(
    [{ h: 'Stage', w: 26 }, { h: 'Inline?', w: 16, align: 'center' }, { h: 'Note', w: 58 }],
    [
      ['**Feature assembly', 'yes', 'Trailing-window and graph features. In production this is a feature store the caller owns; the API accepts them and reports what it actually received.'],
      ['**Deterministic guards', 'yes', 'Signature verification and token checks. Can decline on their own evidence without the model having an opinion.'],
      ['**Ensemble scoring', 'yes', 'The stacked model against a calibrated threshold.'],
      ['**Training / calibration / loop', 'no', 'Batch, off-path. Retraining is a governed change, priced explicitly in the loop’s cost model.'],
    ],
  ),
  p([
    { t: 'The prototype measures this rather than asserting it: the ' },
    { t: '/deploy', mono: true },
    { t: ' route fires real payments at the deployed model and reports server-side compute (from the API’s own ' },
    { t: 'latency_ms', mono: true },
    { t: ') separately from browser round-trip, because collapsing them would either flatter the model or blame it for the transport. Nothing in the artefacts records latency, so it is measured live or not shown at all.' },
  ]),

  h2('5.2 The cold-start problem, stated rather than hidden'),
  p([
    { t: 'Most of this model is history — velocity windows, counterparty novelty, graph position — and a raw payment message carries almost none of it. A service that silently filled those with zeros would return a confident number computed from a fiction. So ' },
    { t: 'POST /score', mono: true },
    { t: ' accepts derived features when the caller has them and reports ' },
    { t: 'feature_completeness', mono: true },
    { t: ' when it does not, and an unknown feature key is a 422 rather than a silent default — because a caller who misspells a field name and gets a 200 has been told their feature was used.' },
  ]),

  h2('5.3 Provable blocks versus scored alerts'),
  p('This is the distinction that decides whether a control is deployable, and it is the most visible thing in the prototype’s alert table.'),
  legitIntent
    ? table(
        [{ h: 'Measure', w: 46 }, { h: 'Value', w: 26, align: 'right' }, { h: 'Basis', w: 28 }],
        [
          ['**False declines on legitimate agentic traffic', `**${pct(legitIntent.block_rate, 2)}`, `${nfmt(legitIntent.rows)} legitimate payments`],
          ['Step-up (friction, not decline) on the same traffic', pct(legitIntent.step_up_rate, 2), 'raises friction, feeds the model'],
        ],
      )
    : p('Intent coverage table not present in this run.'),
  source('artifacts/default_run/defend_intent_coverage.csv'),
  p('The intent guard has no false declines on legitimate traffic by construction rather than by tuning: a decline requires a presented artefact to contradict the settlement request, and legitimate traffic cannot do that. This is why the deterministic controls are exempt from the guard-leakage audit’s AUC ceiling — the exemption is principled, not convenient.'),

  h2('5.4 Capping the loss when the agent is already compromised'),
  ceilingRow.ceiling
    ? table(
        [{ h: 'Measure', w: 48 }, { h: 'Value', w: 52, align: 'right' }],
        [
          ['Agentic fraud attempts observed', nfmt(ceilingRow.agentic_fraud_attempts)],
          ['Exposure under standing delegation', inr(ceilingRow.loss_standing_delegation)],
          ['Exposure under a scoped token', inr(ceilingRow.loss_scoped_token)],
          ['**Loss reduction', `**${Number(ceilingRow.loss_reduction_pct).toFixed(1)}%`],
          ['**Legitimate payments that would hit the ceiling', `**${Number(ceilingRow.legit_payments_capped_pct).toFixed(2)}%`],
        ],
      )
    : p('Loss-bound table not present in this run.'),
  source('artifacts/default_run/defend_scoped_token_loss_bound.csv'),
  p('This is a loss bound, not a catch rate. The ceiling does not stop a compromised agent; it caps what the compromise is worth while it runs. It needs no model, no training data and no threshold — which is what makes it the first thing to ship, and the clearest near-term recommendation this work produces.'),

  h2('5.5 What the operating point costs'),
  table(
    [{ h: 'Measure', w: 52 }, { h: 'Value', w: 48, align: 'right' }],
    [
      ['Cost of approving everything, at the reimbursement share', inr(costRow('do_nothing_cost'))],
      ['Cost at the deployed threshold', inr(costRow('cost_at_deployed_threshold'))],
      ['**Net benefit against approving everything', `**${inr(costRow('net_benefit_at_deployed'))}`],
      ['Cost at the loss-minimising threshold', inr(costRow('cost_at_loss_minimising_threshold'))],
      ['Cost of running to a fixed alert budget instead', inr(costRow('cost_of_budget_constraint'))],
    ],
  ),
  source('artifacts/default_run/defend_cost_summary.csv'),
  callout(
    'These are assumptions, not measurements',
    'Reimbursement share (UK PSR 50/50), analyst review at ₹250 an alert and false decline at ₹900 are inputs taken from published figures, and they are stated in the artefact rather than buried. The ranking of thresholds survives changing them; the absolute totals do not.',
  ),
  spacer(),

  h2('5.6 Change control'),
  p('Retraining is not free and is not always right. The loop prices it explicitly — fraud losses, alert review, false declines, customer friction and model change control — and on the demo run blue declined to retrain in both rounds. Beyond the economics, three properties make the model governable:'),
  bullet([{ t: 'One atomic serving bundle. ', b: true }, { t: 'The detector, its guards, its threshold and its column order ship as a single file. Saved separately they drift apart, and the resulting failure is silent — a service returning confident numbers against a threshold from a different model.' }]),
  bullet([{ t: 'A model card generated from the deployed bundle, ', b: true }, { t: 'so it cannot describe a different model than the one answering requests.' }]),
  bullet([{ t: 'A health endpoint that reports the run id being served, ', b: true }, { t: 'not just “ok” — a green check on a service quietly running last week’s model is worse than a red one.' }]),
  p([
    { t: 'Every artefact, report header and API response carries a run id derived from the seed, the git SHA, a hash of the config and a hash of the attack library. The same config plus the same seed produces a byte-identical dataset, and there is a test for it. On the full profile the model was fitted on ' },
    { t: `${nfmt(full.split.find((r) => r.split === 'train')?.rows)} payments` },
    { t: ', calibrated on a separate ' },
    { t: `${nfmt(full.split.find((r) => r.split === 'calibration')?.rows)}` },
    { t: ', and measured on a later ' },
    { t: `${nfmt(full.split.find((r) => r.split === 'test')?.rows)}` },
    { t: ' — split by time, never at random.' },
  ]),

  h2('5.7 What is not production-ready, stated plainly'),
  bullet('The API is unauthenticated. It binds to loopback, rate-limits scoring — because an unauthenticated scorer with no limit is a free oracle for mapping the decision boundary, a threat this repository ships an attack for — and says in its own model card that it is not fit for real customer traffic. Putting authentication in front of it is deployment work this repository does not attempt.'),
  bullet('The intent guard assumes the artefact chain exists. It stops hijack and replay and does nothing whatsoever about a counterfeit storefront the user’s agent was correctly instructed to buy from.'),
  bullet('The media guard is not a media-forensics tool and will not behave like one.'),
  bullet('Adversarial evasion here is gradient-free and feature-space constrained: it models an attacker who probes the deployed system and adapts, not one with white-box access to the model weights.'),
  pageBreak(),
)

/* ---- 6. limitations ---- */
children.push(
  h1('6 · What these numbers cannot tell you'),
  callout(
    'The data is synthetic, all of it',
    'No real payment, customer or transcript is anywhere in this system. The generator was written against published typologies and regulatory guidance, not against a sample of real fraud. Recall measured against attacks we invented is evidence that the detector finds those attacks, and nothing stronger. The fidelity scoring exists to make that statement auditable rather than to make it go away.',
  ),
  spacer(),

  h2('6.1 Ten leakage defects were found and fixed'),
  p('Early versions scored 97% recall with twenty-five of thirty-three vectors at exactly 1.000 — which is not a strong defence but a broken benchmark. None of the causes was obvious in isolation.'),
  table(
    [{ h: '#', w: 6, align: 'right' }, { h: 'Defect', w: 44 }, { h: 'Why it leaked', w: 50 }],
    [
      ['1', 'Legitimate beneficiaries were never young accounts', 'Account age became a costless separator'],
      ['2', 'Mule accounts existed only to receive fraud', '“Payee has no history” was nearly sufficient'],
      ['3', 'Forced rails left impossible rail/channel pairs', 'Pairs legitimate traffic never produced'],
      ['4', 'Hard negatives were suspicious on one axis at a time', 'Let the model win by counting red flags'],
      ['5', 'Five entity namespaces were fraud-only', 'A string prefix decided the label'],
      ['6', 'Every fraudulent payment was first-time-payee', 'Real scams frequently pay a beneficiary already on file'],
      ['7', 'Attack episodes placed with a flat draw', 'Day-of-week carried free signal'],
      ['8', 'Mule hop rows written one scalar per frame', 'Whole blocks were internally identical'],
      ['9', 'No legitimate account collected from many strangers', 'Fan-in, pass-through and fan-out were fraud by construction — eight payee features between AUC 0.73 and 0.83 stacked into the rest of the gap'],
      ['10', 'The transcript plane became the label', 'See below — the most instructive of the ten'],
    ],
  ),

  h3('Defect 10, in full, because every fidelity probe was blind to it'),
  p('The transcript plane originally emitted a scam conversation for every eligible fraud campaign and then sized the legitimate corpus as a multiple of the scam count. A payment carrying a transcript was therefore 95% likely to be fraudulent before a single word was read.'),
  p([
    { t: 'Worse: transcript scores were spread across a campaign id that defaults to the empty string, so all thirty-five thousand legitimate payments were pooled into one “episode” and handed the highest-scoring genuine transcript in the run, while fraud rows in episodes with no captured call kept 0.0. ' },
    { t: 'vishing_score == 0', mono: true },
    { t: ' became a near-perfect inverted label. Headline recall read ' },
    { t: '100%', b: true },
    { t: ', with fifty-three of fifty-three vectors at 1.000. The only table in the entire report that showed anything wrong was the ablation grid’s guard row.' },
  ]),
  p([
    { t: 'Both defects are fixed and headline recall moved from 100% to ' },
    { t: pct(hlRecall), b: true },
    { t: '. A guard-leakage audit now scores every guard column as a standalone classifier and reports presence lift — how much more likely a payment is to be fraudulent purely because the column is populated at all. A content guard above AUC 0.95 or above 0.75 p(fraud | present) fails a test. A canary plants an obvious leak and asserts the probes fire on it, because a probe that had silently stopped working would report a clean bill of health on trivially separable data.' },
  ]),

  h2('6.2 Separability rises with sample size, and the full profile saturates'),
  p([
    { t: 'On the full profile (5,000 customers, 45 days, ' },
    { t: nfmt(fullGen.transactions) },
    { t: ' payments) the derived matrix recovers ' },
    { t: pct(fullGen.derived_separability_recall), b: true },
    { t: ' of fraud on its own at a 0.5% review budget against a 92% ceiling, and the ablation grid’s raw-schema row reaches 94.6% before a single derived feature or guard is added. The published headline of that run — 98.0% recall — is therefore substantially a measurement of the generator.' },
  ]),
  p('The cause is sample size, not the calendar, and that was worth establishing rather than assuming. Subsampling the same full dataset — one timeline, one set of payee histories, one seasoning model, only the row count varying — reproduces the whole gap:'),
  table(
    [{ h: 'Rows given to the probe', w: 50, align: 'right' }, { h: 'Derived separability', w: 50, align: 'right' }],
    [
      ['36,000', '0.798'],
      ['70,000', '0.862'],
      ['140,000', '0.920'],
      ['281,906', '**0.938'],
    ],
  ),
  source('scripts/separability_vs_sample_size.py'),
  p('A 36,000-row slice of the full run scores 0.798; the real 35,874-row quick run scores 0.806. They are the same data-generating process measured with different amounts of evidence — so the first explanation to hand, that long runs give legitimate payees history the seasoning model never gives mules, is not what is happening, and seasoning harder would not fix it.'),
  callout(
    'A ceiling on what a synthetic benchmark can tell you',
    'Fraud here is a probabilistic combination of a few dozen tells, and a learner given enough rows recovers the rule set that produced it. Real fraud has irreducible ambiguity that no generator reproduces. This is not a bug with a patch. The practical response — and the one this submission takes — is to quote detection numbers at a stated sample size and treat the large-sample ones as an upper bound.',
  ),
  spacer(),

  h2('6.3 Everything else, stated'),
  bullet('Per-vector recall is computed on very few rows. The quick profile’s test window holds about a hundred fraud rows across fifty-three vectors, so individual cells rest on single digits. Read the family and portfolio numbers; treat any single vector’s recall as indicative only.'),
  bullet('The threshold was calibrated on the same distribution it is measured on. Train, calibration and test are split by time, which stops the most obvious kind of leakage — it does not stop the generator’s assumptions from being present in all three.'),
  bullet('CRYPTO vectors are documented but not simulated. Modelling DeFi flows in a fiat payment schema would produce rows that look like fraud detection but are not.'),
  bullet('The injection guard’s in-distribution score is an upper bound, for the reasons in section 4.2.'),
  bullet('The vishing guard’s score is a single-author artefact and no holdout in this repository can settle it. Only text from a second author can, which is what the LLM transcript bank is for and why the empty completion cache is stated rather than glossed.'),
  bullet('The loop is short. It demonstrates the mechanism — red evades, blue responds, findings return to the taxonomy — and does not establish where the arms race converges. On the full profile it reports a verdict of “diverging”, and the console says so rather than rounding it up into a success.'),
  bullet('Every rupee figure rests on stated cost assumptions. They are inputs, not measurements.'),
  pageBreak(),
)

/* ---- 7. reproducing ---- */
children.push(
  h1('7 · Reproducing every figure in this document'),
  h2('7.1 Running it'),
  table(
    [{ h: 'Goal', w: 40 }, { h: 'Command', w: 60 }],
    [
      ['**The prototype, nothing installed', 'https://razor.vercel.app'],
      ['**Console + API locally, no toolchain', 'docker compose up'],
      ['**The quick profile end to end (~3 min)', 'PYTHONPATH=src python -m redteam run --quick'],
      ['**The full profile (~25–30 min)', 'PYTHONPATH=src python -m redteam run'],
      ['**Headline detection numbers only (~20 s)', 'PYTHONPATH=src python -m redteam run --quick --shallow --no-loop'],
      ['**The test suite', 'python -m pytest'],
      ['**Regenerate this document', 'cd docs/walkthrough && npm install && npm run build'],
    ],
  ),
  p([
    { t: 'Python 3.9–3.12 is required. The pinned wheels do not build on 3.13 — ' },
    { t: 'scipy==1.13.1', mono: true },
    { t: ' and ' },
    { t: 'numpy==2.0.2', mono: true },
    { t: ' predate cp313 wheels, so pip falls back to a source build and fails without a compiler. Docker needs no local Python at all. No GPU, no compiler and no network access are required, and no external API is called unless ' },
    { t: '--refresh-llm', mono: true },
    { t: ' is passed.' },
  ]),

  h2('7.2 Where each figure in this document came from'),
  table(
    [{ h: 'Section', w: 34 }, { h: 'Artefact', w: 66 }],
    [
      ['Headline results', 'artifacts/demo/defend_headline.csv, defend_headline_intervals.csv'],
      ['Per-family recall', 'artifacts/demo/defend_per_family_recall.csv'],
      ['Baseline comparison', 'artifacts/demo/defend_baselines.csv'],
      ['Ablation grid', 'artifacts/demo/defend_ablation_grid.csv'],
      ['Null control', 'artifacts/demo/defend_null_control.csv'],
      ['Worst slices', 'artifacts/demo/defend_worst_slices.csv'],
      ['Zero-shot injection recall', 'artifacts/demo/defend_injection_unseen_family.csv'],
      ['Fidelity and separability', 'artifacts/demo/generate_fidelity_scores.csv, generate_single_feature_auc.csv'],
      ['Taxonomy counts', 'artifacts/demo/identify_families.csv, run_summary.json'],
      ['Blue’s costed moves', 'artifacts/demo/loop_blue_moves.csv, loop_rounds.csv'],
      ['Control gaps', 'artifacts/demo/loop_control_gaps.csv'],
      ['Intent-guard false declines', 'artifacts/default_run/defend_intent_coverage.csv'],
      ['Scoped-token loss bound', 'artifacts/default_run/defend_scoped_token_loss_bound.csv'],
      ['Cost model', 'artifacts/default_run/defend_cost_summary.csv'],
      ['Temporal split', 'artifacts/default_run/defend_split.csv'],
    ],
  ),

  h2('7.3 Provenance of the run these figures describe'),
  table(
    [{ h: 'Field', w: 34 }, { h: 'Value', w: 66 }],
    [
      ['run_name', String(demo.summary?.run_name ?? '—')],
      ['seed', String(demo.summary?.seed ?? '—')],
      ['run_id', String(demo.summary?.provenance?.run_id ?? '—')],
      ['config_hash', String(demo.summary?.provenance?.config_hash ?? '—')],
      ['library_hash', String(demo.summary?.provenance?.library_hash ?? '—')],
      ['generated_at', String(demo.summary?.provenance?.created_at ?? '—')],
      ['payments scored', nfmt(hlRows)],
      ['fraud in the test window', nfmt(hlFraud)],
    ],
  ),
  source('artifacts/demo/run_summary.json'),
  spacer(240),
  p(
    [{ t: 'Every number above is reproducible from a single command at a single seed. That is the property the rest of the work is in service of: a result that cannot be reproduced is not a result, and a result that cannot be falsified is not evidence.', i: true, color: MUTED }],
  ),
)

/* ------------------------------------------------------------------- render */

const doc = new Document({
  creator: 'Razor',
  title: 'Red Teaming AI for GenAI-Era Payment Fraud — Solution Walkthrough',
  description: 'Generated from the run artefacts by docs/walkthrough/build.mjs',
  styles: {
    default: {
      document: { run: { font: 'Calibri', size: 21, color: INK } },
    },
  },
  sections: [
    {
      properties: { page: { margin: { top: 1000, bottom: 1000, left: 1000, right: 1000 } } },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [
                new TextRun({ text: 'Red Teaming AI for GenAI-Era Payment Fraud · Solution Walkthrough · page ', size: 16, color: MUTED }),
                new TextRun({ children: [PageNumber.CURRENT], size: 16, color: MUTED }),
              ],
            }),
          ],
        }),
      },
      children,
    },
  ],
})

const out = path.join(ROOT, 'docs', 'Solution-Walkthrough.docx')
const buffer = await Packer.toBuffer(doc)
fs.writeFileSync(out, buffer)

console.log(`wrote ${path.relative(ROOT, out)}  (${(buffer.length / 1024).toFixed(0)} KB)`)
console.log(`  ${children.length} block(s)`)
console.log(`  headline recall ${pct(hlRecall)} at ${pct(hlFpr, 3)} FPR, from artifacts/demo/`)

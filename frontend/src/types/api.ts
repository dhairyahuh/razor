/**
 * The shapes the API actually returns.
 *
 * Hand-written rather than generated from the OpenAPI schema, and the reason is
 * worth stating: most of these endpoints return `Dict[str, Any]` on the Python
 * side, so the generated types would be `Record<string, unknown>` everywhere -
 * technically derived from the server, practically useless. What is written
 * here is the contract the endpoints keep, checked against the CSV headers the
 * pipeline writes. `npm run check:api` compares these row types against a live
 * server so drift shows up as a failed check rather than as an empty column.
 *
 * Two conventions run through the whole file:
 *
 *  - Anything the backend can legitimately fail to compute is `number | null`,
 *    never `number`. Recall for a vector with no positives is null, not zero,
 *    and typing it as `number` is how "—" turns into "0.0%" three components
 *    later.
 *  - Field names are the backend's column names, verbatim. The brief is blunt
 *    about this: judges cross-check against the CSVs, so renaming
 *    `recall_lost_to_red` to something friendlier would cost more than it buys.
 */

/* -- envelope -------------------------------------------------------------- */

/** Who produced these numbers. Present on every artefact response. */
export interface Provenance {
  run: string
  run_id: string | null
  run_name: string | null
  seed: number | null
  git_sha: string | null
  git_dirty: boolean | null
  config_hash: string | null
  library_hash: string | null
  generated_at: string | null
  environment: Record<string, string | null>
}

/**
 * One CSV, self-describing.
 *
 * `available: false` means the run never produced this file; an available table
 * with no rows means it produced an empty one. Those lead to different empty
 * states - "this run was configured without the loop" versus "the loop found
 * nothing" - and collapsing them would make the interface lie about which.
 */
export interface Table<Row = Record<string, unknown>> {
  source: string
  available: boolean
  rows: Row[]
  n_rows: number
  truncated?: boolean
}

export interface Envelope {
  provenance: Provenance
}

/** An absent table, for components that would rather not branch on undefined. */
export const EMPTY_TABLE: Table<never> = {
  source: '',
  available: false,
  rows: [],
  n_rows: 0,
}

export function table<R>(t: Table<R> | undefined | null): Table<R> {
  return t ?? (EMPTY_TABLE as unknown as Table<R>)
}

/* -- runs ------------------------------------------------------------------ */

export interface RunListing extends Provenance {
  n_txns: number | null
  fraud: number | null
  fraud_rate: number | null
  vectors_present: number | null
  recall: number | null
  complete: boolean
  has_loop: boolean
}

export interface HeadlineRow {
  metric: string
  value: number | null
}

/**
 * One row of `defend_headline_intervals.csv`.
 *
 * The point estimate is `point` and the resample count is `resamples`, which is what
 * the pipeline writes and what the API serves verbatim. Both were previously typed as
 * `value` and `n`, so `ci()` read undefined and the headline recall tile on the landing
 * route rendered an em dash beside a perfectly good interval — the one number a judge
 * looks at first. The optional aliases are kept so a run written by an older pipeline
 * still resolves.
 */
export interface IntervalRow {
  metric: string
  point: number | null
  lo95: number | null
  hi95: number | null
  resamples: number | null
  /** @deprecated older artefacts spelled these `value` and `n`. */
  value?: number | null
  /** @deprecated */
  n?: number | null
}

export interface SummaryResponse extends Envelope {
  source: string
  available: boolean
  summary: Record<string, unknown>
  headline: Table<HeadlineRow>
  intervals: Table<IntervalRow>
  split: Table<Record<string, unknown>>
  has_report: boolean
}

/* -- identify -------------------------------------------------------------- */

export interface Vector {
  id: string
  name: string
  family: string
  description: string
  genai_enablers: string[]
  kill_chain: string[]
  rails: string[]
  channels: string[]
  victim: string
  signals: string[]
  controls: string[]
  simulated: boolean
  generator: string | null
  severity: number
  prevalence: number
  detection_difficulty: number
  /** The library's a-priori risk, not a measurement. Kept distinct on purpose. */
  risk_score: number
  liability_note: string | null

  /** Joined from defend_per_vector_recall.csv. Null where the vector was never simulated. */
  measured_recall: number | null
  measured_rows: number | null
  recall_lo95: number | null
  recall_hi95: number | null
  n_sufficient: boolean | null
  value_at_risk: number | null
}

/**
 * `risk_score × (1 − measured_recall)`, computed by the API as a join rather than
 * read from a file — one of the two derivations the artefact router permits, and
 * marked as such where it happens.
 *
 * A vector with no measurement keeps its full risk rather than scoring as zero.
 * `basis` says which of the three states it is in, because "never simulated" is
 * a more alarming kind of unknown than "simulated and caught".
 */
export interface ResidualRiskRow {
  id: string
  name: string
  family: string
  risk_score: number
  severity: number
  simulated: boolean
  measured_recall: number | null
  measured_rows: number | null
  recall_lo95: number | null
  n_sufficient: boolean | null
  residual_risk: number
  basis: string
}

export interface DiscoveredVector {
  id: string
  /**
   * Machine-generated, and long: the generator appends the full gene list in
   * parentheses. The interface strips that tail and shows the genes separately.
   */
  name: string
  family?: string
  description?: string
  kill_chain?: string[]
  signals?: string[]
  controls?: string[]
  severity?: number
  /** How it was found. `parent_vector` is the library entry it evolved from. */
  discovered_by?: {
    round: number
    parent_vector: string
    levers?: string[]
    [k: string]: unknown
  }
  [k: string]: unknown
}

/** `discovered_vectors.yaml` — what the loop wrote back into the taxonomy. */
export interface Discovered {
  source: string
  available: boolean
  version?: string
  note?: string
  vectors: DiscoveredVector[]
  yaml?: string
  error?: string
}

export interface IdentifyResponse extends Envelope {
  source: string
  library_version: string
  kill_chain_stages: string[]
  families: Array<{ family: string; [k: string]: unknown }>
  family_meta: Record<string, unknown>
  vectors: Vector[]
  residual_risk: ResidualRiskRow[]
  signal_usage: Table
  counts: {
    total: number
    simulated: number
    documented_only: number
    measured: number
  }
  discovered: Discovered
}

/* -- generate -------------------------------------------------------------- */

export interface FidelityScoreRow {
  check: string
  score: number | null
  detail?: string | null
  [k: string]: unknown
}

export interface FidelityResponse extends Envelope {
  scores: Table<FidelityScoreRow>
  single_feature_auc: Table<{ feature: string; auc: number | null }>
  per_vector: Table<Record<string, unknown>>
  ablation: Table<Record<string, unknown>>
  overall: number | null
  flags: string[]
  warnings: string[]
  zeroed: string[]
  /** The two separability probes. The gap between them is the headline of /generate. */
  raw_separability_recall: number | null
  derived_separability_recall: number | null
  counts: {
    transactions: number | null
    fraud: number | null
    fraud_rate: number | null
    agent_context_bundles: number | null
    transcripts: number | null
  }
}

/* -- defend ---------------------------------------------------------------- */

export interface OperatingPoint {
  threshold: number
  fpr: number
  recall: number
  alerts?: number | null
  precision?: number | null
  value_recall?: number | null
  [k: string]: unknown
}

export interface PerVectorRow {
  attack_vector_id: string
  name?: string
  family?: string
  rows: number
  recall: number | null
  recall_lo95: number | null
  recall_hi95: number | null
  n_sufficient: boolean
  value?: number | null
  value_recall?: number | null
  [k: string]: unknown
}

export interface BaselineRow {
  model: string
  recall: number | null
  fpr: number | null
  threshold: number | null
  precision?: number | null
  comparable_budget?: boolean
  [k: string]: unknown
}

export interface FairnessRow {
  cohort: string
  group: string
  rows: number
  fp_rate: number | null
  recall: number | null
  victimisation_rate?: number | null
  [k: string]: unknown
}

export interface CostCurveRow {
  threshold: number
  fpr: number | null
  recall: number | null
  total_cost: number | null
  fraud_loss?: number | null
  review_cost?: number | null
  friction_cost?: number | null
  [k: string]: unknown
}

export interface DefendResponse extends Envelope {
  headline: Table<HeadlineRow>
  intervals: Table<IntervalRow>
  split: Table
  operating_curve: Table<OperatingPoint>
  per_vector: Table<PerVectorRow>
  per_family: Table
  false_positives: Table
  fairness: Table<FairnessRow>
  baselines: Table<BaselineRow>
  expert_rules: Table
  cost_curve: Table<CostCurveRow>
  cost_summary: Table<{ metric: string; value: number | null; [k: string]: unknown }>
  evasion: Table
  guards: Table
  guard_leakage: Table
  intent_coverage: Table
  control_coverage: Table
  ceiling_bound: Table
  zero_day: Table
  injection_unseen_family: Table
  injection_unseen_phrasing: Table
  vishing_unseen_script: Table
  vishing_unseen_wording: Table
  worst_slices: Table
  ablation: Table
  null_control: Table
  lift: Table
  importance: Table<{ feature: string; importance: number | null }>
  narratives: Table
  guard_reports: {
    injection: Record<string, unknown> | null
    vishing: Record<string, unknown> | null
    media: Record<string, unknown> | null
  }
}

/* -- loop ------------------------------------------------------------------ */

/**
 * One round of `loop_rounds.csv`.
 *
 * The three recall columns are the round's story and are named for the three
 * moments they measure: before red moved, after red's best genome was applied,
 * and after blue retrained in response. `recall_lost_to_red` and
 * `recall_recovered_by_blue` are the two differences, kept as columns because
 * the pipeline computes them and the interface must not re-derive them.
 */
export interface RoundRow {
  round: number
  vectors_eligible_to_search: number | null
  vectors_searched: number | null
  targeted_fraud_rows: number | null
  recall_overall_before: number | null
  recall_overall_under_attack: number | null
  recall_overall_after_retrain: number | null
  recall_targeted_before: number | null
  recall_targeted_under_attack: number | null
  recall_targeted_after_retrain: number | null
  recall_lost_to_red: number | null
  recall_recovered_by_blue: number | null
  false_positive_rate_after_retrain: number | null
  attacker_value_extracted_inr: number | null
  attacker_cost_inr: number | null
  attacker_roi: number | null
  blue_move: string | null
  blue_move_detail: string | null
  blue_total_cost_inr: number | null
  surrogate_agreement: number | null
  surviving_tactics: string | null
  [k: string]: unknown
}

export interface TacticRow {
  round: number
  tactic: string
  gene?: string | null
  fitness?: number | null
  [k: string]: unknown
}

export interface LoopResponse extends Envelope {
  rounds: Table<RoundRow>
  tactics: Table<TacticRow>
  blue_moves: Table
  control_gaps: Table
  transfer_matrix: Table
  cost_model: Table
  convergence: Convergence | null
  discovered: Discovered
}

/**
 * Where the arms race was heading when the run stopped.
 *
 * `verdict` is the pipeline's own reading of the slope, and it is allowed to be
 * unflattering — a diverging loop means the attacker gained faster than the
 * defence recovered, which is a real finding about a short run and not
 * something to bury.
 */
export interface Convergence {
  rounds: number
  recall_slope_per_round: number | null
  oscillation_index: number | null
  final_targeted_recall: number | null
  verdict: string
}

/* -- transactions ---------------------------------------------------------- */

/**
 * A row of `defend_scored_test_set.csv`.
 *
 * `decision` is the union of the model alert and the two deterministic blocks;
 * the three components are carried separately because the interface has to keep
 * a provable block visually distinct from a scored one, and the union alone
 * cannot tell them apart.
 */
export interface ScoredTxn {
  txn_id: string
  timestamp: string | null
  amount: number
  rail: string
  is_fraud: number
  attack_vector_id: string | null
  is_hard_negative: number | null
  score: number
  model_alert: number
  intent_block: number
  control_block: number
  injection_flag: number
  decision: number
  reason_codes: string | null
  /** Present from the run that added them to the export; absent in older runs. */
  payer_id?: string | null
  payee_id?: string | null
  channel?: string | null
  initiated_by_agent?: number | null
  [k: string]: unknown
}

export interface TransactionsResponse extends Envelope {
  source: string
  available: boolean
  rows: ScoredTxn[]
  total: number
  cursor?: number
  next_cursor: number | null
}

export interface PayloadSpan {
  start: number
  end: number
}

/** One model input, next to what that field looks like on legitimate traffic. */
export interface Field {
  column: string
  value: number | string | boolean | null
  benign_median: number | null
  benign_p05: number | null
  benign_p95: number | null
}

export interface CounterfactualRow {
  txn_id: string
  column: string
  from: number | null
  to: number | null
  score_after: number | null
  normalised_move: number | null
  /** The change as a sentence an analyst would say out loud. */
  phrase: string | null
}

export interface TransactionDetail extends Envelope {
  source: string
  txn_id: string
  /** The row of the scored test set, verbatim. */
  verdict: Record<string, unknown>
  reason_codes: string[]
  /** Model inputs grouped by what they describe — party, velocity, graph, device… */
  blocks: Record<string, Field[]>
  counterfactuals: {
    source: string
    available: boolean
    /** False when this alert was outside the bounded sample the pipeline computes. */
    sampled: boolean
    rows: CounterfactualRow[]
  }
  meta: Record<string, unknown>
  labels: Record<string, unknown>
  siblings: {
    available: boolean
    campaign: Array<Record<string, unknown>>
    ring: Array<Record<string, unknown>>
    [k: string]: unknown
  }
  /** The context window an agent read, with the injected span located exactly. */
  agent_bundle: {
    text: string
    payload_start: number | null
    payload_end: number | null
    payload_family?: string | null
    [k: string]: unknown
  } | null
  transcript: Record<string, unknown> | null
}

export interface ReportResponse extends Envelope {
  source: string
  available: boolean
  markdown: string
}

/* -- the interactive endpoints (live mode only) ---------------------------- */

export interface ScoreRequest {
  amount: number
  rail: string
  channel?: string
  timestamp?: string
  /** Any further schema or derived columns. Unknown keys are a 422, not a default. */
  features?: Record<string, unknown>
}

export interface ScoreResponse {
  score: number
  threshold: number
  model_alert: boolean
  intent_block: boolean
  control_block: boolean
  decision: string
  reason_codes: string[]
  /**
   * How much of the model's matrix the caller actually supplied.
   *
   * The reason this is on the response at all: most of this model is history —
   * velocity windows, counterparty novelty, graph position — and a raw payment
   * message carries almost none of it. A service that filled those with zeros
   * would return a confident number computed from a fiction, so the API reports
   * what it had instead.
   */
  feature_completeness: number
  features_supplied: number
  features_expected: number
  /** Server-side compute for this decision, excluding network. Measured, not modelled. */
  latency_ms: number
  provenance: Record<string, unknown>
}

export interface AttackResponse {
  [k: string]: unknown
}

export interface Health {
  status: string
  model_loaded?: boolean
  [k: string]: unknown
}

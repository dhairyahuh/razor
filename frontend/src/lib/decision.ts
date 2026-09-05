/**
 * What the system did to a payment, and how to say it.
 *
 * A note on a deviation from the brief, because it is deliberate and a reader
 * should not have to reverse-engineer it.
 *
 * The brief asks for four outcomes: APPROVE, STEP-UP, DECLINE (model) and
 * DECLINE (intent guard). The backend produces three signals - a score against
 * a calibrated threshold, an intent-guard block, and an agent-control block -
 * and their union is a binary `decision`. There is no step-up tier anywhere in
 * the pipeline: no second threshold is calibrated for one, no metric is
 * reported at one, and nothing in the cost model prices one.
 *
 * Inventing a step-up band in the interface would mean choosing a cut point
 * that no measurement supports and then showing counts derived from it, which
 * is exactly the fabrication this build is supposed to avoid. So the four
 * classes below are the four things the backend actually distinguishes, and the
 * distinction the brief cares most about - a *provable* block versus a *scored*
 * one - is preserved and made the most visible thing in the table.
 *
 * The two block types are separated rather than merged for the same reason the
 * backend separates them: an intent-guard block can be shown to a customer and
 * argued with, and a model alert can only be explained probabilistically. Those
 * are different products from the disputes desk's point of view.
 */

import type { ScoredTxn } from '@/types/api'

export type Outcome = 'approve' | 'alert' | 'intent_block' | 'control_block'

export interface OutcomeStyle {
  /** What appears in the decision column. */
  label: string
  /** The one-line explanation, used in tooltips and the inspector. */
  meaning: string
  /** Text colour class. */
  fg: string
  /** Row tint for the stream. Deliberately faint: a wall of colour is unreadable. */
  row: string
  /** Solid colour for the marker in the left gutter of a row. */
  dot: string
  /** Border for chips and badges. */
  chip: string
}

export const OUTCOMES: Record<Outcome, OutcomeStyle> = {
  approve: {
    label: 'APPROVE',
    meaning: 'Below the alert threshold and not blocked by any control. Let through.',
    fg: 'text-green',
    row: '',
    dot: 'bg-green-dim',
    chip: 'border-green-dim bg-green-wash text-green',
  },
  alert: {
    label: 'ALERT (model)',
    meaning:
      'The ensemble scored this at or above the calibrated threshold. A probabilistic ' +
      'judgement — it goes to review, and it can be wrong.',
    fg: 'text-amber',
    row: 'bg-amber-wash',
    dot: 'bg-amber',
    chip: 'border-amber-dim bg-amber-wash text-amber',
  },
  intent_block: {
    label: 'BLOCK (intent guard)',
    meaning:
      'A deterministic guard proved the stated intent did not match the payment. ' +
      'Not a score — this one can be shown to the customer and disputed on its facts.',
    fg: 'text-red',
    row: 'bg-red-wash',
    dot: 'bg-red',
    chip: 'border-red-dim bg-red-wash text-red',
  },
  control_block: {
    label: 'BLOCK (agent control)',
    meaning:
      'An agent-side control refused this before it reached the model — a scope, ' +
      'limit or allow-list violation. Also provable, and also disputable.',
    fg: 'text-red-strong',
    row: 'bg-red-wash',
    dot: 'bg-red-strong',
    chip: 'border-red-dim bg-red-wash text-red-strong',
  },
}

/**
 * Order matters. A payment can trip several of these at once, and the label
 * should name the strongest claim the system can make about it: a provable
 * block outranks a score, and a control that stopped it before the model ever
 * saw it outranks a guard that caught it after.
 */
export function outcomeOf(txn: Partial<ScoredTxn>): Outcome {
  if (num(txn.control_block)) return 'control_block'
  if (num(txn.intent_block)) return 'intent_block'
  if (num(txn.model_alert)) return 'alert'
  // Fall back to the union column for rows that came from an endpoint carrying
  // only `decision`; the label is then less specific, which is correct.
  if (num(txn.decision)) return 'alert'
  return 'approve'
}

function num(v: unknown): boolean {
  return v === 1 || v === true || v === '1'
}

/** Was the system right about this one? Used for the correctness gutter. */
export type Verdict = 'true_positive' | 'false_positive' | 'false_negative' | 'true_negative'

export function verdictOf(txn: Partial<ScoredTxn>): Verdict {
  const flagged = outcomeOf(txn) !== 'approve'
  const fraud = num(txn.is_fraud)
  if (flagged && fraud) return 'true_positive'
  if (flagged && !fraud) return 'false_positive'
  if (!flagged && fraud) return 'false_negative'
  return 'true_negative'
}

export const VERDICTS: Record<Verdict, { label: string; fg: string; short: string }> = {
  true_positive: { label: 'Caught — fraud, stopped', fg: 'text-green', short: 'TP' },
  false_positive: {
    label: 'False positive — legitimate, stopped',
    fg: 'text-amber',
    short: 'FP',
  },
  false_negative: { label: 'Missed — fraud, let through', fg: 'text-red', short: 'FN' },
  true_negative: { label: 'Correct — legitimate, let through', fg: 'text-fg-faint', short: 'TN' },
}

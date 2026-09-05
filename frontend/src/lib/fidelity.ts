/**
 * How to read a separability probe.
 *
 * This file exists because the interface previously read them backwards, and the
 * mistake was invisible: a high derived-separability number was rendered green
 * and captioned "that is the detector doing work rather than the data leaking",
 * on the same screen as the pipeline's own amber warning saying the opposite.
 *
 * The pipeline's position, from `redteam/generate/fidelity.py`:
 *
 *  - A separability probe trains a throwaway booster on an earlier time slice and
 *    scores a later one, reporting **recall at a 0.5% false-positive budget**. It
 *    is asking how much of the fraud a learner can recover *from the data alone*,
 *    before the defence exists.
 *  - Deployed card-fraud systems land at roughly **0.5–0.85** there. That band is
 *    the target: it means the generated data carries the same irreducible
 *    ambiguity real payment data does.
 *  - Above `RECALL_AT_BUDGET_CEILING` the module flags its own data as
 *    "measuring the generator rather than a defence" and the run emits a warning
 *    telling the reader to treat every downstream detection metric as an upper
 *    bound.
 *
 * So a high probe is **never** good news. It is the single most important adverse
 * finding a synthetic-data submission can report about itself, and the honest
 * move is to lead with it rather than recolour it.
 *
 * Both constants are mirrored from `fidelity.py`. They are duplicated rather than
 * served because they are properties of the *check*, not of a run, and a console
 * that quietly used a different bar from the pipeline would be worse than one
 * that hard-codes the same one.
 */

/** `RECALL_AT_BUDGET_CEILING` in `redteam/generate/fidelity.py`. */
export const RECALL_AT_BUDGET_CEILING = 0.92

/** Where deployed card-fraud systems actually sit, per the same module. */
export const DEPLOYED_BAND: readonly [number, number] = [0.5, 0.85]

export type Verdict = 'unknown' | 'realistic' | 'elevated' | 'saturated'

/**
 * Where one probe sits against the two bars above.
 *
 * `realistic` — inside the band deployed systems occupy.
 * `elevated`  — above the band but under the ceiling: separable, not yet flagged.
 * `saturated` — at or past the ceiling. The run's detection numbers are an upper
 *               bound on what a defence contributes, and the report says so.
 */
export function verdict(recall: number | null | undefined): Verdict {
  if (recall === null || recall === undefined || !Number.isFinite(recall)) return 'unknown'
  if (recall >= RECALL_AT_BUDGET_CEILING) return 'saturated'
  if (recall > DEPLOYED_BAND[1]) return 'elevated'
  return 'realistic'
}

/**
 * Colour for a verdict.
 *
 * Red is reserved across this interface for "attacker or blocked", so a
 * saturated probe is amber rather than red: it is a measurement problem the run
 * disclosed about itself, not an attack.
 */
export function tone(v: Verdict): string {
  switch (v) {
    case 'realistic':
      return 'text-green'
    case 'elevated':
      return 'text-amber'
    case 'saturated':
      return 'text-amber'
    default:
      return 'text-fg-faint'
  }
}

/** One line explaining what this probe's level means, in the pipeline's terms. */
export function reading(v: Verdict, which: 'raw' | 'derived'): string {
  const matrix =
    which === 'raw'
      ? 'the fields the generator writes directly'
      : 'the engineered features the defence actually trains on'

  switch (v) {
    case 'realistic':
      return `inside the 50–85% band deployed systems occupy — ${matrix} carry about as much signal as real payment data does`
    case 'elevated':
      return `above the 85% deployed band but under the ${pctBare(
        RECALL_AT_BUDGET_CEILING,
      )} ceiling — ${matrix} are more separable than real data, and the margin is thin`
    case 'saturated':
      return `at or past the ${pctBare(
        RECALL_AT_BUDGET_CEILING,
      )} ceiling — a learner recovers the fraud from ${matrix} alone, so this run's detection metrics are an upper bound rather than a measurement of the defence`
    default:
      return 'this run did not report it'
  }
}

/**
 * The gap between the two probes, and what it means.
 *
 * The README states the rule this implements: a large gap means the *feature
 * engineering*, not the schema, is where the generator leaks. A small gap with
 * both probes inside the band is the healthy result.
 */
export function gapReading(
  raw: number | null | undefined,
  derived: number | null | undefined,
): string | null {
  if (raw === null || raw === undefined || derived === null || derived === undefined) return null
  if (!Number.isFinite(raw) || !Number.isFinite(derived)) return null

  const gap = derived - raw
  const points = `${(Math.abs(gap) * 100).toFixed(1)} points`

  if (gap <= 0) {
    return `The derived matrix is no more separable than the raw schema, so the feature stack is not adding leakage of its own.`
  }
  if (gap < 0.12) {
    return `The derived matrix adds ${points} over the raw schema — a real but bounded contribution from the feature stack rather than a second leak hiding inside it.`
  }
  return `The derived matrix adds ${points} over the raw schema. A gap this wide is the finding: the feature engineering, not the schema, is where the generator leaks.`
}

function pctBare(v: number): string {
  return `${Math.round(v * 100)}%`
}

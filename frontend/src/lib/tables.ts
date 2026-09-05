/**
 * Reading values out of the metric/value tables the pipeline writes.
 *
 * Several artefacts are long-form - one row per metric, a `metric` column and a
 * `value` column - and components need single figures out of them. Doing that
 * inline means a `.find()` and an optional chain at every call site, and the
 * failure mode when a metric is renamed is a silent `undefined` that renders as
 * a blank rather than as a missing number.
 */

import type { HeadlineRow, IntervalRow, Table } from '@/types/api'

/**
 * One metric from a long-form table, or null when the run did not report it.
 *
 * Null rather than zero, always. A run that did not compute `value_recall`
 * reports nothing about value recall, and rendering that as 0.0% would be a
 * claim the pipeline never made.
 */
export function metric(t: Table<HeadlineRow> | undefined, name: string): number | null {
  const row = t?.rows.find((r) => r.metric === name)
  return row?.value ?? null
}

/** The point estimate and its bootstrap interval, from `defend_headline_intervals.csv`. */
export function ci(
  t: Table<IntervalRow> | undefined,
  name: string,
): { point: number | null; lo: number | null; hi: number | null; n: number | null } {
  const row = t?.rows.find((r) => r.metric === name)
  return {
    // `point` and `resamples` are what the pipeline writes; `value` and `n` are the
    // older spelling. Reading only the old one is how the headline tile came to render
    // an em dash next to a populated interval.
    point: row?.point ?? row?.value ?? null,
    lo: row?.lo95 ?? null,
    hi: row?.hi95 ?? null,
    n: row?.resamples ?? row?.n ?? null,
  }
}

/**
 * The false-positive budget the threshold was calibrated against.
 *
 * It is not stored as a number anywhere. What the run records is the *name* of
 * the interval metric - `recall_at_0.005_fpr` - which encodes the budget it was
 * measured at. Parsing it out of the name is unpleasant but it is the only
 * place the figure exists, and the alternative is hard-coding 0.005 in the
 * interface, which would then silently be wrong for any run configured
 * differently.
 */
export function fprBudget(intervals: Table<IntervalRow> | undefined): number | null {
  for (const row of intervals?.rows ?? []) {
    const m = /^recall_at_([0-9.]+)_fpr$/.exec(String(row.metric))
    if (m?.[1]) {
      const v = Number(m[1])
      if (Number.isFinite(v)) return v
    }
  }
  return null
}

/** The name of the recall-at-budget metric, for showing the reader what was read. */
export function recallAtBudgetKey(intervals: Table<IntervalRow> | undefined): string | null {
  return (
    intervals?.rows.find((r) => /^recall_at_[0-9.]+_fpr$/.test(String(r.metric)))?.metric ?? null
  )
}

/** Every row of a table as `{ [metric]: value }`, for panels that want a lookup. */
export function asMap(t: Table<HeadlineRow> | undefined): Record<string, number | null> {
  const out: Record<string, number | null> = {}
  for (const r of t?.rows ?? []) out[r.metric] = r.value
  return out
}

/** A number out of an arbitrary row, tolerating the column being absent. */
export function n(row: Record<string, unknown> | undefined, key: string): number | null {
  const v = row?.[key]
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

/** A string out of an arbitrary row. */
export function s(row: Record<string, unknown> | undefined, key: string): string | null {
  const v = row?.[key]
  return v === null || v === undefined ? null : String(v)
}

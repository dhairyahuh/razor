/**
 * Number, money and date formatting.
 *
 * Everything the interface renders as a number goes through here. The reason is
 * not tidiness: it is that a fraud console is read by comparing figures across
 * panels, and two panels that round differently quietly tell the reader the
 * numbers disagree when they do not.
 *
 * Three rules hold throughout:
 *
 *  - Missing is not zero. `null`, `undefined` and NaN render as an em dash, never
 *    as "0" or "0.0%". The backend genuinely cannot compute some of these (a
 *    vector with no positives has no recall) and saying zero would be a lie.
 *  - Money is Indian-grouped. The scenario is UPI and IMPS in rupees; 12,34,567
 *    is what the domain reads, not 1,234,567.
 *  - Precision is chosen per quantity, not globally. An FPR budget of 0.5% needs
 *    two decimals to be meaningful; a recall of 62% does not.
 */

/** What we render where a number is genuinely unavailable. */
export const MISSING = '—'

function absent(v: number | null | undefined): v is null | undefined {
  return v === null || v === undefined || !Number.isFinite(v)
}

/* -- proportions ----------------------------------------------------------- */

/**
 * A rate held on the backend as a fraction in [0, 1], rendered as a percentage.
 *
 * `digits` defaults to 1. Pass 2 for anything compared against the FPR budget,
 * where the difference between 0.48% and 0.52% is the entire point.
 */
export function pct(v: number | null | undefined, digits = 1): string {
  if (absent(v)) return MISSING
  return `${(v * 100).toFixed(digits)}%`
}

/** A percentage with an explicit sign, for deltas between rounds. */
export function pctDelta(v: number | null | undefined, digits = 1): string {
  if (absent(v)) return MISSING
  const s = (v * 100).toFixed(digits)
  return v > 0 ? `+${s}%` : `${s}%`
}

/**
 * Percentage points, for the difference between two rates. Deliberately a
 * separate function from `pctDelta`: "recall fell 8pp" and "recall fell 8%" are
 * different claims and the backend means the first one.
 */
export function pp(v: number | null | undefined, digits = 1): string {
  if (absent(v)) return MISSING
  const s = (v * 100).toFixed(digits)
  return `${v > 0 ? '+' : ''}${s}pp`
}

/** A Wilson interval rendered as a range, matching the point estimate's precision. */
export function interval(
  lo: number | null | undefined,
  hi: number | null | undefined,
  digits = 1,
): string {
  if (absent(lo) || absent(hi)) return MISSING
  return `${(lo * 100).toFixed(digits)}–${(hi * 100).toFixed(digits)}%`
}

/**
 * A point estimate with its interval, in the one form used everywhere:
 * `74.5% [71.3–77.7]`. Wilson intervals on small vectors are asymmetric, so the
 * `± ` notation the brief offers as an alternative would misstate them - it
 * implies a symmetry that is not there. Having picked this form, nothing else
 * in the interface renders an interval any other way.
 */
export function withInterval(
  v: number | null | undefined,
  lo: number | null | undefined,
  hi: number | null | undefined,
  digits = 1,
): string {
  if (absent(v)) return MISSING
  const point = `${(v * 100).toFixed(digits)}%`
  if (absent(lo) || absent(hi)) return point
  return `${point} [${(lo * 100).toFixed(digits)}–${(hi * 100).toFixed(digits)}]`
}

/* -- money ----------------------------------------------------------------- */

const INR = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
})

const INR_PAISE = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
})

/** Rupees, Indian grouping, no paise. The default for every amount on screen. */
export function inr(v: number | null | undefined): string {
  if (absent(v)) return MISSING
  return INR.format(v)
}

/** Rupees with paise, for per-transaction costs where the rounding matters. */
export function inrExact(v: number | null | undefined): string {
  if (absent(v)) return MISSING
  return INR_PAISE.format(v)
}

/**
 * Rupees abbreviated on the Indian scale for axis ticks and counters, where the
 * full figure would not fit. Lakh and crore rather than K and M, because the
 * reader of this console thinks in lakh and crore.
 */
export function inrShort(v: number | null | undefined): string {
  if (absent(v)) return MISSING
  const sign = v < 0 ? '-' : ''
  const n = Math.abs(v)
  if (n >= 1e7) return `${sign}₹${trim(n / 1e7)} Cr`
  if (n >= 1e5) return `${sign}₹${trim(n / 1e5)} L`
  if (n >= 1e3) return `${sign}₹${trim(n / 1e3)} K`
  return `${sign}₹${Math.round(n)}`
}

/**
 * Any currency, formatted by its own conventions rather than by rupee rules
 * applied to a different symbol.
 *
 * Nothing in this run is denominated in anything but INR - the scenario is UPI
 * and IMPS. This exists because the cost model's assumptions are quoted from
 * published figures that are not all rupee-denominated, and converting them
 * into rupees at an invented rate to make the formatter's life easier would be
 * fabricating a number.
 */
export function money(
  v: number | null | undefined,
  currency: 'INR' | 'USD' | 'EUR' = 'INR',
): string {
  if (absent(v)) return MISSING
  // en-IN groups in lakh and crore; USD and EUR want thousands, and the euro
  // wants the symbol trailing in most of its locales.
  const locale = currency === 'INR' ? 'en-IN' : currency === 'EUR' ? 'de-DE' : 'en-US'
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    maximumFractionDigits: 0,
  }).format(v)
}

function trim(v: number): string {
  // Two significant-ish digits, without a trailing ".0" that adds no information.
  const s = v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)
  return s.replace(/\.?0+$/, '')
}

/* -- counts ---------------------------------------------------------------- */

const COUNT = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 })

/** A count of things. Indian grouping, so it matches the amounts beside it. */
export function count(v: number | null | undefined): string {
  if (absent(v)) return MISSING
  return COUNT.format(v)
}

/** A count abbreviated for a tick label or a tight counter. */
export function countShort(v: number | null | undefined): string {
  if (absent(v)) return MISSING
  const n = Math.abs(v)
  if (n >= 1e7) return `${trim(v / 1e7)}Cr`
  if (n >= 1e5) return `${trim(v / 1e5)}L`
  if (n >= 1e3) return `${trim(v / 1e3)}K`
  return COUNT.format(v)
}

/* -- scores and plain numbers ---------------------------------------------- */

/**
 * A model score in [0, 1]. Three decimals: the operating thresholds this system
 * chooses differ in the third place, so two would collapse distinct cut points
 * into the same displayed number.
 */
export function score(v: number | null | undefined): string {
  if (absent(v)) return MISSING
  return v.toFixed(3)
}

/** A bare number at a chosen precision, for feature values and z-scores. */
export function num(v: number | null | undefined, digits = 2): string {
  if (absent(v)) return MISSING
  return v.toFixed(digits)
}

/**
 * A feature value where the sensible precision depends on the magnitude - graph
 * degrees are integers, velocities are fractional, amounts are large.
 */
export function auto(v: number | null | undefined): string {
  if (absent(v)) return MISSING
  if (Number.isInteger(v)) return COUNT.format(v)
  const n = Math.abs(v)
  if (n >= 1000) return COUNT.format(Math.round(v))
  if (n >= 1) return v.toFixed(2)
  if (n >= 0.01) return v.toFixed(3)
  return v.toExponential(1)
}

/* -- time ------------------------------------------------------------------ */

/**
 * Times render in IST regardless of where the browser is, and say so.
 *
 * The scenario is Indian retail payments and the run was produced in IST; a
 * judge opening this from another timezone should see the same clock the
 * artefacts were written against, not a silent local translation of it. Pinning
 * the zone and printing the label is the only way that is unambiguous.
 */
const ZONE = 'Asia/Kolkata'
export const ZONE_LABEL = 'IST'

/** Wall-clock time for a streaming row. Seconds included; the stream ticks. */
export function clock(iso: string | Date | null | undefined): string {
  const d = toDate(iso)
  if (!d) return MISSING
  return d.toLocaleTimeString('en-IN', {
    timeZone: ZONE,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

/** Date and time with the zone, for provenance strips and run metadata. */
export function stamp(iso: string | Date | null | undefined): string {
  const d = toDate(iso)
  if (!d) return MISSING
  const date = d.toLocaleDateString('en-IN', {
    timeZone: ZONE,
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })
  const time = d.toLocaleTimeString('en-IN', {
    timeZone: ZONE,
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
  return `${date} ${time} ${ZONE_LABEL}`
}

/** A duration in seconds, rendered the way a run log reads. */
export function duration(seconds: number | null | undefined): string {
  if (absent(seconds)) return MISSING
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  if (m < 60) return `${m}m ${s.toString().padStart(2, '0')}s`
  return `${Math.floor(m / 60)}h ${(m % 60).toString().padStart(2, '0')}m`
}

function toDate(v: string | Date | null | undefined): Date | null {
  if (!v) return null
  const d = v instanceof Date ? v : new Date(v)
  return Number.isNaN(d.getTime()) ? null : d
}

/* -- identifiers ----------------------------------------------------------- */

/**
 * Shorten a long id for a narrow column, keeping both ends. Judges cross-check
 * ids against the CSVs, so the head and tail both have to survive - an ellipsis
 * at the end alone makes two different ids look identical.
 */
export function shortId(id: string | null | undefined, head = 8, tail = 4): string {
  if (!id) return MISSING
  if (id.length <= head + tail + 1) return id
  return `${id.slice(0, head)}…${id.slice(-tail)}`
}

/** A git SHA, at the length everyone actually quotes. */
export function sha(v: string | null | undefined): string {
  if (!v) return MISSING
  return v.slice(0, 7)
}

/**
 * Turn a backend column name into a human label without losing the original.
 * Used only where the raw name is also shown; the brief is explicit that the
 * exact backend metric names must appear so judges can grep the CSVs.
 */
export function humanise(column: string): string {
  return column
    .replace(/_/g, ' ')
    .replace(/\b(fpr|auc|pr|roc|ci|inr|upi|imps)\b/gi, (m) => m.toUpperCase())
    .replace(/^./, (c) => c.toUpperCase())
}

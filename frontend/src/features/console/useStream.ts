/**
 * The live payment stream.
 *
 * A cursor walking a preloaded slice of the run's scored test window, emitting
 * rows on an interval. Every row is a real payment the pipeline actually scored
 * - the *arrival* is simulated, the payment and its score are not. That
 * distinction is stated in the panel rather than left for someone to work out,
 * because a stream that looked live but was generating rows in the browser
 * would be the single most damaging thing this interface could do to its own
 * credibility.
 *
 * Implementation notes that are not obvious:
 *
 *  - Rows are appended to a *ref* and flushed to state on a rAF tick, not set
 *    directly. At eight rows a second with a slide-over open, setting state per
 *    row re-renders the whole console eight times a second for no visual gain.
 *  - The buffer is capped. An unbounded list grows without limit over a demo
 *    that is left running, and nobody scrolls back four thousand rows.
 *  - The interval is `setTimeout` rescheduled rather than `setInterval`, so a
 *    backgrounded tab that throttles timers resumes cleanly instead of firing a
 *    burst of catch-up rows the moment it is focused.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { ScoredTxn } from '@/types/api'

/** Rows per second. Fast enough to feel alive, slow enough to read one. */
export const DEFAULT_RATE = 8

/** How many rows stay on screen before the oldest fall off. */
const BUFFER = 300

/**
 * Running totals over everything the stream has emitted.
 *
 * These are the only figures in the application computed in the browser, and
 * they are computed *from rows the backend scored* - a tally of what the viewer
 * has watched go past, not a new measurement. They accumulate over every row
 * emitted rather than over the visible buffer: the buffer is capped at a few
 * hundred rows so the DOM stays cheap, and counters that reset every time a row
 * fell off the bottom would be meaningless.
 *
 * The realised FPR here will not exactly equal the one in
 * `defend_headline.csv`. It is over however many rows have streamed so far
 * rather than the full test window, and the panel says so rather than letting
 * two numbers on the same screen quietly disagree.
 */
export interface Counters {
  scored: number
  alerts: number
  blocks: number
  valueStopped: number
  fraudSeen: number
  fraudStopped: number
  legitimate: number
  falsePositives: number
  /** Null until enough legitimate rows have gone past for a rate to mean anything. */
  realisedFpr: number | null
}

const MIN_FOR_RATE = 200

const ZERO: Counters = {
  scored: 0,
  alerts: 0,
  blocks: 0,
  valueStopped: 0,
  fraudSeen: 0,
  fraudStopped: 0,
  legitimate: 0,
  falsePositives: 0,
  realisedFpr: null,
}

function accumulate(into: Counters, r: ScoredTxn) {
  const blocked = r.intent_block === 1 || r.control_block === 1
  const flagged = blocked || r.model_alert === 1 || r.decision === 1

  into.scored += 1
  if (blocked) into.blocks += 1
  else if (flagged) into.alerts += 1

  if (r.is_fraud === 1) {
    into.fraudSeen += 1
    if (flagged) {
      into.fraudStopped += 1
      into.valueStopped += Number(r.amount) || 0
    }
  } else {
    into.legitimate += 1
    if (flagged) into.falsePositives += 1
  }
}

export interface StreamState {
  rows: ScoredTxn[]
  counters: Counters
  running: boolean
  /** How far through the slice the cursor is, for the progress hint. */
  position: number
  total: number
  toggle: () => void
  reset: () => void
  setRate: (perSecond: number) => void
  rate: number
}

export function useStream(source: ScoredTxn[] | undefined, autoStart = true): StreamState {
  const [rows, setRows] = useState<ScoredTxn[]>([])
  const [counters, setCounters] = useState<Counters>(ZERO)
  const [running, setRunning] = useState(autoStart)
  const [rate, setRate] = useState(DEFAULT_RATE)
  const [position, setPosition] = useState(0)

  const cursor = useRef(0)
  const pending = useRef<ScoredTxn[]>([])
  const tally = useRef<Counters>({ ...ZERO })
  const frame = useRef<number | null>(null)
  const timer = useRef<number | null>(null)

  // Flush whatever the cursor emitted since the last paint. One state write per
  // frame regardless of the rate, so raising the rate costs the browser
  // nothing extra in renders.
  const flush = useCallback(() => {
    frame.current = null
    if (!pending.current.length) return
    const batch = pending.current
    pending.current = []
    setRows((prev) => {
      const next = batch.concat(prev)
      return next.length > BUFFER ? next.slice(0, BUFFER) : next
    })
    const t = tally.current
    setCounters({
      ...t,
      realisedFpr: t.legitimate >= MIN_FOR_RATE ? t.falsePositives / t.legitimate : null,
    })
    setPosition(cursor.current)
  }, [])

  useEffect(() => {
    if (!running || !source || source.length === 0) return

    let cancelled = false

    const tick = () => {
      if (cancelled) return
      // Wrapping rather than stopping: the slice is a loop of real payments and
      // a demo left running should not run out. The counters keep climbing,
      // which is honest - they count rows watched, not distinct payments.
      const row = source[cursor.current % source.length]
      if (row) {
        pending.current.unshift(row)
        accumulate(tally.current, row)
      }
      cursor.current += 1

      if (frame.current === null) frame.current = requestAnimationFrame(flush)
      timer.current = window.setTimeout(tick, 1000 / rate)
    }

    timer.current = window.setTimeout(tick, 1000 / rate)

    return () => {
      cancelled = true
      if (timer.current !== null) clearTimeout(timer.current)
      if (frame.current !== null) cancelAnimationFrame(frame.current)
      frame.current = null
    }
  }, [running, source, rate, flush])

  const reset = useCallback(() => {
    cursor.current = 0
    pending.current = []
    tally.current = { ...ZERO }
    setRows([])
    setCounters(ZERO)
    setPosition(0)
  }, [])

  return {
    rows,
    counters,
    running,
    position,
    total: source?.length ?? 0,
    rate,
    setRate,
    toggle: () => setRunning((r) => !r),
    reset,
  }
}

import { useCallback, useRef, useState } from 'react'
import { api, ApiError } from '@/api/client'
import type { ScoreResponse } from '@/types/api'

/**
 * Measuring what a decision actually costs, rather than asserting it.
 *
 * Nothing in `artifacts/` records scoring latency. The pipeline's stage timings
 * are batch figures over a whole training run and using one as an inline latency
 * would be a category error — so rather than derive a number from something that
 * does not mean it, this fires real payments at the deployed model and reports
 * what came back.
 *
 * Two clocks, because they answer different questions and a single figure hides
 * which one a reader is getting:
 *
 *  - **`latency_ms` from the response** is server-side compute: feature
 *    assembly, the ensemble, the deterministic guards. This is the number that
 *    would have to fit inside an authorisation window.
 *  - **Round-trip measured here** adds HTTP, JSON and the loopback hop. It is
 *    always the larger of the two and is reported separately rather than folded
 *    in, because on a real deployment that transport is a different component's
 *    budget.
 *
 * Requests are sequential on purpose. Firing them in parallel would measure how
 * well the box handles concurrency, which is a fair question and not this one —
 * and the scoring endpoint is rate limited, because an unauthenticated scorer
 * with no limit is a free oracle for mapping the decision boundary.
 */

/** Enough for a stable p50 and an indicative p95, without tripping the rate limit. */
const DEFAULT_SAMPLES = 40

export interface LatencySample {
  server_ms: number
  round_trip_ms: number
}

export interface LatencyResult {
  samples: LatencySample[]
  server: Percentiles
  roundTrip: Percentiles
  /** From the last response: how much of the matrix a bare payment message fills. */
  feature_completeness: number | null
  features_supplied: number | null
  features_expected: number | null
  /** Rate-limited before the full sample was collected. */
  truncated: boolean
}

export interface Percentiles {
  p50: number | null
  p95: number | null
  p99: number | null
  min: number | null
  max: number | null
}

export function useLatencyProbe(samples = DEFAULT_SAMPLES) {
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<LatencyResult | null>(null)
  const [error, setError] = useState<unknown>(null)
  const cancelled = useRef(false)

  const run = useCallback(async () => {
    cancelled.current = false
    setRunning(true)
    setError(null)
    setProgress(0)

    const collected: LatencySample[] = []
    let last: ScoreResponse | null = null
    let truncated = false

    try {
      for (let i = 0; i < samples; i++) {
        if (cancelled.current) break

        // The amount varies so the run is not measuring one repeated path. Every
        // other field is held fixed: the question is the cost of a decision, not
        // the variance across payment shapes.
        const started = performance.now()
        const res = await api.score({
          amount: 4000 + i * 137,
          rail: 'UPI_P2P',
          channel: 'mobile_app',
          features: {},
        })
        const elapsed = performance.now() - started

        last = res
        collected.push({
          server_ms: Number(res.latency_ms),
          round_trip_ms: elapsed,
        })
        setProgress(i + 1)
      }
    } catch (err) {
      // A 429 mid-run is a real answer about the deployed posture, not a failure
      // of the measurement. Keep what was collected and say it was cut short.
      if (err instanceof ApiError && err.status === 429 && collected.length > 0) {
        truncated = true
      } else {
        setError(err)
        setRunning(false)
        return
      }
    }

    if (collected.length > 0) {
      setResult({
        samples: collected,
        server: percentiles(collected.map((s) => s.server_ms)),
        roundTrip: percentiles(collected.map((s) => s.round_trip_ms)),
        feature_completeness: last ? Number(last.feature_completeness) : null,
        features_supplied: last ? Number(last.features_supplied) : null,
        features_expected: last ? Number(last.features_expected) : null,
        truncated,
      })
    }
    setRunning(false)
  }, [samples])

  const cancel = useCallback(() => {
    cancelled.current = true
  }, [])

  return { run, cancel, running, progress, result, error, samples }
}

/**
 * Nearest-rank percentiles.
 *
 * Not interpolated. At forty samples an interpolated p99 is a value between two
 * observations that nothing actually took, and reporting a latency nobody
 * measured is exactly the kind of number this project refuses elsewhere.
 */
function percentiles(values: number[]): Percentiles {
  const clean = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b)
  if (clean.length === 0) return { p50: null, p95: null, p99: null, min: null, max: null }

  const at = (q: number): number | null => {
    const i = Math.min(clean.length - 1, Math.max(0, Math.ceil(q * clean.length) - 1))
    return clean[i] ?? null
  }

  return {
    p50: at(0.5),
    p95: at(0.95),
    p99: at(0.99),
    min: clean[0] ?? null,
    max: clean[clean.length - 1] ?? null,
  }
}

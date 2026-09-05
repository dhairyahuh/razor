import { ApiError } from '@/api/client'
import { pct } from '@/lib/format'
import { useLatencyProbe, type Percentiles } from './useLatencyProbe'

/**
 * What one decision actually costs, measured rather than claimed.
 *
 * The two clocks are kept apart because collapsing them is the easy dishonesty
 * here: server-side compute is the figure that has to fit inside an
 * authorisation window, and round-trip includes an HTTP hop that a real
 * deployment would not pay in the same place. Reporting only the smaller one
 * flatters; reporting only the larger one blames the model for the transport.
 *
 * `feature_completeness` is the point of the panel rather than a footnote. The
 * probe deliberately sends a bare payment message with no derived features, which
 * is the worst case a caller can present — and the resulting completeness figure
 * is the honest answer to "what does this need that a payment message does not
 * carry". A demo that only ever scored fully-populated rows would be hiding the
 * integration work behind the model.
 */
export function LatencyProbe() {
  const { run, running, progress, result, error, samples } = useLatencyProbe()

  const demo = error instanceof ApiError && error.status === 503

  return (
    <div className="flex h-full flex-col gap-2 p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="label">inline decision cost</div>
        <button
          type="button"
          onClick={() => run()}
          disabled={running}
          className="rounded border border-accent-dim bg-accent-wash px-2 py-1 font-display text-2xs font-semibold uppercase tracking-wider text-accent transition-colors hover:bg-accent-dim hover:text-fg-inverse disabled:opacity-50"
        >
          {running ? `${progress}/${samples}…` : 'Measure now'}
        </button>
      </div>

      {error && !demo ? (
        <p className="text-2xs leading-relaxed text-amber">
          The probe did not run.{' '}
          <span className="font-mono">
            {error instanceof Error ? error.message : String(error)}
          </span>
        </p>
      ) : null}

      {demo ? (
        <div className="rounded border border-line bg-raised p-2">
          <p className="text-2xs leading-relaxed text-fg-secondary">
            This panel measures — it scores real payments against a deployed model and times
            them.
          </p>
          <p className="mt-1 text-2xs leading-relaxed text-fg-faint">
            The console is reading a recorded run, so there is no model to score against. No
            latency figure is shown, because none was recorded and inventing one would be the
            only fabricated number on the site. Start the API (
            <span className="font-mono">docker compose up</span>) and this becomes live.
          </p>
        </div>
      ) : null}

      {result ? (
        <div className="space-y-2">
          <Percentile
            label="server compute"
            note="feature assembly, ensemble, guards"
            p={result.server}
            tone="text-fg"
          />
          <Percentile
            label="round trip"
            note="the above plus HTTP and loopback"
            p={result.roundTrip}
            tone="text-fg-secondary"
          />

          <div className="rounded border border-line-subtle bg-raised p-2">
            <div className="label">feature_completeness</div>
            <div className="num text-lg leading-none text-amber">
              {pct(result.feature_completeness, 1)}
            </div>
            <p className="mt-1 text-2xs leading-snug text-fg-faint">
              {result.features_supplied} of {result.features_expected} columns, from a bare
              payment message with no derived features. Most of this model is history — velocity
              windows, counterparty novelty, graph position — so this is the number that says
              what a feature store has to supply before the score means anything.
            </p>
          </div>

          <p className="text-2xs leading-snug text-fg-faint">
            {result.samples.length} sequential samples
            {result.truncated ? ', cut short by the endpoint rate limit — which is the deployed posture working' : ''}
            . Percentiles are nearest-rank, not interpolated.
          </p>
        </div>
      ) : null}

      {!result && !error && !running ? (
        <p className="text-2xs leading-relaxed text-fg-muted">
          Fires {samples} payments at the deployed model, one at a time, and reports what came
          back. Nothing in the run's artefacts records scoring latency, so this is measured here
          or not shown at all.
        </p>
      ) : null}
    </div>
  )
}

function Percentile({
  label,
  note,
  p,
  tone,
}: {
  label: string
  note: string
  p: Percentiles
  tone: string
}) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="flex items-baseline gap-3">
        <Stat v={p.p50} q="p50" tone={tone} big />
        <Stat v={p.p95} q="p95" tone={tone} />
        <Stat v={p.p99} q="p99" tone={tone} />
      </div>
      <div className="mt-0.5 text-2xs text-fg-faint">{note}</div>
    </div>
  )
}

function Stat({
  v,
  q,
  tone,
  big = false,
}: {
  v: number | null
  q: string
  tone: string
  big?: boolean
}) {
  return (
    <div>
      <span className={`num ${big ? 'text-lg' : 'text-sm'} leading-none ${tone}`}>
        {v === null ? '—' : `${v.toFixed(1)}`}
      </span>
      <span className="ml-0.5 text-2xs text-fg-faint">ms</span>
      <div className="text-2xs text-fg-faint">{q}</div>
    </div>
  )
}

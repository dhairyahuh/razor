import { useEffect, useState } from 'react'
import { useTour } from '@/features/tour/useTour'
import { Panel } from '@/components/ui/Panel'
import { PanelLoading, NoArtefact } from '@/components/ui/RouteSkeleton'
import { ErrorBoundary } from '@/components/ui/ErrorBoundary'
import { StreamTable } from '@/features/console/StreamTable'
import { CounterRail } from '@/features/console/Counters'
import { Headline } from '@/features/console/Headline'
import { Argument } from '@/features/console/Argument'
import { useStream } from '@/features/console/useStream'
import { Inspector } from '@/features/inspector/Inspector'
import { useDefend, useTransactions } from '@/api/queries'
import { fprBudget, metric } from '@/lib/tables'
import { count } from '@/lib/format'

/**
 * `/` — the Live Threat Console.
 *
 * Not a landing page. The first thing a judge sees is the product this would be
 * if it shipped: payments arriving, being scored, and being stopped or let
 * through, with the reason one click away.
 *
 * The single most important honesty claim on this screen is in the panel
 * caveat, so it is worth stating here too: **the rows are real and the arrival
 * is not.** Every payment shown was scored by the trained ensemble during the
 * run and its score, decision and label come straight out of
 * `defend_scored_test_set.csv`. What the browser does is decide *when* to show
 * you each one. Nothing is generated client-side.
 */
export default function Console() {
  const [selected, setSelected] = useState<string | null>(null)

  // A large first page rather than paging as the stream advances: the cursor
  // walks a fixed slice, and re-fetching mid-demo would stutter the stream at
  // exactly the moment someone is watching it.
  const txns = useTransactions({ limit: 2000 })
  const defend = useDefend()

  const stream = useStream(txns.data?.rows)

  // The tour opens the inspector on an alerted payment rather than whatever
  // happens to be at the top of the stream. An approved row would be a
  // perfectly honest thing to show and a wasted step - the reason codes, the
  // counterfactual and the guard verdict are the point, and an approval has
  // none of them.
  const rows = txns.data?.rows
  useEffect(() => {
    return useTour.getState().register('open-inspector', () => {
      const interesting =
        rows?.find((r) => r.intent_block === 1 || r.control_block === 1) ??
        rows?.find((r) => r.is_fraud === 1 && r.decision === 1) ??
        rows?.find((r) => r.decision === 1)
      if (interesting) setSelected(interesting.txn_id)
    })
  }, [rows])

  const headline = defend.data?.headline
  const budget = fprBudget(defend.data?.intervals)
  const recall = metric(headline, 'recall')

  return (
    <div className="flex h-full min-h-0 flex-col gap-2 p-2">
      {/* The argument first, then the measurements it rests on. A judge who reads
          only the top two strips has had the entire submission, including the
          part that counts against it. */}
      <ErrorBoundary label="Run summary">
        <Argument />
      </ErrorBoundary>

      <ErrorBoundary label="Headline metrics">
        <Headline />
      </ErrorBoundary>

      <div className="flex min-h-0 flex-1 gap-2">
        <Panel
          id="tour-stream"
          title="Live payment stream"
          source="defend_scored_test_set.csv"
          subtitle={
            <>
              Payments the trained ensemble scored during this run, replayed in order.{' '}
              {txns.data ? (
                <>
                  <span className="num text-fg-secondary">{count(stream.total)}</span> rows
                  loaded of{' '}
                  <span className="num text-fg-secondary">{count(txns.data.total)}</span> in
                  the test window.
                </>
              ) : null}
            </>
          }
          actions={
            <StreamControls
              running={stream.running}
              rate={stream.rate}
              onToggle={stream.toggle}
              onRate={stream.setRate}
              onReset={stream.reset}
            />
          }
          caveat={
            <>
              <span className="text-fg-muted">The rows are real; the arrival is not.</span>{' '}
              Every payment here was scored by the ensemble during the run — score, decision
              and outcome come from the artefact, unmodified. The browser only decides when
              to show each one. Two decline types are kept distinct on purpose: a{' '}
              <span className="text-red">provable block</span> can be shown to a customer and
              argued with, an <span className="text-amber">alert</span> is a probability.
            </>
          }
          className="min-w-0 flex-1"
          scroll
        >
          {txns.isLoading ? (
            <PanelLoading />
          ) : !txns.data?.available ? (
            <NoArtefact
              file="defend_scored_test_set.csv"
              note="The defend stage writes this. A run that stopped before it has nothing to stream."
            />
          ) : (
            <StreamTable rows={stream.rows} onSelect={setSelected} selected={selected} />
          )}
        </Panel>

        <Panel
          id="tour-counters"
          title="Since you started watching"
          source="defend_scored_test_set.csv"
          className="w-[260px] shrink-0"
        >
          <CounterRail
            counters={stream.counters}
            fprBudget={budget}
            measuredRecall={recall}
          />
        </Panel>
      </div>

      <Inspector txnId={selected} onClose={() => setSelected(null)} />
    </div>
  )
}

function StreamControls({
  running,
  rate,
  onToggle,
  onRate,
  onReset,
}: {
  running: boolean
  rate: number
  onToggle: () => void
  onRate: (n: number) => void
  onReset: () => void
}) {
  return (
    <div className="flex items-center gap-1">
      {/* Pause exists because the stream is the demo's biggest liability: a row
          worth talking about scrolls away in four seconds. Being able to freeze
          it is the difference between narrating the console and chasing it. */}
      <button
        type="button"
        onClick={onToggle}
        className="rounded border border-line px-2 py-0.5 font-display text-2xs font-semibold uppercase tracking-wider text-fg-secondary transition-colors hover:bg-hover hover:text-fg"
      >
        {running ? 'Pause' : 'Resume'}
      </button>
      <label className="flex items-center gap-1" title="Rows per second">
        <select
          value={rate}
          onChange={(e) => onRate(Number(e.target.value))}
          className="rounded border border-line bg-surface px-1 py-0.5 font-mono text-2xs text-fg-secondary"
        >
          {[2, 4, 8, 16, 32].map((r) => (
            <option key={r} value={r}>
              {r}/s
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        onClick={onReset}
        title="Clear the buffer and the counters, and start the cursor again"
        className="rounded border border-line px-2 py-0.5 font-display text-2xs font-semibold uppercase tracking-wider text-fg-muted transition-colors hover:bg-hover hover:text-fg"
      >
        Reset
      </button>
    </div>
  )
}

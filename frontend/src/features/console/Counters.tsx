import type { Counters } from './useStream'
import { count, inr, pct } from '@/lib/format'

/**
 * The right rail: what has gone past since the stream started.
 *
 * Two decisions worth naming.
 *
 * Value stopped gets the money colour rather than the success colour. It is a
 * different axis from severity - a large number here is good news about the
 * defence and bad news about the traffic - and colouring it green would let it
 * read as "everything is fine", which is not what it says.
 *
 * The realised false-positive rate is shown against the budget the model was
 * calibrated to, because on its own the number means nothing. Over the budget
 * is amber, not red: the budget is a target the threshold was chosen to hit
 * over the full window, and a few hundred rows will scatter around it. Treating
 * ordinary sampling noise as a failure would be crying wolf.
 */

interface Props {
  counters: Counters
  /** From `defend_headline.csv`, so the comparison is against a measured target. */
  fprBudget: number | null
  /** Recall over the full test window, for the same reason. */
  measuredRecall: number | null
}

export function CounterRail({ counters, fprBudget, measuredRecall }: Props) {
  const c = counters
  const over = fprBudget !== null && c.realisedFpr !== null && c.realisedFpr > fprBudget

  return (
    <div className="flex h-full flex-col gap-px overflow-y-auto">
      <Counter
        label="transactions scored"
        value={count(c.scored)}
        note="rows replayed from the run's test window"
      />

      <Counter
        label="alerts raised"
        value={count(c.alerts + c.blocks)}
        tone="text-amber"
        note={
          <>
            <span className="text-amber">{count(c.alerts)}</span> by score,{' '}
            <span className="text-red">{count(c.blocks)}</span> by a provable control
          </>
        }
      />

      <Counter
        label="value at risk stopped"
        value={inr(c.valueStopped)}
        tone="text-money"
        note={`across ${count(c.fraudStopped)} fraudulent payments the system flagged`}
      />

      <Counter
        label="fraud seen / stopped"
        value={`${count(c.fraudSeen)} / ${count(c.fraudStopped)}`}
        note={
          measuredRecall !== null ? (
            <>
              full-window recall is{' '}
              <span className="text-fg-secondary">{pct(measuredRecall)}</span> —
              defend_headline.csv
            </>
          ) : (
            'this run reported no headline recall'
          )
        }
      />

      <Counter
        label="realised FPR"
        value={c.realisedFpr === null ? '—' : pct(c.realisedFpr, 2)}
        tone={over ? 'text-amber' : 'text-fg'}
        note={
          c.realisedFpr === null ? (
            'needs a few hundred legitimate rows before this means anything'
          ) : fprBudget === null ? (
            'this run reported no FPR budget to compare against'
          ) : (
            <>
              budget <span className="text-fg-secondary">{pct(fprBudget, 2)}</span>
              {over ? ' — above it on the rows so far, which a short window will do' : ''}
            </>
          )
        }
      />

      <div className="mt-auto border-t border-line-subtle px-3 py-2">
        <p className="text-2xs leading-relaxed text-fg-faint">
          Counters tally rows as they stream. They are running totals of what you have
          watched, not a re-measurement — the full-window figures are on{' '}
          <span className="font-mono">/defend</span>.
        </p>
      </div>
    </div>
  )
}

function Counter({
  label,
  value,
  note,
  tone = 'text-fg',
}: {
  label: string
  value: string
  note?: React.ReactNode
  tone?: string
}) {
  return (
    <div className="border-b border-line-subtle px-3 py-2">
      <div className="label">{label}</div>
      <div className={`num mt-0.5 text-xl font-medium leading-none ${tone}`}>{value}</div>
      {note ? <div className="mt-1 text-2xs leading-snug text-fg-faint">{note}</div> : null}
    </div>
  )
}

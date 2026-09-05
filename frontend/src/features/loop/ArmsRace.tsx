import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { RoundRow, Table } from '@/types/api'
import { NoArtefact, NoRows } from '@/components/ui/RouteSkeleton'
import { pct, pp } from '@/lib/format'

/**
 * The arms race: three recall lines across the rounds, with blue's move marked
 * where it was made.
 *
 * The reason this is three lines and not one is the entire argument of the
 * route. A single "recall over time" line would show a system that wobbles and
 * recovers, which is unremarkable. Three lines show the *mechanism*: recall
 * before red moved, recall after red's best genome landed, and recall after
 * blue retrained in response. The vertical gap between the first two is what
 * the attack cost; the gap between the second and third is what the defence
 * won back. Those two quantities are the loop.
 *
 * Targeted recall is drawn rather than overall recall by default because the
 * loop attacks a handful of campaigns out of forty-six, and overall recall
 * dilutes a large effect on the targeted vectors into a small effect on the
 * whole window. Both are available; the toggle says which is which and the
 * caption says why they differ.
 */

export function ArmsRace({ rounds }: { rounds: Table<RoundRow> | undefined }) {
  if (!rounds || !rounds.available) {
    return (
      <NoArtefact
        file="loop_rounds.csv"
        note="The co-evolution stage writes this. A run started with --no-loop has no rounds."
      />
    )
  }
  if (rounds.rows.length === 0) {
    return (
      <NoRows
        file="loop_rounds.csv"
        note="The loop ran but no campaign was large enough to search. Raising the run size or lowering loop.min_fraud_rows gives it something to work with."
      />
    )
  }

  const data = rounds.rows.map((r) => ({
    round: r.round,
    before: r.recall_targeted_before,
    under: r.recall_targeted_under_attack,
    after: r.recall_targeted_after_retrain,
    lost: r.recall_lost_to_red,
    recovered: r.recall_recovered_by_blue,
    move: r.blue_move,
    detail: r.blue_move_detail,
  }))

  // Where blue did something other than absorb the loss. Marked on the chart
  // because "blue held" and "blue shipped a rule" are the two outcomes of a
  // round and the difference is invisible in the lines alone.
  const acted = data.filter((d) => d.move && d.move !== 'hold')

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
            <CartesianGrid strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="round"
              tickLine={false}
              axisLine={{ stroke: 'var(--border)' }}
              label={{ value: 'round', position: 'insideBottomRight', offset: -2 }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              tickLine={false}
              axisLine={false}
              width={38}
            />
            <Tooltip content={<RoundTooltip />} />

            <Line
              type="monotone"
              dataKey="before"
              name="before red moved"
              stroke="var(--fg-muted)"
              strokeWidth="var(--stroke-thin)"
              strokeDasharray="3 3"
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="under"
              name="after red's genome"
              stroke="var(--red)"
              strokeWidth="var(--stroke)"
              dot={{ r: 2.5, fill: 'var(--red)', strokeWidth: 0 }}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="after"
              name="after blue retrained"
              stroke="var(--accent)"
              strokeWidth="var(--stroke)"
              dot={{ r: 2.5, fill: 'var(--accent)', strokeWidth: 0 }}
              isAnimationActive={false}
            />

            {acted.map((d) => (
              <ReferenceDot
                key={d.round}
                x={d.round}
                y={d.after ?? 0}
                r={6}
                fill="none"
                stroke="var(--accent-strong)"
                strokeWidth="var(--stroke-thin)"
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex shrink-0 flex-wrap items-center gap-x-4 gap-y-1 border-t border-line-subtle px-3 py-1.5">
        <Key colour="var(--fg-muted)" dashed label="recall_targeted_before" />
        <Key colour="var(--red)" label="recall_targeted_under_attack" />
        <Key colour="var(--accent)" label="recall_targeted_after_retrain" />
        <span className="ml-auto text-2xs text-fg-faint">
          ringed points are rounds where blue shipped a countermeasure rather than holding
        </span>
      </div>
    </div>
  )
}

function Key({ colour, label, dashed }: { colour: string; label: string; dashed?: boolean }) {
  return (
    <span className="flex items-center gap-1.5">
      <svg width="16" height="4" aria-hidden="true">
        <line
          x1="0"
          y1="2"
          x2="16"
          y2="2"
          stroke={colour}
          strokeWidth="2"
          strokeDasharray={dashed ? '3 3' : undefined}
        />
      </svg>
      <span className="font-mono text-2xs text-fg-muted">{label}</span>
    </span>
  )
}

interface TooltipPayload {
  payload: {
    round: number
    before: number | null
    under: number | null
    after: number | null
    lost: number | null
    recovered: number | null
    move: string | null
    detail: string | null
  }
}

function RoundTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  const d = payload?.[0]?.payload
  if (!active || !d) return null

  return (
    <div className="max-w-72 rounded border border-line bg-overlay p-2 shadow-overlay">
      <div className="label mb-1">round {d.round}</div>
      <dl className="space-y-0.5">
        <Line2 label="before red moved" value={pct(d.before)} />
        <Line2 label="after red's genome" value={pct(d.under)} tone="text-red" />
        <Line2 label="after blue retrained" value={pct(d.after)} tone="text-accent" />
      </dl>
      <div className="mt-1.5 border-t border-line-subtle pt-1.5 text-2xs">
        <p>
          <span className="text-fg-faint">red took </span>
          <span className="num text-red">{pp(-(d.lost ?? 0))}</span>
          <span className="text-fg-faint">, blue won back </span>
          <span className="num text-accent">{pp(d.recovered ?? 0)}</span>
        </p>
        {d.move ? (
          <p className="mt-1 text-fg-muted">
            <span className="font-mono">{d.move}</span>
            {d.detail ? <span className="text-fg-faint"> — {d.detail}</span> : null}
          </p>
        ) : null}
      </div>
    </div>
  )
}

function Line2({ label, value, tone = 'text-fg-secondary' }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 text-2xs">
      <dt className="text-fg-faint">{label}</dt>
      <dd className={`num ${tone}`}>{value}</dd>
    </div>
  )
}

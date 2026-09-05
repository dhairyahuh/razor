import {
  Area,
  ComposedChart,
  CartesianGrid,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { DEPLOYED_BAND } from '@/lib/fidelity'
import { Panel } from '@/components/ui/Panel'
import { DataTable } from '@/components/ui/DataTable'
import { PanelLoading } from '@/components/ui/RouteSkeleton'
import { useDefend } from '@/api/queries'
import { ci, fprBudget, metric } from '@/lib/tables'
import { inr, inrShort, pct, withInterval } from '@/lib/format'
import type { CostCurveRow, PerVectorRow, Table } from '@/types/api'

/**
 * `/defend` — what the defence is actually worth.
 *
 * Four questions, in the order a sceptic asks them. Where does it sit on the
 * trade-off? How confident is it, per vector, given how few rows some vectors
 * have? Is it better than something trivial? And who pays for the false
 * positives?
 *
 * The confidence intervals are not decoration. Several vectors in this taxonomy
 * have single-digit row counts in the test window, and a point estimate of
 * "71.4% recall" on seven rows is a number that means almost nothing. Showing
 * the Wilson interval beside it - and greying the ones the pipeline flagged as
 * insufficient - is the difference between a report and a claim.
 */
export default function Defend() {
  const { data, isLoading } = useDefend()

  const budget = fprBudget(data?.intervals)
  const recall = ci(data?.intervals, 'recall_at_0.005_fpr')
  const threshold = metric(data?.headline, 'threshold')

  return (
    <div className="grid h-full min-h-0 grid-cols-2 grid-rows-2 gap-2 overflow-auto p-2">
      <Panel
        id="tour-operating"
        title="Operating curve"
        source="defend_operating_curve.csv"
        subtitle={
          budget !== null ? (
            <>
              Recall achievable at each false-positive budget. The deployed point is{' '}
              <span className="num text-fg-secondary">{pct(budget, 2)}</span> FPR, threshold{' '}
              <span className="num text-fg-secondary">{threshold?.toFixed(3) ?? '—'}</span>.
            </>
          ) : (
            'Recall achievable at each false-positive budget.'
          )
        }
        caveat={
          <>
            Recall at the deployed budget is{' '}
            <span className="num text-fg-secondary">
              {withInterval(recall.point, recall.lo, recall.hi)}
            </span>{' '}
            — a bootstrap interval over {recall.n?.toLocaleString('en-IN') ?? '—'} resamples, not
            a point estimate dressed up as certainty. The shaded band is where deployed
            card-fraud systems sit at this budget ({pct(DEPLOYED_BAND[0], 0)}–
            {pct(DEPLOYED_BAND[1], 0)}); landing above it on synthetic data is a reason to check
            the generator, which is what <span className="font-mono">/generate</span> does.
          </>
        }
      >
        {isLoading ? <PanelLoading /> : <OperatingCurve table={data?.operating_curve} budget={budget} />}
      </Panel>

      <Panel
        id="tour-pervector"
        title="Recall by attack vector"
        source="defend_per_vector_recall.csv"
        subtitle="Every simulated vector, with a Wilson interval on its recall."
        caveat="Rows with n_sufficient = 0 have too few positives in the test window for their recall to mean much. They are shown rather than hidden, because which vectors we cannot measure is itself a finding."
        scroll
      >
        {isLoading ? <PanelLoading /> : <PerVector table={data?.per_vector} />}
      </Panel>

      <Panel
        title="Against simpler detectors"
        source="defend_baselines.csv"
        subtitle="The ensemble beside a single rule, an expert rule set and logistic regression."
        caveat={
          <>
            All four are measured on the same rows at the same false-positive budget where
            they can reach it. <span className="font-mono">comparable_budget = 0</span> marks a
            detector too coarse to hit the budget at all — its realised FPR is shown, and its
            recall should be read against that rather than against the others.
          </>
        }
        scroll
      >
        {isLoading ? (
          <PanelLoading />
        ) : (
          <DataTable
            table={data?.baselines}
            columns={[
              { key: 'model', align: 'left', width: '30%' },
              { key: 'recall', help: 'Share of fraud caught' },
              { key: 'realised_fpr', help: 'False-positive rate actually achieved' },
              { key: 'precision' },
              { key: 'comparable_budget', help: '1 when the detector could reach the target FPR' },
              { key: 'note', align: 'left' },
            ]}
          />
        )}
      </Panel>

      <Panel
        title="Who pays, and what it costs"
        source={['defend_fairness.csv', 'defend_cost_summary.csv']}
        subtitle="False-positive burden by cohort, and the operating point in rupees."
        caveat="Every rupee figure rests on cost assumptions stated in defend_cost_summary.csv — reimbursement share, review cost, false-decline cost. They are assumptions, not measurements, and the ranking of thresholds is more trustworthy than the absolute totals."
        scroll
      >
        {isLoading ? <PanelLoading /> : <CostAndFairness data={data} />}
      </Panel>
    </div>
  )
}

function OperatingCurve({
  table,
  budget,
}: {
  table: Table<Record<string, unknown>> | undefined
  budget: number | null
}) {
  if (!table?.available || table.rows.length === 0) {
    return <DataTable table={table} />
  }

  const data = table.rows.map((r) => ({
    fpr: Number(r.target_fpr),
    recall: Number(r.recall),
    precision: Number(r.precision),
    value_recall: Number(r.value_recall),
  }))

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 10, right: 14, bottom: 18, left: 4 }}>
        <CartesianGrid strokeDasharray="2 4" vertical={false} />

        {/* Where deployed card-fraud systems actually sit at this budget, per
            `redteam/generate/fidelity.py`. Drawn because a recall figure with no
            external scale reads as either miraculous or mediocre depending on
            what the reader last saw, and neither reaction is informative. */}
        <ReferenceArea
          y1={DEPLOYED_BAND[0]}
          y2={DEPLOYED_BAND[1]}
          fill="var(--green-wash)"
          fillOpacity={1}
          stroke="none"
          label={{
            value: 'deployed systems',
            fill: 'var(--fg-faint)',
            fontSize: 9,
            position: 'insideTopLeft',
          }}
        />
        <XAxis
          dataKey="fpr"
          type="number"
          scale="log"
          domain={['dataMin', 'dataMax']}
          tickFormatter={(v: number) => `${(v * 100).toFixed(v < 0.01 ? 2 : 1)}%`}
          tickLine={false}
          label={{ value: 'target_fpr (log)', position: 'insideBottom', offset: -12 }}
        />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          tickLine={false}
          axisLine={false}
          width={38}
        />
        <Tooltip
          contentStyle={{
            background: 'var(--bg-overlay)',
            border: '1px solid var(--border)',
            borderRadius: 4,
            fontSize: 11,
          }}
          labelFormatter={(v: number) => `target_fpr ${pct(v, 3)}`}
          formatter={(v: number, name: string) => [pct(v), name]}
        />
        <Area
          dataKey="value_recall"
          name="value_recall"
          stroke="var(--money)"
          fill="var(--money-wash)"
          strokeWidth="var(--stroke-thin)"
          isAnimationActive={false}
        />
        <Line
          dataKey="recall"
          name="recall"
          stroke="var(--accent)"
          strokeWidth="var(--stroke)"
          dot={false}
          isAnimationActive={false}
        />
        <Line
          dataKey="precision"
          name="precision"
          stroke="var(--fg-muted)"
          strokeWidth="var(--stroke-thin)"
          strokeDasharray="3 3"
          dot={false}
          isAnimationActive={false}
        />
        {budget !== null ? (
          <ReferenceLine
            x={budget}
            stroke="var(--amber)"
            strokeDasharray="4 3"
            label={{ value: 'deployed', fill: 'var(--amber)', fontSize: 10, position: 'top' }}
          />
        ) : null}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

function PerVector({ table }: { table: Table<PerVectorRow> | undefined }) {
  return (
    <DataTable
      table={table as Table<Record<string, unknown>> | undefined}
      sortBy="rows"
      columns={[
        { key: 'attack_vector_id', align: 'left', width: '34%' },
        { key: 'rows', help: 'Fraudulent payments for this vector in the test window' },
        {
          key: 'recall',
          help: 'Point estimate with its Wilson 95% interval',
          format: (_v, row) => (
            <span className={row.n_sufficient ? '' : 'text-fg-faint'}>
              {withInterval(
                row.recall as number | null,
                row.recall_lo95 as number | null,
                row.recall_hi95 as number | null,
              )}
            </span>
          ),
        },
        { key: 'value_recall', help: 'Share of rupees at risk that was caught' },
        {
          key: 'value',
          help: 'Total value at risk for this vector',
          format: (v) => inrShort(v as number),
        },
        {
          key: 'n_sufficient',
          help: 'Whether there are enough rows for the recall figure to be interpretable',
          format: (v) => (v ? <span className="text-fg-faint">yes</span> : <span className="text-amber">no</span>),
        },
      ]}
    />
  )
}

function CostAndFairness({ data }: { data: ReturnType<typeof useDefend>['data'] }) {
  const summary = data?.cost_summary
  const optimal = (data?.cost_curve.rows ?? []).find((r) => r.is_optimal === 1) as
    | CostCurveRow
    | undefined
  const deployed = (data?.cost_curve.rows ?? []).find((r) => r.is_deployed === 1) as
    | CostCurveRow
    | undefined

  return (
    <div className="space-y-3 p-3">
      {summary?.available ? (
        <div className="flex flex-wrap gap-4">
          {summary.rows.slice(0, 4).map((r) => (
            <div key={String(r.metric)} title={String(r.note ?? '')}>
              <div className="label">{String(r.metric)}</div>
              <div className="num text-lg leading-none text-money">{inr(r.value)}</div>
            </div>
          ))}
        </div>
      ) : null}

      {deployed && optimal ? (
        <p className="text-2xs leading-relaxed text-fg-muted">
          The deployed threshold costs{' '}
          <span className="num text-fg-secondary">{inr(deployed.total_cost)}</span>; the
          loss-minimising one is at{' '}
          <span className="num text-fg-secondary">{optimal.threshold?.toFixed(3)}</span> and
          costs <span className="num text-fg-secondary">{inr(optimal.total_cost)}</span>. The
          gap is what calibrating to a false-positive budget rather than to expected loss buys
          in customer friction and gives up in rupees.
        </p>
      ) : null}

      <div>
        <div className="label mb-1">false-positive burden by cohort</div>
        <DataTable
          table={data?.fairness}
          columns={[
            { key: 'dimension', align: 'left' },
            { key: 'segment', align: 'left' },
            { key: 'legitimate_rows' },
            {
              key: 'fp_rate',
              help: 'Share of this cohort’s legitimate payments that were stopped',
              format: (_v, row) =>
                withInterval(
                  row.fp_rate as number | null,
                  row.fp_rate_lo95 as number | null,
                  row.fp_rate_hi95 as number | null,
                  2,
                ),
            },
            { key: 'victimisation_rate', help: 'How often this cohort is actually targeted' },
            { key: 'recall', help: 'Share of fraud against this cohort that was caught' },
          ]}
          maxRows={20}
        />
      </div>
    </div>
  )
}

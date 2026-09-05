import { Panel } from '@/components/ui/Panel'
import { DataTable } from '@/components/ui/DataTable'
import { PanelLoading } from '@/components/ui/RouteSkeleton'
import { AuthFlow } from '@/features/deploy/AuthFlow'
import { LatencyProbe } from '@/features/deploy/LatencyProbe'
import { useDefend, useSummary } from '@/api/queries'
import { metric, n as num, s as str } from '@/lib/tables'
import { count, duration, inr, inrShort, pct, stamp } from '@/lib/format'

/**
 * `/deploy` — could this run in a live payment system?
 *
 * The other five routes answer whether the detector is any good. This one
 * answers the question a payments audience asks next and which nothing else on
 * the site addresses: where does it sit, what does a decision cost, what happens
 * when it is wrong, and how do you change it without breaking the thing that is
 * already running.
 *
 * Every figure here is either measured by the pipeline, measured live by the
 * probe in the first panel, or an explicitly-labelled assumption from the cost
 * model. There is no capacity planning and no throughput extrapolation, because
 * one laptop's batch timings do not project onto a payments estate and pretending
 * otherwise would be the least defensible number on the site.
 */
export default function Deploy() {
  const defend = useDefend()
  const summary = useSummary()

  return (
    <div className="grid h-full min-h-0 grid-cols-2 grid-rows-[auto_1fr_1fr] gap-2 overflow-auto p-2">
      <Panel
        id="tour-deploy"
        title="The inline decision path"
        source={null}
        subtitle="Two of the three layers sit inside the authorisation window. Everything that trains does not."
        caveat={
          <>
            The diagram is architecture, not measurement — it is the only thing on this site
            that is. The numbers beside it are measured live against the deployed model when
            one is reachable, and the panel says so rather than showing a figure no run
            produced.
          </>
        }
        className="col-span-2"
      >
        <div className="grid grid-cols-[1fr_320px] gap-2">
          <AuthFlow />
          <div className="border-l border-line-subtle">
            <LatencyProbe />
          </div>
        </div>
      </Panel>

      <Panel
        title="Provable blocks, and what they cost legitimate traffic"
        source={['defend_intent_coverage.csv', 'defend_guard_layers.csv']}
        subtitle="A decline a bank can defend to a customer, against one it can only score."
        caveat={
          <>
            The <span className="font-mono">(legitimate)</span> row is the one that matters for
            deployment. A deterministic block requires a presented artefact to contradict the
            settlement request — legitimate traffic cannot do that, so the control has no false
            declines by construction rather than by tuning.
          </>
        }
        scroll
      >
        {defend.isLoading ? <PanelLoading /> : <Provable defend={defend.data} />}
      </Panel>

      <Panel
        title="What the operating point costs"
        source={['defend_cost_summary.csv', 'defend_headline.csv']}
        subtitle="The review queue this creates, and the rupees it moves."
        caveat="Every rupee figure rests on the three assumptions at the bottom of this panel, stated in the artefact and taken from published figures. They are inputs, not measurements: the ranking of thresholds survives changing them, the absolute totals do not."
        scroll
      >
        {defend.isLoading ? <PanelLoading /> : <Operating defend={defend.data} />}
      </Panel>

      <Panel
        title="Delegated authority: capping the loss"
        source="defend_scoped_token_loss_bound.csv"
        subtitle="What a spend ceiling is worth when an agent is already compromised."
        caveat="This is a loss bound, not a catch rate. The ceiling does not stop a compromised agent and reporting it as detection would be a category error — it caps what the compromise is worth while it runs."
        scroll
      >
        {defend.isLoading ? <PanelLoading /> : <LossBound defend={defend.data} />}
      </Panel>

      <Panel
        title="Change control and retraining"
        source={['defend_split.csv', 'run_summary.json']}
        subtitle="What the model was fitted on, and what changing it costs."
        caveat="Retraining is priced in the loop's cost model rather than assumed free — on one recorded round blue re-thresholded instead of retraining because the retrain cost more than the recall it recovered. A defence that always retrains has never had to justify a model change to a risk committee."
        scroll
      >
        {summary.isLoading ? <PanelLoading /> : <ChangeControl summary={summary.data} />}
      </Panel>
    </div>
  )
}

/* -- provable blocks -------------------------------------------------------- */

function Provable({ defend }: { defend: ReturnType<typeof useDefend>['data'] }) {
  const rows = defend?.intent_coverage.rows ?? []
  const legit = rows.find((r) => str(r, 'attack_vector_id') === '(legitimate)')

  const legitRows = num(legit, 'rows')
  const legitBlocked = num(legit, 'block_rate')
  const legitStepUp = num(legit, 'step_up_rate')

  return (
    <div className="space-y-3 p-3">
      {legit ? (
        <div className="flex flex-wrap gap-5">
          <Figure
            label="false declines on legitimate agentic traffic"
            value={pct(legitBlocked, 2)}
            tone={legitBlocked === 0 ? 'text-green' : 'text-amber'}
            note={`over ${count(legitRows)} legitimate payments carrying an intent artefact`}
          />
          <Figure
            label="step_up_rate"
            value={pct(legitStepUp, 2)}
            note="friction, not a decline — a missing artefact describes every pre-protocol integration too"
          />
        </div>
      ) : null}

      <div>
        <div className="label mb-1">which layer stopped each vector</div>
        <DataTable
          table={defend?.guards}
          sortBy="rows"
          columns={[
            { key: 'attack_vector_id', align: 'left', width: '32%' },
            { key: 'rows' },
            { key: 'intent_guard', help: 'Share stopped by the cryptographic intent guard alone' },
            { key: 'injection_guard', help: 'Share the content guard flagged' },
            { key: 'tabular_model', help: 'Share the ensemble alerted on' },
            {
              key: 'model_only',
              help: 'Share where the model was the only thing that stopped it — no deterministic control could prove a case',
            },
          ]}
          maxRows={20}
        />
      </div>

      <div>
        <div className="label mb-1">intent guard, by vector</div>
        <DataTable
          table={defend?.intent_coverage}
          sortBy="block_rate"
          columns={[
            { key: 'attack_vector_id', align: 'left', width: '32%' },
            { key: 'rows' },
            { key: 'block_rate', help: 'Declined on presented evidence' },
            { key: 'step_up_rate', help: 'Raised friction rather than declining' },
            { key: 'top_reason', align: 'left', help: 'The reason code an investigator would see' },
          ]}
          maxRows={12}
        />
      </div>
    </div>
  )
}

/* -- operating cost --------------------------------------------------------- */

function Operating({ defend }: { defend: ReturnType<typeof useDefend>['data'] }) {
  const rows = defend?.cost_summary.rows ?? []
  const find = (m: string) => rows.find((r) => String(r.metric) === m)

  const alerts = metric(defend?.headline, 'alerts_per_10k')
  const doNothing = find('do_nothing_cost')
  const deployed = find('cost_at_deployed_threshold')
  const net = find('net_benefit_at_deployed')
  const budgetCost = find('cost_of_budget_constraint')

  const assumptions = rows.filter((r) => String(r.note ?? '').startsWith('assumption'))

  return (
    <div className="space-y-3 p-3">
      <div className="flex flex-wrap gap-5">
        <Figure
          label="alerts_per_10k"
          value={count(alerts !== null ? Math.round(alerts) : null)}
          note="what the review queue actually receives"
        />
        <Figure
          label="net_benefit_at_deployed"
          value={inrShort(net?.value as number | null)}
          tone="text-money"
          note="against approving everything"
        />
        <Figure
          label="cost_of_budget_constraint"
          value={inrShort(budgetCost?.value as number | null)}
          note="what a fixed alert budget costs versus minimising loss"
        />
      </div>

      <div className="space-y-1 text-2xs leading-relaxed text-fg-muted">
        <p>
          Approving everything costs{' '}
          <span className="num text-fg-secondary">{inr(doNothing?.value as number | null)}</span>{' '}
          at the reimbursement share below. The deployed threshold costs{' '}
          <span className="num text-fg-secondary">{inr(deployed?.value as number | null)}</span>.
        </p>
        <p>
          The gap between the deployed and the loss-minimising threshold is the price of running
          to a review budget an operations team can actually staff, rather than to whatever
          alert volume minimises modelled loss.
        </p>
      </div>

      {assumptions.length ? (
        <div className="rounded border border-line-subtle bg-raised p-2">
          <div className="label mb-1">stated assumptions, not measurements</div>
          <dl className="space-y-0.5 text-2xs">
            {assumptions.map((a) => (
              <div key={String(a.metric)} className="flex items-baseline justify-between gap-3">
                <dt className="font-mono text-fg-faint">{String(a.metric)}</dt>
                <dd className="num text-fg-secondary" title={String(a.note ?? '')}>
                  {String(a.metric) === 'reimbursement_share'
                    ? pct(a.value as number)
                    : inr(a.value as number | null)}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}
    </div>
  )
}

/* -- loss bound ------------------------------------------------------------- */

function LossBound({ defend }: { defend: ReturnType<typeof useDefend>['data'] }) {
  const row = defend?.ceiling_bound.rows?.[0]

  if (!row) {
    return (
      <p className="p-3 text-2xs text-fg-muted">
        This run produced no <span className="font-mono">defend_scoped_token_loss_bound.csv</span>.
      </p>
    )
  }

  const reduction = num(row, 'loss_reduction_pct')
  const capped = num(row, 'legit_payments_capped_pct')

  return (
    <div className="space-y-3 p-3">
      <div className="flex flex-wrap gap-5">
        <Figure
          label="loss_reduction_pct"
          value={reduction !== null ? `${reduction.toFixed(1)}%` : '—'}
          tone="text-money"
          note="of agentic fraud value, capped rather than caught"
        />
        <Figure
          label="legit_payments_capped_pct"
          value={capped !== null ? `${capped.toFixed(2)}%` : '—'}
          tone={capped !== null && capped < 2 ? 'text-green' : 'text-amber'}
          note="legitimate payments that would hit the ceiling"
        />
        <Figure
          label="ceiling"
          value={inrShort(num(row, 'ceiling'))}
          note={`the ${pct(num(row, 'ceiling_quantile'), 0)} quantile of agentic spend`}
        />
      </div>

      <p className="text-2xs leading-relaxed text-fg-muted">
        A scoped token with a spend ceiling turns an unbounded standing delegation into a
        bounded one. Across{' '}
        <span className="num text-fg-secondary">{count(num(row, 'agentic_fraud_attempts'))}</span>{' '}
        agentic fraud attempts it moves the exposure from{' '}
        <span className="num text-fg-secondary">{inrShort(num(row, 'loss_standing_delegation'))}</span>{' '}
        to <span className="num text-fg-secondary">{inrShort(num(row, 'loss_scoped_token'))}</span>{' '}
        — a control that needs no model, no training data and no threshold, which is what makes
        it the first thing to ship.
      </p>
    </div>
  )
}

/* -- change control --------------------------------------------------------- */

function ChangeControl({ summary }: { summary: ReturnType<typeof useSummary>['data'] }) {
  const rows = summary?.split.rows ?? []
  const p = summary?.provenance
  const timings = (summary?.summary?.stage_timings_seconds ?? {}) as Record<string, number>

  return (
    <div className="space-y-3 p-3">
      <div>
        <div className="label mb-1">what the model was fitted on, split by time</div>
        <DataTable
          table={summary?.split}
          columns={[
            { key: 'split', align: 'left' },
            { key: 'rows' },
            { key: 'fraud' },
            { key: 'fraud_rate' },
            { key: 'vectors', help: 'Distinct attack vectors present in this window' },
            { key: 'start', align: 'left' },
            { key: 'end', align: 'left' },
          ]}
        />
        <p className="mt-1 text-2xs leading-relaxed text-fg-faint">
          Split by time, never at random. Fraud is campaign-structured: a random split puts half
          a mule ring in training and half in test, and the resulting figure measures nothing.
          {rows.length === 3 ? (
            <>
              {' '}
              Calibration is a separate window again, so the threshold is not chosen on rows the
              base learners were fitted on.
            </>
          ) : null}
        </p>
      </div>

      {Object.keys(timings).length ? (
        <div>
          <div className="label mb-1">offline pipeline cost, per stage</div>
          <dl className="flex flex-wrap gap-4 text-2xs">
            {Object.entries(timings).map(([stage, secs]) => (
              <div key={stage}>
                <dt className="font-mono text-fg-faint">{stage}</dt>
                <dd className="num text-fg-secondary">{duration(secs)}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-1 text-2xs leading-relaxed text-fg-faint">
            Batch figures for a whole run on one machine, and deliberately not converted into a
            throughput claim — most of the defend stage is the stress tests, not the model.
          </p>
        </div>
      ) : null}

      <div>
        <div className="label mb-1">what is deployed</div>
        <dl className="space-y-0.5 text-2xs">
          <Row label="run_id" value={p?.run_id} />
          <Row label="config_hash" value={p?.config_hash} />
          <Row label="library_hash" value={p?.library_hash} />
          <Row label="seed" value={p?.seed} />
          <Row label="generated_at" value={stamp(p?.generated_at)} />
        </dl>
        <p className="mt-1 text-2xs leading-relaxed text-fg-faint">
          The serving bundle holds the detector, its guards, its threshold and its column order
          in one atomic file. Saved separately they drift apart, and the failure is silent — a
          service returning confident numbers against a threshold from a different model.
        </p>
      </div>
    </div>
  )
}

/* -- shared ----------------------------------------------------------------- */

function Figure({
  label,
  value,
  note,
  tone = 'text-fg',
}: {
  label: string
  value: string
  note: string
  tone?: string
}) {
  return (
    <div className="max-w-56">
      <div className="label">{label}</div>
      <div className={`num text-xl leading-none ${tone}`}>{value}</div>
      <div className="mt-0.5 text-2xs leading-snug text-fg-faint">{note}</div>
    </div>
  )
}

function Row({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="font-mono text-fg-faint">{label}</dt>
      <dd className="num truncate text-fg-secondary" title={String(value ?? '')}>
        {value ?? '—'}
      </dd>
    </div>
  )
}

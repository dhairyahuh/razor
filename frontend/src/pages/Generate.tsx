import { Panel } from '@/components/ui/Panel'
import { DataTable } from '@/components/ui/DataTable'
import { PanelLoading } from '@/components/ui/RouteSkeleton'
import { useFidelity } from '@/api/queries'
import { count, pct } from '@/lib/format'
import {
  DEPLOYED_BAND,
  RECALL_AT_BUDGET_CEILING,
  gapReading,
  reading,
  tone,
  verdict,
} from '@/lib/fidelity'

/**
 * `/generate` — is the synthetic data honest?
 *
 * The screen leads with the separability pair because it is the single most
 * important number about a synthetic-data submission, and because it is the one
 * a sceptical reader is looking for.
 *
 * Both probes ask the same question of different matrices: train a throwaway
 * booster on an earlier time slice, score a later one, and report recall at a
 * 0.5% false-positive budget. `raw` sees only the fields the generator writes.
 * `derived` sees the engineered features the defence actually trains on.
 *
 * **A high number is bad on both.** Deployed card-fraud systems recover roughly
 * 50–85% of fraud at that budget; past the 92% ceiling in
 * `redteam/generate/fidelity.py` the pipeline flags its own data as measuring the
 * generator rather than a defence, and every detection metric downstream becomes
 * an upper bound. The gap between the two probes localises where any leak is:
 * wide means the feature engineering rather than the schema.
 *
 * This screen previously said the opposite — it rendered a high derived probe
 * green and captioned it "the detector doing work rather than the data leaking",
 * directly beside the run's own warning saying the reverse. The interpretation
 * now lives in `lib/fidelity.ts` next to the constants it depends on, so the two
 * cannot drift apart again.
 */
export default function Generate() {
  const { data, isLoading } = useFidelity()

  const raw = data?.raw_separability_recall ?? null
  const derived = data?.derived_separability_recall ?? null
  const flags = data?.flags ?? []
  const warnings = data?.warnings ?? []
  const zeroed = data?.zeroed ?? []

  return (
    <div className="grid h-full min-h-0 grid-cols-2 grid-rows-[auto_1fr_1fr] gap-2 overflow-auto p-2">
      <Panel
        id="tour-separability"
        title="Separability"
        source="generate_fidelity_scores.csv"
        subtitle="How much of the fraud a learner recovers from the data alone, before the defence exists."
        caveat={
          <>
            Both probes report recall at a 0.5% false-positive budget. Deployed card-fraud
            systems sit at{' '}
            <span className="num text-fg-secondary">
              {pct(DEPLOYED_BAND[0], 0)}–{pct(DEPLOYED_BAND[1], 0)}
            </span>
            ; past{' '}
            <span className="num text-fg-secondary">{pct(RECALL_AT_BUDGET_CEILING, 0)}</span>{' '}
            the pipeline flags its own data as measuring the generator rather than a defence.{' '}
            <span className="text-fg-muted">A high number here is an adverse finding, not a good one.</span>
          </>
        }
        className="col-span-2"
      >
        {isLoading ? (
          <PanelLoading />
        ) : (
          <div className="space-y-3 p-3">
            <div className="flex flex-wrap items-start gap-8">
              <Probe
                label="raw_separability_recall"
                value={raw}
                caption="a probe on the generator's own fields"
                which="raw"
              />
              <Probe
                label="derived_separability_recall"
                value={derived}
                caption="the same probe over the engineered features"
                which="derived"
              />

              <div className="min-w-56 flex-1">
                <div className="label mb-1">counts</div>
                <dl className="space-y-0.5 text-2xs">
                  <Row label="transactions" value={count(data?.counts.transactions)} />
                  <Row label="fraud" value={count(data?.counts.fraud)} />
                  <Row label="fraud_rate" value={pct(data?.counts.fraud_rate, 2)} />
                  <Row
                    label="agent_context_bundles"
                    value={count(data?.counts.agent_context_bundles)}
                  />
                  <Row label="transcripts" value={count(data?.counts.transcripts)} />
                </dl>
              </div>
            </div>

            <Scale raw={raw} derived={derived} />

            {gapReading(raw, derived) ? (
              <p className="text-2xs leading-relaxed text-fg-muted">{gapReading(raw, derived)}</p>
            ) : null}
          </div>
        )}
      </Panel>

      <Panel
        title="Fidelity checks"
        source="generate_fidelity_scores.csv"
        subtitle="Every check the generator ran on itself."
        caveat={
          flags.length || warnings.length || zeroed.length ? (
            <>
              <span className="text-amber">
                {flags.length} flag{flags.length === 1 ? '' : 's'}, {warnings.length} warning
                {warnings.length === 1 ? '' : 's'}
                {zeroed.length ? `, ${zeroed.length} scored zero` : ''}
              </span>{' '}
              — shown below rather than summarised away. A generator that reports no problems is
              usually one that is not looking.
            </>
          ) : (
            'No check failed on this run.'
          )
        }
        scroll
      >
        {isLoading ? (
          <PanelLoading />
        ) : (
          <div>
            {flags.length || warnings.length || zeroed.length ? (
              <div className="space-y-1 border-b border-line-subtle p-2">
                {flags.map((f) => (
                  <p key={f} className="text-2xs leading-snug text-red">
                    flag: {f}
                  </p>
                ))}
                {warnings.map((w) => (
                  <p key={w} className="text-2xs leading-snug text-amber">
                    warning: {w}
                  </p>
                ))}
                {/* A check can score zero without tripping a flag - flags fire only when
                    the interval's lower bound clears the ceiling. The run log names those
                    separately and so does this, because a silent zero is how the full
                    profile once reported "no flags" on a check that had bottomed out. */}
                {zeroed.map((z) => (
                  <p key={z} className="text-2xs leading-snug text-amber">
                    scored zero: <span className="font-mono">{z}</span> — the component
                    bottomed out without its interval clearing the ceiling, so no flag fired.
                  </p>
                ))}
              </div>
            ) : null}
            <DataTable table={data?.scores} sortBy="score" />
          </div>
        )}
      </Panel>

      <Panel
        title="Single-feature separability"
        source="generate_single_feature_auc.csv"
        subtitle="Any one feature that alone separates fraud from legitimate is a leak."
        caveat="A feature near 1.0 here means the generator wrote the label into that column. The ones at the top of this list are the ones worth arguing about."
        scroll
      >
        {isLoading ? (
          <PanelLoading />
        ) : (
          <DataTable table={data?.single_feature_auc} sortBy="auc" maxRows={40} />
        )}
      </Panel>

      <Panel
        title="Per-vector generation"
        source="generate_per_vector.csv"
        subtitle="What each vector actually produced."
        className="col-span-2"
        scroll
      >
        {isLoading ? <PanelLoading /> : <DataTable table={data?.per_vector} maxRows={60} />}
      </Panel>
    </div>
  )
}

function Probe({
  label,
  value,
  caption,
  which,
}: {
  label: string
  value: number | null
  caption: string
  which: 'raw' | 'derived'
}) {
  const v = verdict(value)

  return (
    <div className="max-w-72">
      <div className="label">{label}</div>
      <div className={`num text-3xl leading-none ${tone(v)}`}>{pct(value)}</div>
      <p className="mt-1 text-2xs text-fg-muted">{caption}</p>
      <p className={`mt-1 text-2xs leading-snug ${tone(v)}`}>{reading(v, which)}</p>
    </div>
  )
}

/**
 * The two probes on one axis, against the band and the ceiling.
 *
 * A number beside a caption is easy to misread — which is exactly how this screen
 * came to claim the opposite of what the pipeline said. Drawing both probes
 * against the target band and the flag line makes the reading structural: if a
 * marker is to the right of the ceiling, no wording can make that look like a
 * success.
 */
function Scale({ raw, derived }: { raw: number | null; derived: number | null }) {
  const [lo, hi] = DEPLOYED_BAND

  // Two label lanes. The probes routinely land within a couple of points of each
  // other — on the committed run they are 91.2% and 93.8% — and a single lane puts
  // one caption on top of the other exactly when both readings matter most.
  const BAR_TOP = 30

  return (
    <div>
      <div className="label mb-1.5">recall at a 0.5% false-positive budget</div>
      <div className="relative h-[60px]">
        {/* The full axis, 0 to 1. */}
        <div
          className="absolute inset-x-0 h-1.5 rounded-sm bg-raised"
          style={{ top: BAR_TOP }}
        />

        {/* The band deployed systems occupy: the target, not the maximum. */}
        <div
          className="absolute h-1.5 rounded-sm bg-green-wash"
          style={{ top: BAR_TOP, left: `${lo * 100}%`, width: `${(hi - lo) * 100}%` }}
          title={`deployed card-fraud systems: ${pct(lo, 0)}–${pct(hi, 0)}`}
        />

        {/* The ceiling past which the pipeline flags its own data. */}
        <div
          className="absolute w-px bg-[var(--amber)]"
          style={{
            left: `${RECALL_AT_BUDGET_CEILING * 100}%`,
            top: BAR_TOP - 4,
            height: 14,
          }}
          title={`the pipeline flags its own data past ${Math.round(RECALL_AT_BUDGET_CEILING * 100)}%`}
        />

        <Marker value={raw} label="raw" lane={0} barTop={BAR_TOP} />
        <Marker value={derived} label="derived" lane={1} barTop={BAR_TOP} />

        <div
          className="absolute inset-x-0 flex justify-between text-2xs text-fg-faint"
          style={{ top: BAR_TOP + 12 }}
        >
          <span className="num">0%</span>
          <span
            className="num absolute -translate-x-1/2"
            style={{ left: `${lo * 100}%` }}
            title="floor of the deployed band"
          >
            {pct(lo, 0)}
          </span>
          <span
            className="num absolute -translate-x-1/2"
            style={{ left: `${hi * 100}%` }}
            title="ceiling of the deployed band"
          >
            {pct(hi, 0)}
          </span>
          <span
            className="num absolute -translate-x-1/2 text-amber"
            style={{ left: `${RECALL_AT_BUDGET_CEILING * 100}%` }}
            title="the pipeline flags its own data past this line"
          >
            {pct(RECALL_AT_BUDGET_CEILING, 0)}
          </span>
          <span className="num">100%</span>
        </div>
      </div>
    </div>
  )
}

/**
 * One probe on the axis: a caption in its own lane, and a stem down to the bar.
 *
 * The stem length is derived from the lane rather than fixed, so both captions sit
 * clear of each other while both ticks still land on the bar itself.
 */
function Marker({
  value,
  label,
  lane,
  barTop,
}: {
  value: number | null
  label: string
  lane: 0 | 1
  barTop: number
}) {
  if (value === null || !Number.isFinite(value)) return null
  const v = verdict(value)
  const colour = v === 'realistic' ? 'var(--green)' : 'var(--amber)'

  const captionTop = lane * 14
  const stemTop = captionTop + 10

  return (
    <div
      className="absolute -translate-x-1/2"
      style={{ left: `${Math.min(Math.max(value, 0), 1) * 100}%`, top: 0 }}
      title={`${label}: ${pct(value)}`}
    >
      <div
        className={`absolute -translate-x-1/2 whitespace-nowrap font-display text-2xs font-semibold uppercase tracking-wider ${tone(v)}`}
        style={{ top: captionTop, lineHeight: 1 }}
      >
        {label} <span className="num normal-case">{pct(value)}</span>
      </div>
      <div
        className="absolute w-0.5 -translate-x-1/2 rounded-sm"
        style={{ top: stemTop, height: barTop - stemTop + 6, background: colour }}
        aria-hidden="true"
      />
    </div>
  )
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="font-mono text-fg-faint">{label}</dt>
      <dd className="num text-fg-secondary">{value}</dd>
    </div>
  )
}

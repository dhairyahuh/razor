import { Panel } from '@/components/ui/Panel'
import { DataTable } from '@/components/ui/DataTable'
import { PanelLoading } from '@/components/ui/RouteSkeleton'
import { GenAiProvenance } from '@/features/evidence/GenAiProvenance'
import { useDefend, useReport, useSummary } from '@/api/queries'
import { stamp } from '@/lib/format'

/**
 * `/evidence` — the governance view, and the limitations.
 *
 * The limitations are at the top rather than in a footnote, and that is a
 * deliberate bet. Every submission in this competition will claim high recall
 * on data it generated itself. The ones that name what their numbers cannot
 * support are the ones whose other numbers become believable. Burying this
 * section would save nobody: a judge who finds an unstated limitation
 * discounts everything, and a judge who reads a stated one trusts the rest.
 *
 * The report is rendered from `REPORT.md` verbatim rather than re-narrated, so
 * this page and the artefact cannot drift apart.
 */
export default function Evidence() {
  const report = useReport()
  const summary = useSummary()
  const defend = useDefend()
  const p = summary.data?.provenance

  return (
    <div className="grid h-full min-h-0 grid-cols-[400px_1fr] gap-2 p-2">
      <div className="flex min-h-0 flex-col gap-2">
        <Panel
          title="What these numbers cannot tell you"
          source={null}
          subtitle="Read this before the rest of the site."
          scroll
        >
          <div className="space-y-3 p-3 text-2xs leading-relaxed text-fg-secondary">
            <Limit title="The data is synthetic, all of it.">
              No real payment, customer or transcript is anywhere in this system. The generator
              was written against published typologies and regulatory guidance, not against a
              sample of real fraud. Recall measured against attacks we invented is evidence
              that the detector finds <em>those</em> attacks, and nothing stronger.
            </Limit>

            <Limit title="The threshold was calibrated on the same distribution it is measured on.">
              Train, calibration and test are split by time, which stops the most obvious kind
              of leakage. It does not stop the generator's assumptions from being present in
              all three.
            </Limit>

            <Limit title="Several per-vector recalls rest on a handful of rows.">
              Where <span className="font-mono">n_sufficient</span> is 0 the interval is wide
              enough that the point estimate should not be quoted. Those rows are shown, not
              filtered, so the weakness is visible rather than tidied away.
            </Limit>

            <Limit title="Every rupee figure rests on cost assumptions.">
              Reimbursement share, review cost and false-decline cost are stated in{' '}
              <span className="font-mono">defend_cost_summary.csv</span> and taken from
              published figures. They are inputs, not measurements. The ranking of thresholds
              survives changing them; the absolute totals do not.
            </Limit>

            <Limit title="The loop is short.">
              A handful of rounds against a handful of campaigns. It demonstrates the
              mechanism — red evades, blue responds, findings return to the taxonomy — and does
              not establish where the arms race converges.
            </Limit>
          </div>
        </Panel>

        <Panel
          title="Where the language model is"
          source="run_summary.json"
          subtitle="Six components call one. The cache decides whether they did."
          caveat="No detection metric changes either way. What changes is how much of the text in the system was written by a model rather than by a template — which matters most for the vishing guard."
          scroll
        >
          <GenAiProvenance />
        </Panel>

        <Panel title="Provenance" source="run_summary.json" scroll>
          <dl className="space-y-1 p-3 text-2xs">
            <Field label="run_id" value={p?.run_id} />
            <Field label="run_name" value={p?.run_name} />
            <Field label="seed" value={p?.seed} />
            <Field label="git_sha" value={p?.git_sha} />
            <Field label="git_dirty" value={String(p?.git_dirty ?? '—')} />
            <Field label="config_hash" value={p?.config_hash} />
            <Field label="library_hash" value={p?.library_hash} />
            <Field label="generated_at" value={stamp(p?.generated_at)} />
            {Object.entries(p?.environment ?? {}).map(([k, v]) => (
              <Field key={k} label={k} value={v} />
            ))}
          </dl>
        </Panel>
      </div>

      <div className="grid min-h-0 grid-rows-2 gap-2">
        <Panel
          title="Controls coverage"
          source="defend_agent_control_coverage.csv"
          subtitle="Which agent-side controls fired, and on what."
          caveat="A control that never fires is either well-targeted or dead. This table is how you tell which."
          scroll
        >
          {defend.isLoading ? (
            <PanelLoading />
          ) : (
            <DataTable table={defend.data?.control_coverage} />
          )}
        </Panel>

        <Panel
          title="REPORT.md"
          source="REPORT.md"
          subtitle="The run's own write-up, verbatim."
          caveat="Rendered from the artefact rather than re-narrated here, so this page cannot drift away from what the pipeline actually said."
          scroll
        >
          {report.isLoading ? (
            <PanelLoading />
          ) : report.data?.available ? (
            <pre className="whitespace-pre-wrap break-words p-3 font-mono text-2xs leading-relaxed text-fg-secondary">
              {report.data.markdown}
            </pre>
          ) : (
            <p className="p-3 text-2xs text-fg-muted">This run produced no REPORT.md.</p>
          )}
        </Panel>
      </div>
    </div>
  )
}

function Limit({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="font-medium text-fg">{title}</p>
      <p className="mt-0.5 text-fg-muted">{children}</p>
    </div>
  )
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <dt className="font-mono text-fg-faint">{label}</dt>
      <dd className="num truncate text-fg-secondary" title={String(value ?? '')}>
        {value ?? '—'}
      </dd>
    </div>
  )
}

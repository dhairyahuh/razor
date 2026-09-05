import { useEffect } from 'react'
import { useTransaction } from '@/api/queries'
import { OUTCOMES, outcomeOf, VERDICTS, verdictOf } from '@/lib/decision'
import { auto, clock, inr, pct, score as fmtScore } from '@/lib/format'
import { ErrorBoundary } from '@/components/ui/ErrorBoundary'
import { PanelLoading } from '@/components/ui/RouteSkeleton'
import { SourceLink } from '@/components/ui/SourceLink'
import { AgentBundle } from './AgentBundle'
import { FieldTable } from './FieldTable'
import type { ScoredTxn } from '@/types/api'

/**
 * The Transaction Inspector.
 *
 * This is the panel that answers the only question that matters about any
 * individual alert - *why this one* - and it is the thing that separates a
 * dashboard from a tool someone could actually work in. The brief lists it as
 * never-cut, and it is the second thing the guided tour opens.
 *
 * Structured as a descent from claim to evidence: what the system did, why it
 * says so, what would have changed it, what the model saw, and - for
 * agent-initiated payments - the exact text that was fed to the agent with the
 * injected span located to the character.
 *
 * A slide-over rather than a route so the stream keeps running behind it. Being
 * able to open a payment, read it, close it and still be in the same place in
 * the stream is what makes it feel like an operations tool.
 */

interface Props {
  txnId: string | null
  onClose: () => void
}

export function Inspector({ txnId, onClose }: Props) {
  const { data, isLoading, error } = useTransaction(txnId)

  // Escape closes. Anything that opens over the top of a running console has to
  // be dismissible without aiming at a small target, especially on a projector.
  useEffect(() => {
    if (!txnId) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [txnId, onClose])

  if (!txnId) return null

  return (
    <>
      {/* Deliberately light. A heavy scrim would black out the stream, and the
          point of a slide-over here is that the console stays visible. */}
      <div
        className="fixed inset-0 z-40 animate-fade-in bg-[var(--scrim)]"
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        role="dialog"
        aria-label={`Transaction ${txnId}`}
        className="fixed right-0 top-0 z-50 flex h-full w-[560px] max-w-[92vw] animate-slide-over flex-col border-l border-line bg-surface shadow-overlay"
      >
        <ErrorBoundary label="Transaction Inspector">
          {isLoading ? (
            <PanelLoading />
          ) : error || !data ? (
            <Failed txnId={txnId} message={error instanceof Error ? error.message : ''} onClose={onClose} />
          ) : (
            <Body data={data} onClose={onClose} />
          )}
        </ErrorBoundary>
      </aside>
    </>
  )
}

function Body({
  data,
  onClose,
}: {
  data: NonNullable<ReturnType<typeof useTransaction>['data']>
  onClose: () => void
}) {
  const txn = data.verdict as unknown as ScoredTxn
  const outcome = outcomeOf(txn)
  const style = OUTCOMES[outcome]
  const verdict = VERDICTS[verdictOf(txn)]
  const cf = data.counterfactuals

  return (
    <>
      <header className="shrink-0 border-b border-line-subtle px-4 py-3">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="label">transaction</div>
            <div className="truncate font-mono text-sm text-fg">{data.txn_id}</div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="shrink-0 rounded border border-line px-2 py-1 font-display text-2xs font-semibold uppercase tracking-wider text-fg-muted transition-colors hover:bg-hover hover:text-fg"
          >
            Close ESC
          </button>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          <Stat label="amount" value={inr(Number(txn.amount))} />
          <Stat label="rail" value={String(txn.rail ?? '—')} />
          <Stat label="time" value={clock(txn.timestamp)} />
          <Stat label="score" value={fmtScore(Number(txn.score))} />
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span
            className={`rounded border px-2 py-1 font-display text-2xs font-semibold uppercase tracking-wider ${style.chip}`}
          >
            {style.label}
          </span>
          <span className={`text-2xs ${verdict.fg}`}>{verdict.label}</span>
        </div>
        <p className="mt-2 text-2xs leading-snug text-fg-muted">{style.meaning}</p>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <Section title="Why" source="defend_scored_test_set.csv">
          {data.reason_codes.length ? (
            <ul className="space-y-1">
              {data.reason_codes.map((code) => (
                <li key={code} className="flex items-start gap-2">
                  <span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-fg-faint" />
                  <span className="font-mono text-2xs text-fg-secondary">{code}</span>
                </li>
              ))}
            </ul>
          ) : (
            <Note>
              {outcome === 'approve'
                ? 'Nothing to explain — this payment was approved.'
                : 'This run produced no reason codes. They are derived from permutation ' +
                  'importance, which a --shallow run skips.'}
            </Note>
          )}
        </Section>

        <Section title="What would have changed it" source="defend_counterfactuals.csv">
          {!cf.available ? (
            <Note>This run did not compute counterfactuals.</Note>
          ) : !cf.sampled ? (
            <Note>
              This alert was outside the sample. Deriving a counterfactual means re-scoring
              the payment many times, so the pipeline does a few hundred alerts rather than
              all of them — this one was not among them.
            </Note>
          ) : cf.rows.length === 0 ? (
            <Note>
              <span className="text-fg-secondary">No single change would have cleared it.</span>{' '}
              Every actionable field was tried at the legitimate quantiles and none on its own
              took the score below the threshold. The alert rests on a combination, which is
              a harder thing to dispute and a more interesting one to have found.
            </Note>
          ) : (
            <ul className="space-y-2">
              {cf.rows.map((r) => (
                <li key={r.column} className="rounded border border-line-subtle bg-raised p-2">
                  <p className="text-xs text-fg">{r.phrase}</p>
                  <p className="mt-1 font-mono text-2xs text-fg-faint">
                    {r.column}: {auto(r.from)} → {auto(r.to)} · score {fmtScore(r.score_after)}
                  </p>
                </li>
              ))}
            </ul>
          )}
        </Section>

        {data.agent_bundle ? (
          <Section title="What the agent read" source="agent_context.parquet">
            <AgentBundle bundle={data.agent_bundle} />
          </Section>
        ) : null}

        <Section title="What the model saw" source="defend_benign_baselines.csv">
          {Object.keys(data.blocks).length === 0 ? (
            <Note>
              The featured row for this payment is not in this run's working data, so its
              inputs cannot be shown.
            </Note>
          ) : (
            <FieldTable blocks={data.blocks} />
          )}
        </Section>

        <Section title="Related payments" source="defend_scored_test_set.csv">
          <Siblings siblings={data.siblings} />
        </Section>

        {Object.keys(data.labels).length ? (
          <Section title="Ground truth" source="data/transactions.parquet">
            <Note>
              Visible only because this is a labelled synthetic test window. A production
              console would not have this section, and none of it was available to the model.
            </Note>
            <dl className="mt-2 grid grid-cols-2 gap-x-4 gap-y-1">
              {Object.entries(data.labels).map(([k, v]) => (
                <div key={k} className="contents">
                  <dt className="font-mono text-2xs text-fg-faint">{k}</dt>
                  <dd className="font-mono text-2xs text-fg-secondary">
                    {v === null || v === undefined ? '—' : String(v)}
                  </dd>
                </div>
              ))}
            </dl>
          </Section>
        ) : null}
      </div>
    </>
  )
}

function Siblings({ siblings }: { siblings: TransactionDetailSiblings }) {
  const campaign = siblings.campaign ?? []
  const ring = siblings.ring ?? []

  if (!siblings.available || (campaign.length === 0 && ring.length === 0)) {
    return (
      <Note>
        No other payment in the test window shares this one's campaign or mule ring. For a
        legitimate payment that is expected; for an attack it means the generator produced it
        as a singleton.
      </Note>
    )
  }

  return (
    <div className="space-y-3">
      {campaign.length ? (
        <SiblingList
          title="Same campaign"
          note="Generated by the same attack run — the fraudster's batch, not the victim's."
          rows={campaign}
        />
      ) : null}
      {ring.length ? (
        <SiblingList
          title="Same mule ring"
          note="Payments funnelling into the same beneficiary cluster."
          rows={ring}
        />
      ) : null}
    </div>
  )
}

interface TransactionDetailSiblings {
  available: boolean
  campaign?: Array<Record<string, unknown>>
  ring?: Array<Record<string, unknown>>
}

function SiblingList({
  title,
  note,
  rows,
}: {
  title: string
  note: string
  rows: Array<Record<string, unknown>>
}) {
  return (
    <div>
      <div className="label">{title}</div>
      <p className="mb-1 text-2xs text-fg-faint">{note}</p>
      <ul className="divide-y divide-line-subtle rounded border border-line-subtle">
        {rows.slice(0, 8).map((r, i) => (
          <li key={String(r.txn_id ?? i)} className="flex items-center gap-3 px-2 py-1">
            <span className="num flex-1 truncate text-2xs text-fg-muted">
              {String(r.txn_id ?? '—')}
            </span>
            <span className="num text-2xs text-fg-secondary">{inr(Number(r.amount))}</span>
            <span
              className={`num text-2xs ${
                Number(r.decision) === 1 ? 'text-amber' : 'text-fg-faint'
              }`}
            >
              {fmtScore(Number(r.score))}
            </span>
          </li>
        ))}
      </ul>
      {rows.length > 8 ? (
        <p className="mt-1 text-2xs text-fg-faint">
          and {rows.length - 8} more · {pct(rows.filter((r) => Number(r.decision) === 1).length / rows.length)} of
          the group was flagged
        </p>
      ) : null}
    </div>
  )
}

function Section({
  title,
  source,
  children,
}: {
  title: string
  source: string
  children: React.ReactNode
}) {
  return (
    <section className="border-b border-line-subtle px-4 py-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="font-display text-xs font-semibold tracking-tight text-fg">{title}</h3>
        <SourceLink file={source} />
      </div>
      <ErrorBoundary label={title}>{children}</ErrorBoundary>
    </section>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="num text-sm text-fg">{value}</div>
    </div>
  )
}

function Note({ children }: { children: React.ReactNode }) {
  return <p className="text-2xs leading-relaxed text-fg-muted">{children}</p>
}

function Failed({
  txnId,
  message,
  onClose,
}: {
  txnId: string
  message: string
  onClose: () => void
}) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 p-6 text-center">
      <p className="text-sm text-fg">Could not load this payment</p>
      <p className="font-mono text-2xs text-fg-muted">{txnId}</p>
      {message ? <p className="max-w-xs text-2xs text-fg-faint">{message}</p> : null}
      <button
        type="button"
        onClick={onClose}
        className="mt-2 rounded border border-line px-2 py-1 font-display text-2xs font-semibold uppercase tracking-wider text-fg-secondary hover:bg-hover"
      >
        Close
      </button>
    </div>
  )
}

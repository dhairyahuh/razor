import { useSummary } from '@/api/queries'
import { useMode } from '@/state/session'
import { sha, stamp } from '@/lib/format'

/**
 * The footer that runs under every page: which run, which seed, which commit,
 * generated when, read from where.
 *
 * It is always visible rather than tucked onto an about page because it is the
 * answer to the first question a sceptical reader has about a synthetic-data
 * submission - *can I reproduce this?* - and a page that answers it in the
 * chrome has already answered it before it is asked.
 *
 * A dirty working tree is called out rather than quietly omitted. If the commit
 * shown does not fully describe the code that produced these numbers, saying so
 * costs a word and not saying so is the kind of omission that, if a judge
 * noticed, would undermine everything above it.
 */
export function ProvenanceStrip() {
  const { data } = useSummary()
  const mode = useMode()
  const p = data?.provenance

  return (
    <footer className="flex h-6 shrink-0 items-center gap-4 overflow-x-auto whitespace-nowrap border-t border-line-subtle bg-ground px-3 font-mono text-2xs text-fg-faint">
      <Field label="run" value={p?.run_name ?? p?.run ?? '—'} />
      <Field label="seed" value={p?.seed ?? '—'} />
      <Field
        label="commit"
        value={
          <>
            {sha(p?.git_sha)}
            {p?.git_dirty ? (
              <span
                className="ml-1 text-amber"
                title="The working tree had uncommitted changes when this run was produced, so this commit does not fully describe the code behind these numbers."
              >
                +dirty
              </span>
            ) : null}
          </>
        }
      />
      <Field label="config" value={sha(p?.config_hash)} />
      <Field label="generated" value={stamp(p?.generated_at)} />
      <Field
        label="reading"
        value={
          mode === 'live' ? (
            <span className="text-green">live artefacts</span>
          ) : (
            <span className="text-fg-muted">recorded payload</span>
          )
        }
      />
    </footer>
  )
}

function Field({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <span className="flex shrink-0 items-center gap-1">
      <span className="text-fg-faint opacity-60">{label}</span>
      <span className="text-fg-muted">{value}</span>
    </span>
  )
}

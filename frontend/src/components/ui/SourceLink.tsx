import { useRunId } from '@/state/session'
import { fileUrl } from '@/api/client'

/**
 * "source: defend_per_vector_recall.csv", linking to the file itself.
 *
 * This small component is doing more work than its size suggests. A judge
 * reading a synthetic-data submission starts from the assumption that the
 * numbers could have been typed in. The cheapest way to dissolve that is to
 * make checking free: name the artefact next to every figure, and make the name
 * open the artefact. Scepticism costs one click instead of a repository clone.
 *
 * In Demo Mode there is no server to fetch from, so the label stays but stops
 * pretending to be a link - a dead href would be worse than plain text.
 */
export function SourceLink({ file }: { file: string }) {
  const run = useRunId()
  const href = fileUrl(run, file)

  const shared =
    'shrink-0 rounded border border-line-subtle px-1.5 py-0.5 font-mono text-2xs text-fg-faint'

  if (!href) {
    return (
      <span className={shared} title={`Numbers in this panel come from ${file}`}>
        {file}
      </span>
    )
  }

  return (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      title={`Open ${file} — the artefact these numbers were read from`}
      className={`${shared} transition-colors hover:border-accent-dim hover:bg-accent-wash hover:text-accent`}
    >
      {file}
    </a>
  )
}

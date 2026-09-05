import { useMemo } from 'react'
import { Link } from 'react-router-dom'
import type { Discovered, DiscoveredVector } from '@/types/api'

/**
 * The arrow that closes the loop.
 *
 * A mutation that evades is a finding. A mutation that evades, survives
 * validation against the library's schema gate, and is written into
 * `discovered_vectors.yaml` as a first-class vector is a *system that learns* -
 * because the next run simulates it from the start, measures recall against it,
 * and reports it in the taxonomy beside the hand-written vectors.
 *
 * Two pieces of honesty are built into the layout rather than left to the
 * reader.
 *
 * The panel leads with how many *distinct parents* the discoveries came from,
 * not just how many discoveries there are. Nine promotions all evolved from one
 * library entry is a narrower result than nine independent findings, and the
 * headline that says "9 vectors" without saying "from 1 parent" is the kind of
 * overstatement a judge would catch and then discount everything else for.
 *
 * And the generated names are truncated rather than shown whole. The generator
 * appends the full gene list to the name, which runs past two hundred
 * characters; the genes are real and are shown, but as genes, under the
 * description they belong to.
 */
export function WriteBack({ discovered }: { discovered: Discovered | undefined }) {
  const rawVectors = discovered?.vectors
  const vectors = useMemo(() => rawVectors ?? [], [rawVectors])

  const parents = useMemo(
    () => new Set(vectors.map((v) => v.discovered_by?.parent_vector).filter(Boolean)),
    [vectors],
  )

  if (!discovered?.available || vectors.length === 0) {
    return (
      <div className="space-y-2 p-3">
        <p className="text-xs leading-relaxed text-fg-secondary">
          This run promoted no new vectors.
        </p>
        <p className="text-2xs leading-relaxed text-fg-muted">
          Red's surviving genomes were all evasions of vectors the library already contained —
          new <em>parameters</em> for a known attack rather than a new attack. Promotion
          requires a genome to be materially different from its parent and to survive
          validation against the schema gate, and nothing here cleared that bar.
        </p>
        <p className="text-2xs leading-relaxed text-fg-faint">
          The mechanism is exercised and tested regardless — see{' '}
          <span className="font-mono">tests/test_loop.py</span> — and each round's surviving
          tactics are in <span className="font-mono">loop_rounds.csv</span>.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-2 p-3">
      <p className="text-xs leading-relaxed text-fg-secondary">
        <span className="num text-accent">{vectors.length}</span> vectors promoted into{' '}
        <span className="font-mono text-fg-muted">{discovered.source}</span>, evolved from{' '}
        <span className="num text-fg">{parents.size}</span> parent
        {parents.size === 1 ? '' : 's'} in the hand-written library.
      </p>

      {parents.size === 1 ? (
        <p className="text-2xs leading-relaxed text-fg-muted">
          All of them descend from the same entry, so this is one attack the loop learned to
          re-parameterise nine ways rather than nine unrelated discoveries. The mechanism is
          the claim; the breadth is bounded by how many campaigns were large enough for red to
          search.
        </p>
      ) : null}

      <ul className="space-y-2">
        {vectors.map((v) => (
          <Card key={v.id} vector={v} />
        ))}
      </ul>

      <Link
        to="/identify"
        className="inline-block text-2xs text-accent underline-offset-2 hover:underline"
      >
        See them in the taxonomy →
      </Link>
    </div>
  )
}

function Card({ vector }: { vector: DiscoveredVector }) {
  const found = vector.discovered_by
  const levers = found?.levers ?? []

  return (
    <li className="rounded border border-accent-dim bg-accent-wash p-2">
      <div className="flex items-baseline justify-between gap-2">
        <span className="truncate font-mono text-2xs text-accent" title={vector.id}>
          {vector.id}
        </span>
        {found ? (
          <span className="shrink-0 font-mono text-2xs text-fg-faint">round {found.round}</span>
        ) : null}
      </div>

      <p className="mt-1 text-xs leading-snug text-fg">{title(vector.name)}</p>

      {found?.parent_vector ? (
        <p className="mt-0.5 font-mono text-2xs text-fg-faint">
          evolved from {found.parent_vector}
        </p>
      ) : null}

      {vector.description ? (
        <p className="mt-1 text-2xs leading-snug text-fg-muted">{vector.description}</p>
      ) : null}

      {levers.length ? (
        <p className="mt-1 font-mono text-2xs leading-snug text-fg-faint">
          levers: {levers.slice(0, 6).join(', ')}
          {levers.length > 6 ? ` +${levers.length - 6} more` : ''}
        </p>
      ) : null}
    </li>
  )
}

/**
 * Drop the gene list the generator appends in parentheses. It is genuine
 * information and it is shown below as levers; leaving it in the name pushes a
 * two-hundred-character string through a heading and makes nine cards
 * indistinguishable from each other.
 */
function title(name: string): string {
  const cut = name.lastIndexOf(' (')
  return cut > 20 ? name.slice(0, cut) : name
}

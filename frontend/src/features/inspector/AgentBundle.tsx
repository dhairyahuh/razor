import { useMemo } from 'react'

/**
 * The context window an agent read, with the injected instruction highlighted to
 * the character.
 *
 * This is the most persuasive single artefact in the submission, so it is worth
 * being precise about why. Prompt injection against payment agents is usually
 * discussed in the abstract. Here the actual bundle the agent consumed is on
 * screen, the injected span is marked at exact offsets the generator recorded
 * when it inserted the text, and the surrounding context is the ordinary
 * merchant and invoice content the payload was hidden inside. Nothing about it
 * is reconstructed or approximate.
 *
 * The offsets come from `payload_start` and `payload_end` in the corpus, written
 * at insertion time. Highlighting by searching the text for the payload would
 * be easy and wrong: payloads repeat innocuous phrasing on purpose, so a search
 * would sometimes mark the wrong occurrence and the reader would have no way to
 * tell.
 */

interface Bundle {
  text: string
  payload_start: number | null
  payload_end: number | null
  payload_family?: string | null
  [k: string]: unknown
}

export function AgentBundle({ bundle }: { bundle: Bundle }) {
  const { before, payload, after, located } = useMemo(() => split(bundle), [bundle])

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        {bundle.payload_family ? (
          <span className="rounded border border-red-dim bg-red-wash px-1.5 py-0.5 font-mono text-2xs text-red">
            {String(bundle.payload_family)}
          </span>
        ) : null}
        {located ? (
          <span className="font-mono text-2xs text-fg-faint">
            chars {bundle.payload_start}–{bundle.payload_end}
          </span>
        ) : null}
      </div>

      <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words rounded border border-line-subtle bg-inset p-3 font-mono text-2xs leading-relaxed text-fg-secondary">
        {located ? (
          <>
            {before}
            <mark className="rounded-sm bg-red-wash px-0.5 text-red-strong underline decoration-red decoration-wavy underline-offset-2">
              {payload}
            </mark>
            {after}
          </>
        ) : (
          bundle.text
        )}
      </pre>

      <p className="text-2xs leading-relaxed text-fg-faint">
        {located ? (
          <>
            The highlighted span is the injected instruction, at the exact offsets the
            generator recorded when it wrote them into the bundle — not a text search after
            the fact. Everything around it is ordinary merchant and invoice content, which is
            the point: the agent had no reason to treat any of it differently.
          </>
        ) : (
          <>
            This bundle carries no payload span, so nothing is highlighted. Benign bundles
            have none by definition; a poisoned one produced before spans were recorded also
            has none, and the honest thing is to show the text unmarked rather than guess.
          </>
        )}
      </p>
    </div>
  )
}

function split(bundle: Bundle) {
  const text = bundle.text ?? ''
  const start = bundle.payload_start
  const end = bundle.payload_end

  const valid =
    typeof start === 'number' &&
    typeof end === 'number' &&
    start >= 0 &&
    end > start &&
    end <= text.length

  if (!valid) return { before: text, payload: '', after: '', located: false }

  return {
    before: text.slice(0, start),
    payload: text.slice(start, end),
    after: text.slice(end),
    located: true,
  }
}

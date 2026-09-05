import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDefend, useIdentify } from '@/api/queries'
import { useSession } from '@/state/session'
import { useTour } from '@/features/tour/useTour'
import { pct } from '@/lib/format'

/**
 * ⌘K / Ctrl+K.
 *
 * Jumps to a route, a vector, or a metric. The metric entries are the ones that
 * earn this component its place: a judge who remembers a number from the report
 * and wants to know where it came from can type `value_recall` and be shown its
 * value and which artefact holds it, without knowing which of six routes it
 * lives on. That is a different thing from navigation, and it is the search a
 * sceptical reader actually performs.
 *
 * Built here rather than pulled in: it is a filtered list and a keydown handler,
 * and the smallest command-palette dependency is larger than this file.
 */

interface Item {
  id: string
  /** What is matched against. */
  label: string
  /** Shown to the right — a value, a source file, a hint. */
  detail?: string
  kind: 'route' | 'vector' | 'metric' | 'action'
  run: () => void
}

const KIND_STYLE: Record<Item['kind'], string> = {
  route: 'text-accent',
  vector: 'text-red',
  metric: 'text-money',
  action: 'text-amber',
}

export function CommandPalette() {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [cursor, setCursor] = useState(0)
  const input = useRef<HTMLInputElement>(null)

  const navigate = useNavigate()
  // Fetched on first open, then cached forever like everything else. The
  // palette is mounted on every route and must not cost the landing screen a
  // quarter of a megabyte for a list nobody has asked for.
  const identify = useIdentify(open)
  const defend = useDefend(open)
  const { togglePresentation, toggleTheme } = useSession()

  const items = useMemo<Item[]>(() => {
    const out: Item[] = [
      { id: 'r/', label: 'Live Threat Console', detail: '/', kind: 'route', run: () => navigate('/') },
      { id: 'r/identify', label: 'Attack Atlas', detail: '/identify', kind: 'route', run: () => navigate('/identify') },
      { id: 'r/generate', label: 'Fidelity Lab', detail: '/generate', kind: 'route', run: () => navigate('/generate') },
      { id: 'r/defend', label: 'Defence', detail: '/defend', kind: 'route', run: () => navigate('/defend') },
      { id: 'r/loop', label: 'The arms race', detail: '/loop', kind: 'route', run: () => navigate('/loop') },
      { id: 'r/deploy', label: 'Deployment and feasibility', detail: '/deploy', kind: 'route', run: () => navigate('/deploy') },
      { id: 'r/evidence', label: 'Evidence and limitations', detail: '/evidence', kind: 'route', run: () => navigate('/evidence') },

      { id: 'a/tour', label: 'Play the guided demo', detail: 'four minutes, interruptible', kind: 'action', run: () => useTour.getState().start() },
      { id: 'a/present', label: 'Toggle Presentation Mode', detail: 'larger type, thicker strokes', kind: 'action', run: togglePresentation },
      { id: 'a/theme', label: 'Toggle theme', detail: 'dark / light', kind: 'action', run: toggleTheme },
    ]

    for (const v of identify.data?.vectors ?? []) {
      out.push({
        id: `v/${v.id}`,
        label: v.id,
        detail:
          v.measured_recall === null
            ? v.simulated
              ? 'simulated, not measured'
              : 'documented only'
            : `recall ${pct(v.measured_recall)}`,
        kind: 'vector',
        run: () => navigate('/identify'),
      })
    }

    // Metric name to value and source file. The point of the palette for anyone
    // cross-checking the report against the site.
    for (const row of defend.data?.headline.rows ?? []) {
      out.push({
        id: `m/${row.metric}`,
        label: row.metric,
        detail: `${row.value ?? '—'} · defend_headline.csv`,
        kind: 'metric',
        run: () => navigate('/defend'),
      })
    }
    for (const row of defend.data?.cost_summary.rows ?? []) {
      out.push({
        id: `c/${row.metric}`,
        label: String(row.metric),
        detail: `${row.value ?? '—'} · defend_cost_summary.csv`,
        kind: 'metric',
        run: () => navigate('/defend'),
      })
    }

    return out
  }, [identify.data, defend.data, navigate, togglePresentation, toggleTheme])

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return items.slice(0, 40)
    // Substring, not fuzzy. The things being searched here are exact
    // identifiers - `APP-PIG-BUTCHERING`, `value_recall` - and fuzzy matching
    // on identifiers surfaces confident nonsense.
    return items.filter((i) => i.label.toLowerCase().includes(q)).slice(0, 40)
  }, [items, query])

  useEffect(() => setCursor(0), [query])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setOpen((o) => !o)
        setQuery('')
      } else if (e.key === 'Escape') {
        setOpen(false)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  useEffect(() => {
    if (open) input.current?.focus()
  }, [open])

  if (!open) return null

  const choose = (item: Item | undefined) => {
    if (!item) return
    item.run()
    setOpen(false)
  }

  return (
    <div
      className="fixed inset-0 z-[80] flex items-start justify-center bg-[var(--scrim)] pt-[12vh]"
      onMouseDown={() => setOpen(false)}
    >
      <div
        onMouseDown={(e) => e.stopPropagation()}
        className="w-[min(640px,92vw)] animate-fade-in overflow-hidden rounded-lg border border-line bg-overlay shadow-overlay"
      >
        <input
          ref={input}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'ArrowDown') {
              e.preventDefault()
              setCursor((c) => Math.min(c + 1, matches.length - 1))
            } else if (e.key === 'ArrowUp') {
              e.preventDefault()
              setCursor((c) => Math.max(c - 1, 0))
            } else if (e.key === 'Enter') {
              choose(matches[cursor])
            }
          }}
          placeholder="Route, attack vector, or metric name…"
          className="w-full border-b border-line-subtle bg-transparent px-4 py-3 font-mono text-sm text-fg outline-none placeholder:text-fg-faint"
        />

        <ul className="max-h-80 overflow-y-auto">
          {matches.map((item, i) => (
            <li key={item.id}>
              <button
                type="button"
                onMouseEnter={() => setCursor(i)}
                onClick={() => choose(item)}
                className={`flex w-full items-baseline gap-3 px-4 py-1.5 text-left ${
                  i === cursor ? 'bg-selected' : ''
                }`}
              >
                <span className={`w-14 shrink-0 font-display text-2xs uppercase tracking-wider ${KIND_STYLE[item.kind]}`}>
                  {item.kind}
                </span>
                <span className="min-w-0 flex-1 truncate font-mono text-2xs text-fg">
                  {item.label}
                </span>
                {item.detail ? (
                  <span className="num shrink-0 text-2xs text-fg-faint">{item.detail}</span>
                ) : null}
              </button>
            </li>
          ))}
          {matches.length === 0 ? (
            <li className="px-4 py-4 text-center text-2xs text-fg-faint">
              {identify.isLoading || defend.isLoading
                ? 'Loading vectors and metrics…'
                : 'Nothing matches.'}
            </li>
          ) : null}
        </ul>

        <div className="flex items-center gap-3 border-t border-line-subtle px-4 py-1.5 text-2xs text-fg-faint">
          <span>↑↓ move</span>
          <span>↵ open</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  )
}

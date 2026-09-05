import { type ReactNode } from 'react'
import { ErrorBoundary } from './ErrorBoundary'
import { SourceLink } from './SourceLink'

/**
 * The container every piece of content on every route sits in.
 *
 * Three things are structural rather than cosmetic, and are why this is a
 * component instead of a div with classes repeated forty times:
 *
 *  1. It error-boundaries its own children. A panel is the unit of failure in
 *     this application - one missing artefact takes out one panel and nothing
 *     else. Making that automatic is the only way it stays true as panels get
 *     added under time pressure.
 *
 *  2. It carries provenance. Judges spot-check numbers against the CSVs, and a
 *     panel that cannot name the file its figures came from should not be
 *     showing figures. `source` is required for that reason; panels that are
 *     genuinely chrome pass `source={null}` and say so explicitly.
 *
 *  3. It reserves a fixed header row whether or not there are actions, so a
 *     grid of panels aligns at the title baseline instead of wherever each
 *     one's content happens to start.
 */

interface PanelProps {
  title: string
  /**
   * The artefact these numbers came from, relative to the run directory - e.g.
   * `defend_per_vector_recall.csv`. Pass `null` only when the panel renders no
   * measured quantity at all.
   */
  source: string | string[] | null
  /** One line under the title: what the reader is looking at and why. */
  subtitle?: ReactNode
  /** Top-right controls - filters, toggles, an export button. */
  actions?: ReactNode
  /**
   * Where the panel's numbers stop being measurement and start being
   * assumption, or what they cannot tell you. Rendered as a footnote in a
   * quieter voice. The brief asks for limitations to be visible, not buried on
   * one page.
   */
  caveat?: ReactNode
  /** Panels own their scroll; the page does not. */
  scroll?: boolean
  className?: string
  /** Targets for the guided tour, which walks specific panels by name. */
  id?: string
  children: ReactNode
}

export function Panel({
  title,
  source,
  subtitle,
  actions,
  caveat,
  scroll = false,
  className = '',
  id,
  children,
}: PanelProps) {
  const sources = source === null ? [] : Array.isArray(source) ? source : [source]

  return (
    <section
      id={id}
      className={`flex min-h-0 min-w-0 flex-col rounded border border-line-subtle bg-surface ${className}`}
    >
      <header className="flex shrink-0 items-start justify-between gap-3 border-b border-line-subtle px-3 py-2">
        <div className="min-w-0">
          <h2 className="truncate font-display text-sm font-semibold tracking-tight text-fg">
            {title}
          </h2>
          {subtitle ? (
            <p className="mt-0.5 text-2xs leading-snug text-fg-muted">{subtitle}</p>
          ) : null}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {actions}
          {sources.map((s) => (
            <SourceLink key={s} file={s} />
          ))}
        </div>
      </header>

      <div className={`min-h-0 flex-1 ${scroll ? 'overflow-auto' : 'overflow-hidden'}`}>
        <ErrorBoundary label={title}>{children}</ErrorBoundary>
      </div>

      {caveat ? (
        <footer className="shrink-0 border-t border-line-subtle px-3 py-1.5 text-2xs leading-snug text-fg-faint">
          {caveat}
        </footer>
      ) : null}
    </section>
  )
}

/**
 * The body wrapper for a panel whose content is prose or a small stack of
 * figures rather than a chart or table. Charts want the raw box.
 */
export function PanelBody({
  className = '',
  children,
}: {
  className?: string
  children: ReactNode
}) {
  return <div className={`p-3 ${className}`}>{children}</div>
}

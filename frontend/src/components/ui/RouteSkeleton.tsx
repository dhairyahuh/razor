/**
 * What a route shows while its chunk is arriving.
 *
 * Panel-shaped rather than a spinner, so the layout does not jump when the real
 * content lands - a screen that reflows on every navigation reads as slow even
 * when it is not. No shimmer animation: at this size it is noise, and it would
 * be the only thing moving on a page whose whole subject is a stream that
 * actually moves.
 */
export function RouteSkeleton() {
  return (
    <div className="grid h-full grid-cols-3 grid-rows-2 gap-2 p-2" aria-busy="true">
      {Array.from({ length: 6 }, (_, i) => (
        <div key={i} className="rounded border border-line-subtle bg-surface opacity-50" />
      ))}
      <span className="sr-only">Loading</span>
    </div>
  )
}

/**
 * The empty state for a panel whose artefact the run never produced.
 *
 * Names the file. "No data" tells a reader nothing they can act on; "this run
 * has no defend_cost_curve.csv" tells them whether they are looking at a
 * partial run, a misconfiguration, or a genuine absence.
 */
export function NoArtefact({ file, note }: { file: string; note?: string }) {
  return (
    <div className="flex h-full min-h-20 flex-col items-center justify-center gap-1 p-4 text-center">
      <p className="text-xs text-fg-muted">
        This run did not produce <span className="font-mono text-fg-secondary">{file}</span>.
      </p>
      {note ? <p className="max-w-sm text-2xs text-fg-faint">{note}</p> : null}
    </div>
  )
}

/** A panel whose artefact exists but is empty — a different fact, said differently. */
export function NoRows({ file, note }: { file: string; note?: string }) {
  return (
    <div className="flex h-full min-h-20 flex-col items-center justify-center gap-1 p-4 text-center">
      <p className="text-xs text-fg-muted">
        <span className="font-mono text-fg-secondary">{file}</span> is empty for this run.
      </p>
      {note ? <p className="max-w-sm text-2xs text-fg-faint">{note}</p> : null}
    </div>
  )
}

/** While a panel's data is in flight. Sized to the panel, not to the page. */
export function PanelLoading() {
  return (
    <div className="flex h-full min-h-20 items-center justify-center" aria-busy="true">
      <span className="label text-fg-faint">loading</span>
    </div>
  )
}

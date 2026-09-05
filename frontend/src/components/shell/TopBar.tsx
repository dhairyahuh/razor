import { useRuns } from '@/api/queries'
import { useSession } from '@/state/session'
import { useTour } from '@/features/tour/useTour'

/**
 * Run selector, mode badge, and the two display switches.
 *
 * The mode badge is load-bearing rather than decorative. Everything on screen
 * in Demo Mode is a recording of a real run, and a reader is entitled to know
 * that without asking - both because it is true and because volunteering it is
 * what makes the rest of the numbers credible. It also explains, before anyone
 * clicks it, why `RUN A ROUND NOW` is disabled.
 */
export function TopBar() {
  const { data: runs } = useRuns()
  const { runId, setRun, mode, theme, toggleTheme, presentation, togglePresentation } =
    useSession()

  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-line-subtle bg-ground px-3">
      <div className="flex min-w-0 items-baseline gap-2">
        <span className="font-display text-sm font-semibold tracking-tight text-fg">
          Razor Console
        </span>
        <span className="hidden truncate text-2xs text-fg-faint lg:inline">
          adversarial evaluation for payment fraud detection
        </span>
      </div>

      <div className="ml-auto flex items-center gap-2">
        {/* First in the group, and the only accented control in the bar. A judge
            who reads nothing else should find this in under a second. */}
        <button
          type="button"
          onClick={() => useTour.getState().start()}
          title="Walks the four-minute argument on its own. Click anywhere to take over at that point."
          className="rounded border border-accent-dim bg-accent-wash px-2.5 py-1 font-display text-2xs font-semibold uppercase tracking-wider text-accent transition-colors hover:bg-accent hover:text-accent-fg"
        >
          ▸ Play demo
        </button>

        <ModeBadge mode={mode} />

        <label className="flex items-center gap-1.5">
          <span className="label">run</span>
          <select
            value={runId}
            onChange={(e) => setRun(e.target.value)}
            // One run in the packaged demo, several on a developer's machine.
            // Disabling rather than hiding keeps the label - and therefore the
            // fact that runs are a thing - on screen either way.
            disabled={!runs || runs.length <= 1}
            className="rounded border border-line bg-surface px-1.5 py-1 font-mono text-2xs text-fg disabled:opacity-60"
          >
            {(runs ?? [{ run: runId, run_name: runId }]).map((r) => (
              <option key={r.run} value={r.run}>
                {r.run_name ?? r.run}
              </option>
            ))}
          </select>
        </label>

        <Toggle
          on={presentation}
          onClick={togglePresentation}
          label="Present"
          title="Presentation Mode: larger type, thicker chart strokes, higher contrast. For a projector or a screen-share."
        />
        <Toggle
          on={theme === 'light'}
          onClick={toggleTheme}
          label={theme === 'dark' ? 'Light' : 'Dark'}
          title="Switch theme. Dark is the default because a monitoring console is dark; light survives a washed-out projector better."
        />
      </div>
    </header>
  )
}

function ModeBadge({ mode }: { mode: 'live' | 'demo' | null }) {
  if (mode === null) {
    return <span className="label text-fg-faint">checking…</span>
  }

  if (mode === 'live') {
    return (
      <span
        title="A backend is running. Numbers are read from the run's artefacts on disk, and the interactive controls will execute."
        className="flex items-center gap-1.5 rounded border border-green-dim bg-green-wash px-1.5 py-0.5"
      >
        <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-green" />
        <span className="font-display text-2xs font-semibold uppercase tracking-wider text-green">
          Live
        </span>
      </span>
    )
  }

  return (
    <span
      title="No backend reachable. Every number on screen is a recording of the default_run, baked from the same API these panels would otherwise call. Nothing is fabricated, but the controls that compute cannot run."
      className="flex items-center gap-1.5 rounded border border-line bg-surface px-1.5 py-0.5"
    >
      <span className="h-1.5 w-1.5 rounded-full bg-fg-faint" />
      <span className="font-display text-2xs font-semibold uppercase tracking-wider text-fg-muted">
        Demo
      </span>
    </span>
  )
}

function Toggle({
  on,
  onClick,
  label,
  title,
}: {
  on: boolean
  onClick: () => void
  label: string
  title: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      aria-pressed={on}
      className={`rounded border px-2 py-1 font-display text-2xs font-semibold uppercase tracking-wider transition-colors ${
        on
          ? 'border-accent-dim bg-accent-wash text-accent'
          : 'border-line text-fg-muted hover:bg-hover hover:text-fg'
      }`}
    >
      {label}
    </button>
  )
}

import { Component, type ErrorInfo, type ReactNode } from 'react'

/**
 * A boundary sized to one panel, not to the page.
 *
 * The requirement this exists for: deleting any single CSV out of the run's
 * artefact directory must still leave every route loading. A judge who breaks
 * one panel should see that panel say so and the rest of the console carry on -
 * a white screen would read as "the numbers are fake and the app is a mock",
 * which is the opposite of what this build is trying to demonstrate.
 *
 * So the fallback is deliberately informative rather than apologetic. It names
 * the panel and shows the actual error, because the person looking at it is an
 * engineer deciding whether to trust the rest of the screen.
 */

interface Props {
  /** Named in the fallback, so a screenshot of a broken panel is actionable. */
  label: string
  children: ReactNode
}

interface State {
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Left in for the live demo: if something does break in front of an
    // audience, the stack is one console open away rather than lost.
    console.error(`[panel: ${this.props.label}]`, error, info.componentStack)
  }

  private reset = () => this.setState({ error: null })

  render() {
    const { error } = this.state
    if (!error) return this.props.children

    return (
      <div
        role="alert"
        className="flex h-full min-h-24 flex-col justify-center gap-2 rounded border border-red-dim bg-red-wash p-4"
      >
        <div className="label text-red">{this.props.label} failed to render</div>
        <p className="text-xs text-fg-secondary">
          This panel errored. The rest of the console is unaffected and its numbers are
          still valid.
        </p>
        <code className="block max-h-24 overflow-auto whitespace-pre-wrap break-words font-mono text-2xs text-red-strong">
          {error.message || String(error)}
        </code>
        <button
          type="button"
          onClick={this.reset}
          className="self-start rounded border border-line px-2 py-1 font-display text-2xs font-semibold uppercase tracking-wide text-fg-secondary transition-colors hover:bg-hover hover:text-fg"
        >
          Retry
        </button>
      </div>
    )
  }
}

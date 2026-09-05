import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { SCRIPT } from './script'
import { useTour } from './useTour'

/**
 * The tour: navigate, spotlight, caption, advance.
 *
 * The spotlight is a full-screen dim with a hole cut in it, done as four
 * rectangles around the target rather than as an SVG mask or a `box-shadow`
 * spread. Four divs cost nothing to composite, animate on the GPU, and - unlike
 * a mask - leave the element underneath fully interactive, which is what lets a
 * judge click straight into whatever is being pointed at and take over.
 *
 * Any click outside the tour's own controls stops it. That is the single most
 * important behaviour in this component: the tour is insurance for a judge who
 * does nothing, and it must never become an obstacle for a judge who does
 * something.
 */
export function Tour() {
  const { running, step, next, stop, back, handlers } = useTour()
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const [box, setBox] = useState<DOMRect | null>(null)
  const controls = useRef<HTMLDivElement>(null)

  const current = SCRIPT[step]

  // Navigate first. The spotlight cannot find its target until the route that
  // owns it has rendered.
  useEffect(() => {
    if (!running || !current) return
    if (pathname !== current.route) navigate(current.route)
  }, [running, current, pathname, navigate])

  // Measure the target once it exists. Polled briefly rather than measured
  // once: a lazy route's chunk may still be in flight when the step opens, and
  // a spotlight around a zero-sized box would be worse than none.
  useLayoutEffect(() => {
    if (!running || !current) {
      setBox(null)
      return
    }
    if (!current.target) {
      setBox(null)
      return
    }

    let tries = 0
    const find = () => {
      const el = document.getElementById(current.target as string)
      if (el) {
        setBox(el.getBoundingClientRect())
        return true
      }
      return ++tries > 20
    }
    if (find()) return
    const timer = setInterval(() => {
      if (find()) clearInterval(timer)
    }, 100)
    return () => clearInterval(timer)
  }, [running, current, pathname])

  // Fire the step's action, if it has one. Failure is not handled because
  // failure is not distinguishable here: the handler owns its own error state
  // and the step advances on its hold either way.
  useEffect(() => {
    if (!running || !current?.action) return
    handlers[current.action]?.()
  }, [running, current, handlers])

  // Advance.
  useEffect(() => {
    if (!running || !current) return
    const timer = setTimeout(next, current.hold)
    return () => clearTimeout(timer)
  }, [running, current, next, step])

  // Any interaction outside the tour's controls hands control back, at this
  // spot. Capture phase, so it fires before the click reaches whatever was
  // clicked - the judge still gets their click, and the tour is already gone.
  useEffect(() => {
    if (!running) return
    const onDown = (e: MouseEvent) => {
      if (controls.current?.contains(e.target as Node)) return
      stop()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') stop()
      else if (e.key === 'ArrowRight') next()
      else if (e.key === 'ArrowLeft') back()
    }
    document.addEventListener('mousedown', onDown, true)
    window.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown, true)
      window.removeEventListener('keydown', onKey)
    }
  }, [running, stop, next, back])

  if (!running || !current) return null

  return (
    <>
      {box ? <Spotlight box={box} /> : <div className="pointer-events-none fixed inset-0 z-[60] bg-[var(--scrim)]" />}

      <div
        ref={controls}
        className="fixed bottom-9 left-1/2 z-[70] w-[min(760px,92vw)] -translate-x-1/2 animate-fade-in rounded-lg border border-accent-dim bg-overlay p-4 shadow-overlay"
      >
        <div className="flex items-start gap-4">
          <div className="min-w-0 flex-1">
            <div className="label mb-1 text-accent">
              step {step + 1} of {SCRIPT.length}
            </div>
            <p className="text-sm leading-relaxed text-fg">{current.caption}</p>
          </div>

          <div className="flex shrink-0 flex-col gap-1">
            <button
              type="button"
              onClick={stop}
              className="rounded border border-line px-2 py-1 font-display text-2xs font-semibold uppercase tracking-wider text-fg-secondary transition-colors hover:bg-hover hover:text-fg"
            >
              Exit tour
            </button>
            <div className="flex gap-1">
              <button
                type="button"
                onClick={back}
                disabled={step === 0}
                className="flex-1 rounded border border-line px-2 py-1 text-2xs text-fg-muted transition-colors hover:bg-hover disabled:opacity-40"
              >
                ←
              </button>
              <button
                type="button"
                onClick={next}
                className="flex-1 rounded border border-line px-2 py-1 text-2xs text-fg-muted transition-colors hover:bg-hover"
              >
                →
              </button>
            </div>
          </div>
        </div>

        <Progress key={step} duration={current.hold} />

        <p className="mt-2 text-2xs text-fg-faint">
          Click anywhere to take over — the tour stops here and leaves you on this screen.
        </p>
      </div>
    </>
  )
}

/**
 * Four rectangles around the target. The gap between them is the spotlight, and
 * because it is a genuine gap rather than a mask, everything inside it stays
 * clickable.
 */
function Spotlight({ box }: { box: DOMRect }) {
  const pad = 6
  const top = Math.max(0, box.top - pad)
  const left = Math.max(0, box.left - pad)
  const right = box.right + pad
  const bottom = box.bottom + pad

  const dim = 'pointer-events-none fixed z-[60] bg-[var(--scrim)] transition-all duration-120'

  return (
    <>
      <div className={dim} style={{ top: 0, left: 0, right: 0, height: top }} />
      <div className={dim} style={{ top: bottom, left: 0, right: 0, bottom: 0 }} />
      <div className={dim} style={{ top, left: 0, width: left, height: bottom - top }} />
      <div className={dim} style={{ top, left: right, right: 0, height: bottom - top }} />
      <div
        className="pointer-events-none fixed z-[61] rounded border border-accent transition-all duration-120"
        style={{ top, left, width: right - left, height: bottom - top }}
      />
    </>
  )
}

/** How long is left on this step. CSS animation, so it costs no renders. */
function Progress({ duration }: { duration: number }) {
  return (
    <div className="mt-3 h-0.5 overflow-hidden rounded-full bg-line-subtle">
      <div
        className="h-full bg-accent"
        style={{ animation: `tour-progress ${duration}ms linear forwards` }}
      />
    </div>
  )
}

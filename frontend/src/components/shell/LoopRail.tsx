import { NavLink, useLocation } from 'react-router-dom'

/**
 * The persistent left rail, drawn as the loop rather than listed as a menu.
 *
 * This is the one piece of chrome that is also the argument. The claim the whole
 * submission rests on is that the four pillars are a *cycle* - Identify feeds
 * Generate feeds Defend feeds Loop, and Loop writes new vectors back into
 * Identify - and a vertical list of four links quietly asserts the opposite,
 * that they are four separate things you look at in order and then stop.
 *
 * So the ring is real geometry: an SVG circle with the four pillars placed on
 * it, a return arrow closing it, and the current route lit on the ring itself.
 * A judge who never clicks anything has still been told the thesis.
 *
 * Drawn by hand rather than with a chart library because it is four points on a
 * circle - reaching for a dependency here would cost more bytes than the
 * geometry does.
 */

interface Stop {
  to: string
  /** Short label on the ring. */
  label: string
  /** What this pillar hands to the next one. Shown under the ring on hover. */
  hands: string
  /** Angle in degrees, clockwise from twelve o'clock. */
  angle: number
}

const STOPS: Stop[] = [
  { to: '/identify', label: 'IDENTIFY', hands: 'a taxonomy of attack vectors', angle: 0 },
  { to: '/generate', label: 'GENERATE', hands: 'synthetic payments carrying them', angle: 90 },
  { to: '/defend', label: 'DEFEND', hands: 'a measured detector and its blind spots', angle: 180 },
  { to: '/loop', label: 'LOOP', hands: 'new vectors, written back into the taxonomy', angle: 270 },
]

const R = 46
const CX = 62
const CY = 62

function at(angle: number, radius = R) {
  const rad = ((angle - 90) * Math.PI) / 180
  return { x: CX + radius * Math.cos(rad), y: CY + radius * Math.sin(rad) }
}

export function LoopRail() {
  const { pathname } = useLocation()
  const active = STOPS.find((s) => pathname.startsWith(s.to))

  return (
    <nav
      aria-label="Pillars"
      className="flex w-[152px] shrink-0 flex-col border-r border-line-subtle bg-ground"
    >
      <div className="px-3 pb-1 pt-3">
        <NavLink
          to="/"
          className={({ isActive }) =>
            `block rounded border px-2 py-1.5 text-center font-display text-2xs font-semibold uppercase tracking-widest transition-colors ${
              isActive
                ? 'border-accent-dim bg-accent-wash text-accent'
                : 'border-line-subtle text-fg-muted hover:bg-hover hover:text-fg'
            }`
          }
        >
          Live console
        </NavLink>
      </div>

      <div className="relative px-2 py-1">
        <svg viewBox="0 0 124 124" className="w-full" role="img" aria-hidden="true">
          {/* The ring. Dashed on the arc that returns from Loop to Identify,
              because that segment is the feedback and should read as the part
              that is still moving rather than as more structure. */}
          <circle
            cx={CX}
            cy={CY}
            r={R}
            fill="none"
            stroke="var(--border)"
            strokeWidth="var(--stroke-thin)"
          />

          {/* Direction of travel: four short arrowheads on the ring between the
              stops, so the cycle has a direction even when nothing is hovered. */}
          {[45, 135, 225, 315].map((a) => {
            const p = at(a)
            return (
              <polygon
                key={a}
                points="0,-3.2 5.2,0 0,3.2"
                fill="var(--border-strong)"
                transform={`translate(${p.x} ${p.y}) rotate(${a})`}
              />
            )
          })}

          {/* The closing arc, Loop back to Identify, drawn over the ring in the
              accent so the feedback edge is the thing the eye lands on. */}
          <path
            d={`M ${at(272).x} ${at(272).y} A ${R} ${R} 0 0 1 ${at(356).x} ${at(356).y}`}
            fill="none"
            stroke="var(--accent)"
            strokeWidth="var(--stroke)"
            strokeLinecap="round"
            strokeDasharray="3 3"
            className="opacity-80"
          />
          <polygon
            points="0,-3.6 6,0 0,3.6"
            fill="var(--accent)"
            transform={`translate(${at(356).x} ${at(356).y}) rotate(356)`}
          />

          {STOPS.map((stop) => {
            const p = at(stop.angle)
            const on = active?.to === stop.to
            return (
              <g key={stop.to}>
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={on ? 6.5 : 4.5}
                  fill={on ? 'var(--accent)' : 'var(--bg-ground)'}
                  stroke={on ? 'var(--accent)' : 'var(--border-strong)'}
                  strokeWidth="var(--stroke-thin)"
                />
                {on ? (
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={10}
                    fill="none"
                    stroke="var(--accent)"
                    strokeWidth="var(--stroke-thin)"
                    className="opacity-40"
                  />
                ) : null}
              </g>
            )
          })}

          <text
            x={CX}
            y={CY - 3}
            textAnchor="middle"
            className="fill-[var(--fg-faint)] font-mono"
            style={{ fontSize: 8 }}
          >
            closed
          </text>
          <text
            x={CX}
            y={CY + 7}
            textAnchor="middle"
            className="fill-[var(--fg-faint)] font-mono"
            style={{ fontSize: 8 }}
          >
            loop
          </text>
        </svg>
      </div>

      {/* The labels sit under the ring rather than around it: at 152px wide,
          text placed at four compass points either collides or shrinks past
          legibility, and an unreadable diagram argues nothing. The ring shows
          the structure, these are the handles. */}
      <ul className="flex flex-col gap-px px-2 pb-2">
        {STOPS.map((stop, i) => (
          <li key={stop.to}>
            <NavLink
              to={stop.to}
              title={`${stop.label} hands the next stage ${stop.hands}`}
              className={({ isActive }) =>
                `group flex items-center gap-2 rounded px-2 py-1.5 transition-colors ${
                  isActive ? 'bg-accent-wash text-accent' : 'text-fg-secondary hover:bg-hover'
                }`
              }
            >
              <span className="num text-2xs text-fg-faint">{i + 1}</span>
              <span className="font-display text-2xs font-semibold uppercase tracking-wider">
                {stop.label}
              </span>
            </NavLink>
          </li>
        ))}
      </ul>

      {/* Below the rule: the two routes that are not pillars. Deploy asks whether
          any of this could run in a live payment system, and Evidence says what
          the numbers cannot support. Both sit outside the ring because neither is
          a stage of the loop — putting them on it would overstate the cycle. */}
      <div className="mt-auto flex flex-col gap-px border-t border-line-subtle px-2 py-2">
        {[
          { to: '/deploy', label: 'Deploy' },
          { to: '/evidence', label: 'Evidence' },
        ].map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            className={({ isActive }) =>
              `rounded px-2 py-1.5 font-display text-2xs font-semibold uppercase tracking-wider transition-colors ${
                isActive ? 'bg-accent-wash text-accent' : 'text-fg-muted hover:bg-hover'
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}

/**
 * Where the three layers sit relative to the authorisation window.
 *
 * This is the one panel on the site that is architecture rather than
 * measurement, and it is drawn rather than tabulated because the claim it makes
 * is structural: two of the three layers are *inline* and must answer inside the
 * window a rail gives you, one is a feature store the caller owns, and
 * everything that trains is off the path entirely.
 *
 * The distinction the diagram has to carry — and the reason a bulleted list will
 * not do — is that a payment does not pass through these in a straight line. The
 * deterministic guards can decline on their own evidence without the model
 * having an opinion, which is what makes their declines disputable on facts.
 *
 * Hand-drawn SVG rather than a diagramming dependency: it is nine boxes and six
 * arrows, and the tokens have to resolve to CSS custom properties for both
 * themes to be correct by construction.
 */

interface Stage {
  x: number
  w: number
  label: string
  detail: string
  /** Inline stages sit inside the authorisation window and are drawn solid. */
  inline: boolean
}

const ROW_Y = 46
const H = 34

const STAGES: Stage[] = [
  { x: 8, w: 96, label: 'payment', detail: 'ISO 20022 / UPI', inline: true },
  { x: 116, w: 116, label: 'feature store', detail: 'trailing windows, graph', inline: true },
  { x: 244, w: 128, label: 'deterministic guards', detail: 'intent · agent controls', inline: true },
  { x: 384, w: 116, label: 'ensemble', detail: 'boosting + forest + iso', inline: true },
  { x: 512, w: 104, label: 'decision', detail: 'union of both', inline: true },
]

export function AuthFlow() {
  return (
    <div className="overflow-x-auto p-3">
      <svg viewBox="0 0 632 132" className="w-full min-w-[560px]" role="img"
           aria-label="Where the defence sits in the authorisation path">
        {/* The authorisation window: everything inside it has a latency budget. */}
        <rect
          x={110}
          y={22}
          width={512}
          height={H + 22}
          rx={4}
          fill="var(--accent-wash)"
          stroke="var(--accent-dim)"
          strokeWidth="var(--stroke-thin)"
          strokeDasharray="4 3"
        />
        <text x={366} y={16} textAnchor="middle" className="fill-[var(--accent)] font-display"
              style={{ fontSize: 9, letterSpacing: '0.08em' }}>
          INLINE — INSIDE THE AUTHORISATION WINDOW
        </text>

        {STAGES.map((s, i) => {
          const next = STAGES[i + 1]
          return (
          <g key={s.label}>
            <rect
              x={s.x}
              y={ROW_Y}
              width={s.w}
              height={H}
              rx={3}
              fill="var(--bg-raised)"
              stroke={i === 2 ? 'var(--red-dim)' : 'var(--border-strong)'}
              strokeWidth="var(--stroke-thin)"
            />
            <text
              x={s.x + s.w / 2}
              y={ROW_Y + 14}
              textAnchor="middle"
              className="fill-[var(--fg)] font-display"
              style={{ fontSize: 9.5, fontWeight: 600 }}
            >
              {s.label}
            </text>
            <text
              x={s.x + s.w / 2}
              y={ROW_Y + 26}
              textAnchor="middle"
              className="fill-[var(--fg-muted)] font-mono"
              style={{ fontSize: 7.5 }}
            >
              {s.detail}
            </text>

            {next ? <Arrow from={s.x + s.w} to={next.x} y={ROW_Y + H / 2} /> : null}
          </g>
          )
        })}

        {/* The guards' own path to a decline. Drawn over the ensemble rather than
            through it, because a provable block does not need the model to agree
            and that independence is the whole reason the layers stay separate. */}
        <path
          d={`M ${244 + 128 / 2} ${ROW_Y} C ${420} ${16}, ${520} ${16}, ${560} ${ROW_Y - 2}`}
          fill="none"
          stroke="var(--red)"
          strokeWidth="var(--stroke)"
          strokeDasharray="3 2"
        />
        <polygon points="0,-3 5,0 0,3" fill="var(--red)"
                 transform={`translate(${562} ${ROW_Y - 1}) rotate(58)`} />
        <text x={470} y={12} textAnchor="middle" className="fill-[var(--red)] font-mono"
              style={{ fontSize: 7.5 }}>
          provable block — no model opinion required
        </text>

        {/* Off-path: everything that trains. */}
        <rect
          x={116}
          y={104}
          width={384}
          height={22}
          rx={3}
          fill="none"
          stroke="var(--border)"
          strokeWidth="var(--stroke-thin)"
          strokeDasharray="3 3"
        />
        <text x={308} y={118} textAnchor="middle" className="fill-[var(--fg-faint)] font-mono"
              style={{ fontSize: 7.5 }}>
          offline · training, calibration, the co-evolution loop — never on the payment path
        </text>
        <path d={`M 308 ${ROW_Y + H} L 308 104`} stroke="var(--border)"
              strokeWidth="var(--stroke-thin)" strokeDasharray="2 3" fill="none" />
      </svg>
    </div>
  )
}

function Arrow({ from, to, y }: { from: number; to: number; y: number }) {
  return (
    <g>
      <line x1={from} y1={y} x2={to - 5} y2={y} stroke="var(--border-strong)"
            strokeWidth="var(--stroke-thin)" />
      <polygon points="0,-3 5,0 0,3" fill="var(--border-strong)"
               transform={`translate(${to - 5} ${y})`} />
    </g>
  )
}

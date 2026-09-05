import type { Field } from '@/types/api'
import { auto } from '@/lib/format'

/**
 * The model's inputs for one payment, each against what that field looks like on
 * legitimate traffic.
 *
 * A bare feature value teaches nothing. `g_payee_in_degree = 41` is meaningless
 * unless you already know that the legitimate middle is 3 and the 95th
 * percentile is 9, at which point it is the whole story. So every numeric field
 * gets a small strip showing the benign 5th-to-95th range with the median
 * marked and this payment's value placed against it.
 *
 * The strip is a div with a percentage offset rather than a chart, because there
 * are sixty of these on screen at once and sixty SVG instances would cost more
 * than the entire rest of the panel.
 *
 * Values outside the benign range are marked, not coloured red. Being unusual is
 * not being fraudulent - most of the unusual values in this table belong to
 * legitimate payments - and colouring them as severity would teach the reader
 * something false.
 */

export function FieldTable({ blocks }: { blocks: Record<string, Field[]> }) {
  return (
    <div className="space-y-3">
      {Object.entries(blocks).map(([block, fields]) => (
        <div key={block}>
          <div className="label mb-1">{block.replace(/_/g, ' ')}</div>
          <table className="w-full table-fixed border-collapse">
            <colgroup>
              <col className="w-[44%]" />
              <col className="w-[20%]" />
              <col className="w-[36%]" />
            </colgroup>
            <tbody>
              {fields.map((f) => (
                <Row key={f.column} field={f} />
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}

function Row({ field }: { field: Field }) {
  const value = typeof field.value === 'number' ? field.value : null
  const { p05, p95, median } = {
    p05: field.benign_p05,
    p95: field.benign_p95,
    median: field.benign_median,
  }

  const comparable =
    value !== null && p05 !== null && p95 !== null && median !== null && p95 > p05

  return (
    <tr className="border-b border-line-subtle last:border-0">
      <td className="truncate py-0.5 pr-2 font-mono text-2xs text-fg-muted" title={field.column}>
        {field.column}
      </td>
      <td className="num py-0.5 pr-2 text-right text-2xs text-fg">
        {typeof field.value === 'number'
          ? auto(field.value)
          : field.value === null || field.value === undefined
            ? '—'
            : String(field.value)}
      </td>
      <td className="py-0.5">
        {comparable ? (
          <Strip value={value} p05={p05} p95={p95} median={median} column={field.column} />
        ) : (
          <span className="text-2xs text-fg-faint">
            {value === null ? '' : 'no benign reference'}
          </span>
        )}
      </td>
    </tr>
  )
}

function Strip({
  value,
  p05,
  p95,
  median,
  column,
}: {
  value: number
  p05: number
  p95: number
  median: number
  column: string
}) {
  const span = p95 - p05
  // Clamped so an extreme outlier still renders inside the strip. The clamp is
  // visible - the marker sits hard against the edge and gets an arrow - so a
  // pinned value never reads as "just at the boundary".
  const raw = (value - p05) / span
  const clamped = Math.max(0, Math.min(1, raw))
  const medianAt = Math.max(0, Math.min(1, (median - p05) / span))
  const outside = raw < 0 || raw > 1

  return (
    <div
      className="relative h-3"
      title={`${column}: this payment ${auto(value)}. Legitimate traffic: 5th percentile ${auto(
        p05,
      )}, median ${auto(median)}, 95th percentile ${auto(p95)}.`}
    >
      <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-line-subtle" />
      <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-line" />
      <div
        className="absolute top-1/2 h-2 w-px -translate-y-1/2 bg-fg-faint"
        style={{ left: `${medianAt * 100}%` }}
      />
      <div
        className={`absolute top-1/2 h-2.5 w-[3px] -translate-x-1/2 -translate-y-1/2 rounded-sm ${
          outside ? 'bg-accent-strong' : 'bg-accent'
        }`}
        style={{ left: `${clamped * 100}%` }}
      />
      {outside ? (
        <span
          className="absolute top-1/2 -translate-y-1/2 font-mono text-[9px] leading-none text-accent-strong"
          style={raw > 1 ? { right: -2 } : { left: -2 }}
        >
          {raw > 1 ? '›' : '‹'}
        </span>
      ) : null}
    </div>
  )
}

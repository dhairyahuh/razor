import { memo } from 'react'
import type { ScoredTxn } from '@/types/api'
import { OUTCOMES, outcomeOf, VERDICTS, verdictOf } from '@/lib/decision'
import { clock, inr, score as fmtScore, shortId } from '@/lib/format'

/**
 * The stream itself.
 *
 * A plain table rather than a virtualised one, and that is a considered choice:
 * the buffer is capped at a few hundred rows, every one of which is mounting
 * and unmounting continuously, and a virtualiser measuring rows that change
 * eight times a second costs more than it saves. The virtualiser earns its
 * place on the full 280,000-row browser, which is a different component.
 *
 * Colour carries meaning and only meaning. Rows tint by outcome, faintly - a
 * saturated row is unreadable and a table where everything is coloured is a
 * table where nothing is. The strongest treatment is reserved for a provable
 * block, because that is the distinction the backend exists to make.
 */

interface Props {
  rows: ScoredTxn[]
  onSelect: (txnId: string) => void
  selected: string | null
}

export function StreamTable({ rows, onSelect, selected }: Props) {
  return (
    <div className="h-full overflow-hidden">
      <table className="w-full table-fixed border-collapse">
        <colgroup>
          <col className="w-1.5" />
          <col className="w-[9%]" />
          <col className="w-[16%]" />
          <col className="w-[13%]" />
          <col className="w-[10%]" />
          <col className="w-[9%]" />
          <col className="w-[8%]" />
          <col className="w-[18%]" />
          <col className="w-[8%]" />
        </colgroup>
        <thead>
          <tr className="border-b border-line-subtle bg-ground">
            <th />
            <Th>time</Th>
            <Th>payer</Th>
            <Th align="right">amount</Th>
            <Th>rail</Th>
            <Th>channel</Th>
            <Th align="right">score</Th>
            <Th>decision</Th>
            <Th align="center" title="Whether the system was right, known only because this is a labelled test window.">
              truth
            </Th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <Row
              key={`${row.txn_id}-${i}`}
              row={row}
              // Only the newest row animates in. Animating all of them would
              // re-trigger on every flush and make the table shimmer.
              fresh={i === 0}
              selected={row.txn_id === selected}
              onSelect={onSelect}
            />
          ))}
        </tbody>
      </table>

      {rows.length === 0 ? (
        <p className="p-6 text-center text-xs text-fg-faint">
          Waiting for the first payment…
        </p>
      ) : null}
    </div>
  )
}

const Row = memo(function Row({
  row,
  fresh,
  selected,
  onSelect,
}: {
  row: ScoredTxn
  fresh: boolean
  selected: boolean
  onSelect: (id: string) => void
}) {
  const outcome = outcomeOf(row)
  const style = OUTCOMES[outcome]
  const verdict = verdictOf(row)
  const v = VERDICTS[verdict]

  return (
    <tr
      onClick={() => onSelect(row.txn_id)}
      title={style.meaning}
      className={`cursor-pointer border-b border-line-subtle transition-colors hover:bg-hover ${style.row} ${
        selected ? 'bg-selected' : ''
      } ${fresh ? 'animate-row-in' : ''}`}
    >
      <td className={`${style.dot}`} aria-hidden="true" />
      <Td className="text-fg-muted">{clock(row.timestamp)}</Td>
      <Td className="text-fg-secondary">{shortId(row.payer_id, 10, 3)}</Td>
      <Td align="right" className="text-fg">
        {inr(row.amount)}
      </Td>
      <Td className="text-fg-muted">{row.rail}</Td>
      <Td className="truncate text-fg-muted">{row.channel ?? '—'}</Td>
      <Td align="right" className={row.model_alert === 1 ? style.fg : 'text-fg-secondary'}>
        {fmtScore(row.score)}
      </Td>
      <Td className={`truncate font-medium ${style.fg}`}>{style.label}</Td>
      <Td align="center" className={v.fg} title={v.label}>
        {v.short}
      </Td>
    </tr>
  )
})

function Th({
  children,
  align = 'left',
  title,
}: {
  children: React.ReactNode
  align?: 'left' | 'right' | 'center'
  title?: string
}) {
  return (
    <th
      title={title}
      className="label px-2 py-1.5 font-normal"
      style={{ textAlign: align }}
    >
      {children}
    </th>
  )
}

function Td({
  children,
  align = 'left',
  className = '',
  title,
}: {
  children: React.ReactNode
  align?: 'left' | 'right' | 'center'
  className?: string
  /** Hover text. The decision column abbreviates to a glyph and needs to expand. */
  title?: string
}) {
  return (
    <td
      title={title}
      className={`num overflow-hidden text-ellipsis whitespace-nowrap px-2 py-1 text-2xs ${className}`}
      style={{ textAlign: align }}
    >
      {children}
    </td>
  )
}

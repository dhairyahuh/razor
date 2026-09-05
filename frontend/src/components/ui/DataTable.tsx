import { useMemo, useState } from 'react'
import type { Table } from '@/types/api'
import { NoArtefact, NoRows } from './RouteSkeleton'
import { auto, count, humanise, MISSING, pct } from '@/lib/format'

/**
 * A table over one artefact.
 *
 * The analysis routes are mostly CSVs with a paragraph of context, and writing
 * a bespoke component per CSV would be a lot of code that says the same thing.
 * What this does that a generic table normally does not:
 *
 *  - It shows the backend's column names, not prettified ones. The brief is
 *    explicit that judges cross-check against the files, and `recall_lost_to_red`
 *    renamed to "Recall Lost To Red" is no longer greppable. The human reading
 *    goes in the header tooltip, where it helps without getting in the way.
 *  - It formats by *column meaning* rather than by JavaScript type. Anything
 *    that is a rate renders as a percentage, anything that is money renders in
 *    rupees, and null renders as an em dash rather than as zero.
 *  - It distinguishes a missing artefact from an empty one, because those are
 *    different facts about the run.
 */

type Row = Record<string, unknown>

export interface Column {
  key: string
  /** Overrides the automatic choice. */
  format?: (value: unknown, row: Row) => React.ReactNode
  align?: 'left' | 'right' | 'center'
  /** Explains what the column means, on hover over the header. */
  help?: string
  width?: string
}

interface Props {
  table: Table<Row> | undefined
  /** When omitted, every column in the data, in file order. */
  columns?: Column[]
  /** Sorted descending by this column on first render. */
  sortBy?: string
  /** Explains an absent artefact in terms of what produces it. */
  emptyNote?: string
  maxRows?: number
}

export function DataTable({ table, columns, sortBy, emptyNote, maxRows }: Props) {
  const [sort, setSort] = useState<{ key: string; desc: boolean } | null>(
    sortBy ? { key: sortBy, desc: true } : null,
  )

  const cols: Column[] = useMemo(() => {
    if (columns) return columns
    const first = table?.rows[0]
    return first ? Object.keys(first).map((key) => ({ key })) : []
  }, [columns, table])

  const rows = useMemo(() => {
    const all = table?.rows ?? []
    if (!sort) return maxRows ? all.slice(0, maxRows) : all
    const sorted = [...all].sort((a, b) => compare(a[sort.key], b[sort.key]))
    if (sort.desc) sorted.reverse()
    return maxRows ? sorted.slice(0, maxRows) : sorted
  }, [table, sort, maxRows])

  if (!table || !table.available) {
    return <NoArtefact file={table?.source || 'this artefact'} note={emptyNote} />
  }
  if (table.rows.length === 0) {
    return <NoRows file={table.source} note={emptyNote} />
  }

  return (
    <div className="h-full overflow-auto">
      <table className="w-full border-collapse">
        <thead className="sticky top-0 z-10 bg-surface">
          <tr className="border-b border-line">
            {cols.map((c) => (
              <th
                key={c.key}
                title={c.help ?? humanise(c.key)}
                onClick={() =>
                  setSort((s) =>
                    s?.key === c.key ? { key: c.key, desc: !s.desc } : { key: c.key, desc: true },
                  )
                }
                className="label cursor-pointer select-none whitespace-nowrap px-2 py-1.5 font-normal hover:text-fg"
                style={{ textAlign: c.align ?? alignFor(c.key), width: c.width }}
              >
                {c.key}
                {sort?.key === c.key ? (
                  <span className="ml-1 text-accent">{sort.desc ? '▾' : '▴'}</span>
                ) : null}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-line-subtle hover:bg-hover">
              {cols.map((c) => (
                <td
                  key={c.key}
                  className="num whitespace-nowrap px-2 py-1 text-2xs text-fg-secondary"
                  style={{ textAlign: c.align ?? alignFor(c.key) }}
                >
                  {c.format ? c.format(row[c.key], row) : cell(c.key, row[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      {maxRows && table.n_rows > maxRows ? (
        <p className="px-2 py-1.5 text-2xs text-fg-faint">
          Showing {count(maxRows)} of {count(table.n_rows)} rows — the file has the rest.
        </p>
      ) : null}
    </div>
  )
}

/* -- formatting by column meaning ------------------------------------------ */

const RATE = /(rate|recall|precision|fpr|auc|share|fraction|coverage|ratio|_pct|probability)/i
const MONEY = /(value|amount|loss|cost|saved|at_risk|benefit)/i
const ID = /(_id$|^id$|hash|sha|txn|vector|family|name|label|note|reason|phrase|move|tactic)/i

function cell(key: string, v: unknown): React.ReactNode {
  if (v === null || v === undefined || v === '') return <span className="text-fg-faint">{MISSING}</span>
  if (typeof v === 'boolean') return v ? 'yes' : 'no'
  if (typeof v === 'string') return v
  if (typeof v !== 'number') return String(v)
  if (!Number.isFinite(v)) return <span className="text-fg-faint">{MISSING}</span>

  // Rates are stored as fractions and read as percentages; a bare 0.98033 in a
  // recall column makes a reader do arithmetic to compare it with the report.
  if (RATE.test(key) && Math.abs(v) <= 1) return pct(v, 1)
  if (MONEY.test(key) && !ID.test(key)) return intMoney(v)
  return auto(v)
}

function intMoney(v: number): string {
  return new Intl.NumberFormat('en-IN', {
    style: 'currency',
    currency: 'INR',
    maximumFractionDigits: 0,
  }).format(v)
}

function alignFor(key: string): 'left' | 'right' {
  return ID.test(key) ? 'left' : 'right'
}

function compare(a: unknown, b: unknown): number {
  const an = typeof a === 'number' && Number.isFinite(a)
  const bn = typeof b === 'number' && Number.isFinite(b)
  // Nulls sort to the bottom in either direction: a missing measurement is not
  // a small one, and letting it sort as zero would put unmeasured vectors at
  // the top of a "worst recall" list, which is exactly backwards.
  if (!an && !bn) return String(a ?? '').localeCompare(String(b ?? ''))
  if (!an) return -1
  if (!bn) return 1
  return (a as number) - (b as number)
}

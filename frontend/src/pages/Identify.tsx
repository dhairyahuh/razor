import { useMemo, useState } from 'react'
import { Panel } from '@/components/ui/Panel'
import { DataTable } from '@/components/ui/DataTable'
import { PanelLoading } from '@/components/ui/RouteSkeleton'
import { useIdentify } from '@/api/queries'
import { count, inrShort, withInterval } from '@/lib/format'
import type { Vector } from '@/types/api'

/**
 * `/identify` — the taxonomy, and what of it has actually been tested.
 *
 * The kill-chain matrix is the point. A list of forty-six attack vectors is a
 * document; the same vectors placed against the stages of a payment attack, and
 * shaded by whether the defence has been *measured* against them, is a map of
 * where the coverage is and - more usefully - where it is not.
 *
 * The distinction the matrix has to carry is between three states that a list
 * cannot express: a vector that is documented only, a vector that is simulated
 * but has too few rows to measure, and a vector with a real measured recall.
 * Collapsing the middle one into either neighbour would overstate the coverage
 * in one direction or the gaps in the other.
 */
export default function Identify() {
  const { data, isLoading } = useIdentify()
  const [selected, setSelected] = useState<string | null>(null)

  const vector = data?.vectors.find((v) => v.id === selected) ?? null

  return (
    <div className="grid h-full min-h-0 grid-cols-[1fr_340px] grid-rows-[1fr_auto] gap-2 p-2">
      <Panel
        id="tour-killchain"
        title="Kill-chain coverage"
        source="src/redteam/identify/attack_library.yaml"
        subtitle={
          data ? (
            <>
              {count(data.counts.total)} vectors · {count(data.counts.simulated)} simulated ·{' '}
              {count(data.counts.measured)} with a measured recall
            </>
          ) : null
        }
        caveat="Shading is measured recall, joined from defend_per_vector_recall.csv. An unshaded cell is a vector this run never simulated — not a vector the defence failed on. Those are different, and the residual-risk table below ranks them accordingly."
        scroll
      >
        {isLoading ? (
          <PanelLoading />
        ) : (
          <KillChain
            stages={data?.kill_chain_stages ?? []}
            vectors={data?.vectors ?? []}
            selected={selected}
            onSelect={setSelected}
          />
        )}
      </Panel>

      <Panel
        title={vector ? vector.name : 'Vector detail'}
        source="src/redteam/identify/attack_library.yaml"
        subtitle={vector ? vector.id : 'Pick a cell in the matrix.'}
        scroll
      >
        {vector ? <Detail vector={vector} /> : <Hint />}
      </Panel>

      <Panel
        title="Residual risk"
        source="src/redteam/identify/attack_library.yaml + defend_per_vector_recall.csv"
        subtitle="Ranked by a-priori risk that this run has not shown the defence covers."
        caveat="Residual is the library's risk score discounted by measured recall. A vector never simulated keeps its full risk, because nothing has been shown about it — which is why the top of this table is mostly things we did not test rather than things we failed."
        className="col-span-2 max-h-[38%] min-h-[180px]"
        scroll
      >
        {isLoading ? (
          <PanelLoading />
        ) : (
          <DataTable
            table={{
              source: 'residual_risk',
              available: true,
              n_rows: data?.residual_risk.length ?? 0,
              rows: (data?.residual_risk ?? []) as unknown as Record<string, unknown>[],
            }}
            sortBy="residual_risk"
            columns={[
              { key: 'id', align: 'left', width: '24%' },
              { key: 'name', align: 'left', width: '20%' },
              { key: 'family', align: 'left' },
              { key: 'risk_score', help: 'The library’s a-priori risk. Not a measurement.' },
              {
                key: 'measured_recall',
                help: 'Null where this run never measured the vector',
              },
              {
                key: 'residual_risk',
                help: 'risk_score × (1 − measured_recall), or the full risk score where there is no measurement',
              },
              { key: 'basis', align: 'left', width: '24%' },
            ]}
          />
        )}
      </Panel>
    </div>
  )
}

function KillChain({
  stages,
  vectors,
  selected,
  onSelect,
}: {
  stages: string[]
  vectors: Vector[]
  selected: string | null
  onSelect: (id: string) => void
}) {
  const families = useMemo(() => {
    const map = new Map<string, Vector[]>()
    for (const v of vectors) {
      const list = map.get(v.family) ?? []
      list.push(v)
      map.set(v.family, list)
    }
    return [...map.entries()].sort((a, b) => a[0].localeCompare(b[0]))
  }, [vectors])

  return (
    <div className="min-w-max p-2">
      <div
        className="grid gap-px"
        style={{ gridTemplateColumns: `140px repeat(${stages.length}, minmax(110px, 1fr))` }}
      >
        <div />
        {stages.map((s) => (
          <div key={s} className="label px-1 pb-1 text-center">
            {s.replace(/_/g, ' ')}
          </div>
        ))}

        {families.map(([family, members]) => (
          <FamilyRow
            key={family}
            family={family}
            members={members}
            stages={stages}
            selected={selected}
            onSelect={onSelect}
          />
        ))}
      </div>
    </div>
  )
}

function FamilyRow({
  family,
  members,
  stages,
  selected,
  onSelect,
}: {
  family: string
  members: Vector[]
  stages: string[]
  selected: string | null
  onSelect: (id: string) => void
}) {
  return (
    <>
      <div className="flex items-center px-1 py-0.5 font-mono text-2xs text-fg-muted">
        {family.replace(/_/g, ' ')}
      </div>
      {stages.map((stage) => {
        const here = members.filter((m) => m.kill_chain.includes(stage))
        return (
          <div key={stage} className="flex flex-wrap content-start gap-px p-0.5">
            {here.map((v) => (
              <Cell key={v.id} vector={v} selected={v.id === selected} onSelect={onSelect} />
            ))}
          </div>
        )
      })}
    </>
  )
}

function Cell({
  vector,
  selected,
  onSelect,
}: {
  vector: Vector
  selected: boolean
  onSelect: (id: string) => void
}) {
  const r = vector.measured_recall
  const measured = r !== null

  // Three states, three treatments. Shading only runs across the measured ones,
  // so an unmeasured vector cannot be mistaken for a poorly-performing one.
  const tone = !vector.simulated
    ? 'border-line-subtle bg-transparent text-fg-faint'
    : !measured
      ? 'border-dashed border-line bg-transparent text-fg-muted'
      : r >= 0.9
        ? 'border-green-dim bg-green-wash text-green'
        : r >= 0.6
          ? 'border-amber-dim bg-amber-wash text-amber'
          : 'border-red-dim bg-red-wash text-red'

  const state = !vector.simulated
    ? 'documented only — this run never simulated it'
    : !measured
      ? 'simulated, but no rows reached the test window'
      : `measured recall ${withInterval(r, vector.recall_lo95, vector.recall_hi95)}${
          vector.n_sufficient ? '' : ' — too few rows to be interpretable'
        }`

  return (
    <button
      type="button"
      onClick={() => onSelect(vector.id)}
      title={`${vector.name}\n${state}`}
      className={`h-4 w-4 rounded-sm border transition-transform hover:scale-125 ${tone} ${
        selected ? 'ring-1 ring-accent ring-offset-1 ring-offset-surface' : ''
      }`}
    >
      <span className="sr-only">{vector.name}</span>
    </button>
  )
}

function Detail({ vector }: { vector: Vector }) {
  return (
    <div className="space-y-3 p-3">
      <p className="text-xs leading-relaxed text-fg-secondary">{vector.description}</p>

      <div className="grid grid-cols-2 gap-3">
        <Figure
          label="measured recall"
          value={withInterval(vector.measured_recall, vector.recall_lo95, vector.recall_hi95)}
          note={
            vector.measured_recall === null
              ? 'not measured in this run'
              : vector.n_sufficient
                ? `over ${count(vector.measured_rows)} rows`
                : `only ${count(vector.measured_rows)} rows — read with care`
          }
        />
        <Figure
          label="value at risk"
          value={inrShort(vector.value_at_risk)}
          note="in the test window"
        />
        <Figure label="library risk" value={vector.risk_score.toFixed(2)} note="a-priori, not measured" />
        <Figure label="severity" value={String(vector.severity)} note={`prevalence ${vector.prevalence}`} />
      </div>

      <Chips label="kill chain" items={vector.kill_chain} />
      <Chips label="rails" items={vector.rails} />
      <Chips label="channels" items={vector.channels} />
      <Chips label="signals the defence uses" items={vector.signals} />
      <Chips label="controls" items={vector.controls} />
      {vector.genai_enablers.length ? (
        <Chips label="what GenAI changes" items={vector.genai_enablers} />
      ) : null}

      {vector.liability_note ? (
        <div className="rounded border border-line-subtle bg-raised p-2">
          <div className="label mb-1">liability</div>
          <p className="text-2xs leading-relaxed text-fg-muted">{vector.liability_note}</p>
        </div>
      ) : null}

      <p className="text-2xs text-fg-faint">
        {vector.simulated ? (
          <>
            Simulated by <span className="font-mono">{vector.generator}</span>.
          </>
        ) : (
          'Documented in the library but not simulated by this run. It contributes to residual risk at full weight.'
        )}
      </p>
    </div>
  )
}

function Figure({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="num text-sm text-fg">{value}</div>
      <div className="text-2xs text-fg-faint">{note}</div>
    </div>
  )
}

function Chips({ label, items }: { label: string; items: string[] }) {
  if (!items?.length) return null
  return (
    <div>
      <div className="label mb-1">{label}</div>
      <div className="flex flex-wrap gap-1">
        {items.map((i) => (
          <span
            key={i}
            className="rounded border border-line-subtle bg-raised px-1.5 py-0.5 font-mono text-2xs text-fg-muted"
          >
            {i}
          </span>
        ))}
      </div>
    </div>
  )
}

function Hint() {
  return (
    <div className="space-y-2 p-3">
      <p className="text-2xs leading-relaxed text-fg-muted">
        Each square is one attack vector, placed at every kill-chain stage it touches.
      </p>
      <ul className="space-y-1 text-2xs text-fg-faint">
        <Legend className="border-green-dim bg-green-wash" label="measured recall ≥ 90%" />
        <Legend className="border-amber-dim bg-amber-wash" label="measured recall 60–90%" />
        <Legend className="border-red-dim bg-red-wash" label="measured recall below 60%" />
        <Legend className="border-dashed border-line" label="simulated, nothing reached the test window" />
        <Legend className="border-line-subtle" label="documented only — never simulated" />
      </ul>
      <p className="text-2xs leading-relaxed text-fg-faint">
        Recall shading is joined from <span className="font-mono">defend_per_vector_recall.csv</span>{' '}
        — the same numbers as <span className="font-mono">/defend</span>, not a second
        calculation.
      </p>
    </div>
  )
}

function Legend({ className, label }: { className: string; label: string }) {
  return (
    <li className="flex items-center gap-2">
      <span className={`h-3 w-3 shrink-0 rounded-sm border ${className}`} />
      {label}
    </li>
  )
}

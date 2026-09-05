import { useEffect, useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api, ApiError } from '@/api/client'
import { useIdentify } from '@/api/queries'
import { useTour } from '@/features/tour/useTour'
import { pct, pp, count } from '@/lib/format'

/**
 * RUN A ROUND NOW.
 *
 * What this actually does, stated plainly because the alternative is a button
 * that implies more than it delivers: it takes one attack vector, generates a
 * fresh campaign for it against a simulated benign backdrop, applies the
 * evasion levers you selected, and scores both the original and the mutated
 * campaign through the **deployed model**. What comes back is recall before and
 * after, measured on the same rows.
 *
 * That is the red half of a co-evolution round, live, in a few seconds.
 *
 * It is not the blue half. Blue's response - retrain on the survivors, choose a
 * countermeasure, re-measure - is minutes of model fitting, and putting minutes
 * on the critical path of a four-minute demo would be a design error rather
 * than an ambition. The recorded rounds beside this panel are where blue's half
 * is, measured properly. Saying which is which is the whole reason this
 * paragraph exists in the interface as well as in this comment.
 */

/** The evasion levers the backend exposes. Names must match `EVASION_MOVES`. */
const LEVERS = [
  { id: 'structure_amounts', label: 'Structure amounts', note: 'split below reporting bands' },
  { id: 'age_the_payee', label: 'Age the payee', note: 'buy an account with history' },
  { id: 'slow_the_burst', label: 'Slow the burst', note: 'spread the campaign over time' },
  { id: 'spread_the_ring', label: 'Spread the ring', note: 'more mules, fewer each' },
  { id: 'mimic_hours', label: 'Mimic hours', note: 'move to normal transaction times' },
]

export function RunRound() {
  const identify = useIdentify()
  const [vector, setVector] = useState('')
  const [levers, setLevers] = useState<string[]>(['structure_amounts', 'age_the_payee'])

  // Only vectors the pipeline can actually generate. Offering a documented-only
  // vector would produce a 422 and teach the reader that the button is flaky
  // rather than that the vector is unsimulated.
  const options = (identify.data?.vectors ?? []).filter((v) => v.simulated)
  const chosen = vector || options[0]?.id || ''

  const round = useMutation({
    mutationFn: () => api.attack({ vector_id: chosen, levers, rows: 400 }),
  })

  // The tour fires this rather than narrating over a button nobody pressed. If
  // there is no backend the mutation fails and the panel explains why, which is
  // a better step than a caption describing something that did not happen.
  const fire = round.mutate
  useEffect(() => {
    return useTour.getState().register('run-round', () => {
      if (chosen) fire()
    })
  }, [chosen, fire])

  const result = round.data as
    | {
        vector_id: string
        rows: number
        recall_before: number
        recall_after: number
        recall_lost: number
        still_caught: number
        note: string
      }
    | undefined

  return (
    <div className="flex h-full flex-col gap-3 p-3">
      <div className="flex flex-wrap items-end gap-2">
        <label className="flex flex-col gap-1">
          <span className="label">attack vector</span>
          <select
            value={chosen}
            onChange={(e) => setVector(e.target.value)}
            disabled={options.length === 0}
            className="w-56 rounded border border-line bg-raised px-2 py-1 font-mono text-2xs text-fg"
          >
            {options.map((v) => (
              <option key={v.id} value={v.id}>
                {v.id}
              </option>
            ))}
          </select>
        </label>

        <button
          type="button"
          onClick={() => round.mutate()}
          disabled={round.isPending || !chosen}
          className="rounded border border-red-dim bg-red-wash px-3 py-1.5 font-display text-2xs font-semibold uppercase tracking-wider text-red transition-colors hover:bg-red-dim hover:text-fg-inverse disabled:opacity-50"
        >
          {round.isPending ? 'Running…' : 'Run a round now'}
        </button>
      </div>

      <fieldset className="flex flex-wrap gap-1.5">
        <legend className="label mb-1">evasion levers</legend>
        {LEVERS.map((l) => {
          const on = levers.includes(l.id)
          return (
            <button
              key={l.id}
              type="button"
              title={l.note}
              onClick={() =>
                setLevers((s) => (on ? s.filter((x) => x !== l.id) : [...s, l.id]))
              }
              className={`rounded border px-2 py-1 text-2xs transition-colors ${
                on
                  ? 'border-red-dim bg-red-wash text-red'
                  : 'border-line text-fg-muted hover:bg-hover'
              }`}
            >
              {l.label}
            </button>
          )
        })}
      </fieldset>

      <div className="min-h-0 flex-1">
        {round.isError ? (
          <Unavailable error={round.error} />
        ) : result ? (
          <Result result={result} />
        ) : (
          <p className="text-2xs leading-relaxed text-fg-muted">
            Pick a vector and some levers, then run it. The campaign is generated fresh and
            scored through the deployed model twice — once as generated, once mutated — so the
            difference between the two numbers is attributable to the levers rather than to
            sampling.
          </p>
        )}
      </div>

      <p className="border-t border-line-subtle pt-2 text-2xs leading-relaxed text-fg-faint">
        This is the <span className="text-fg-muted">red half</span> of a round. Blue's half —
        retrain on the survivors, pick a countermeasure, re-measure — is minutes of model
        fitting and is not run here. The recorded rounds above are where blue actually
        responded.
      </p>
    </div>
  )
}

function Result({
  result,
}: {
  result: {
    vector_id: string
    rows: number
    recall_before: number
    recall_after: number
    recall_lost: number
    still_caught: number
    note: string
  }
}) {
  const worked = result.recall_lost > 0

  return (
    <div className="space-y-2">
      <div className="flex items-baseline gap-4">
        <Figure label="recall before" value={pct(result.recall_before)} />
        <span className="text-fg-faint">→</span>
        <Figure
          label="recall after"
          value={pct(result.recall_after)}
          tone={worked ? 'text-red' : 'text-green'}
        />
        <Figure
          label="lost to red"
          value={pp(-result.recall_lost)}
          tone={worked ? 'text-red' : 'text-fg-muted'}
        />
      </div>

      <p className="text-xs text-fg-secondary">
        {worked ? (
          <>
            The levers moved <span className="num text-red">{pct(result.recall_lost)}</span> of
            this campaign past the deployed model. {count(result.still_caught)} of{' '}
            {count(result.rows)} payments were still caught.
          </>
        ) : (
          <>
            The levers cost the defence nothing on this vector — the model caught the mutated
            campaign as well as the original. That is a real result, and the interesting kind:
            it means the features this vector trips are not the ones the levers move.
          </>
        )}
      </p>

      <p className="text-2xs leading-relaxed text-fg-faint">{result.note}</p>
    </div>
  )
}

function Figure({ label, value, tone = 'text-fg' }: { label: string; value: string; tone?: string }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className={`num text-xl leading-none ${tone}`}>{value}</div>
    </div>
  )
}

function Unavailable({ error }: { error: unknown }) {
  const demo = error instanceof ApiError && error.status === 503
  return (
    <div className="rounded border border-line bg-raised p-3">
      <p className="text-xs text-fg-secondary">
        {demo
          ? 'This button computes — it fires a genome at the deployed model and needs one loaded.'
          : 'The round did not run.'}
      </p>
      <p className="mt-1 text-2xs leading-relaxed text-fg-faint">
        {demo ? (
          <>
            The console is reading a recorded run, so there is no model to attack. Start the
            API (<span className="font-mono">docker compose up</span>) and this becomes live.
            The recorded rounds above were produced by exactly this mechanism, run for real.
          </>
        ) : (
          <span className="font-mono">{error instanceof Error ? error.message : String(error)}</span>
        )}
      </p>
    </div>
  )
}

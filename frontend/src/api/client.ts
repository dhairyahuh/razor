/**
 * One transport, two sources.
 *
 * The console has to work in two situations that look identical to a component
 * and are completely different underneath:
 *
 *  - **Live.** A FastAPI process is up. Everything is read from the run's
 *    artefacts through HTTP, the interactive endpoints work, and a round of the
 *    loop can actually be executed.
 *  - **Demo.** There is no server - the app is static files, possibly opened
 *    over conference wi-fi that has already failed, possibly with no network at
 *    all. Every read resolves against JSON baked out of `artifacts/default_run/` at
 *    build time by `scripts/build_demo_payload.py`.
 *
 * The rule that keeps this honest: **the demo payload is generated from the same
 * endpoints it replaces.** It is a recording, not a fixture. Nobody hand-writes
 * a number into it, so a figure on screen in Demo Mode is the same figure the
 * live server would return for that run, and the failure mode of the recording
 * going stale is caught by regenerating rather than by noticing.
 *
 * Mode is decided once, by probing, and can be forced either way from the UI. A
 * live server that dies mid-demo falls back to the recording on the next read
 * instead of emptying the screen.
 */

import type {
  AttackResponse,
  DefendResponse,
  FidelityResponse,
  Health,
  IdentifyResponse,
  LoopResponse,
  ReportResponse,
  RunListing,
  ScoreRequest,
  ScoreResponse,
  SummaryResponse,
  TransactionDetail,
  TransactionsResponse,
} from '@/types/api'

export type Mode = 'live' | 'demo'

/** Where the baked payload lives, relative to the site root. */
const DEMO_ROOT = '/demo'

/** The run the committed demo payload was baked from. */
export const DEMO_RUN = 'default_run'

/* -- mode ------------------------------------------------------------------ */

let mode: Mode | null = null
let probe: Promise<Mode> | null = null

/**
 * Decide which source to read from, once.
 *
 * Two things can override the probe. `?mode=demo` in the URL forces the
 * recording even when a server is up - which is what you want when rehearsing,
 * because you should rehearse against exactly what will be on screen. And
 * `VITE_MODE` pins it at build time for the static bundle, so the packaged demo
 * never spends two seconds failing to reach a server that was never deployed.
 */
export function resolveMode(): Promise<Mode> {
  if (mode) return Promise.resolve(mode)
  if (probe) return probe

  const forced = forcedMode()
  if (forced) {
    mode = forced
    return Promise.resolve(forced)
  }

  probe = (async () => {
    try {
      // Short timeout on purpose. This runs before the first paint of data; a
      // judge watching a spinner for a dead backend is the worst opening frame
      // this app could have.
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 1200)
      const res = await fetch('/api/runs', { signal: controller.signal })
      clearTimeout(timer)
      mode = res.ok ? 'live' : 'demo'
    } catch {
      mode = 'demo'
    }
    return mode
  })()

  return probe
}

function forcedMode(): Mode | null {
  const env = import.meta.env.VITE_MODE
  if (env === 'demo' || env === 'live') return env
  if (typeof window !== 'undefined') {
    const q = new URLSearchParams(window.location.search).get('mode')
    if (q === 'demo' || q === 'live') return q
  }
  return null
}

/** The mode already resolved, or null before the probe lands. For display only. */
export function currentMode(): Mode | null {
  return mode
}

/** Drop to the recording for the rest of the session. */
export function fallToDemo(reason: string) {
  if (mode !== 'demo') {
    console.warn(`falling back to the recorded demo payload: ${reason}`)
    mode = 'demo'
  }
}

/* -- fetching -------------------------------------------------------------- */

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly path: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    // FastAPI puts the useful part in `detail`; the status alone tells a
    // reader nothing about which artefact was missing.
    let detail = res.statusText
    try {
      const body = await res.json()
      if (body?.detail) detail = String(body.detail)
    } catch {
      /* not JSON - the status text is all there is */
    }
    throw new ApiError(detail, res.status, url)
  }
  return res.json() as Promise<T>
}

/**
 * A read that works in either mode.
 *
 * `live` builds the HTTP path; `demo` names the file in the baked payload. When
 * a live read fails with anything other than a 404 - the server died, the
 * network went - it retries against the recording rather than surfacing the
 * error, because a stale-but-real number beats an error card in front of an
 * audience. A 404 is passed through: that is the server correctly saying the
 * run or transaction does not exist, and hiding it would be misleading.
 */
async function read<T>(live: string, demo: string): Promise<T> {
  const m = await resolveMode()

  if (m === 'live') {
    try {
      return await json<T>(live)
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) throw err
      fallToDemo(err instanceof Error ? err.message : String(err))
    }
  }

  return json<T>(`${DEMO_ROOT}/${demo}`)
}

/**
 * A read that only exists live - scoring a payment, firing an attack, running a
 * round. There is no recording of these because they are computations, not
 * artefacts, and a pre-recorded "result" of a button press would be a lie about
 * what the button does.
 */
async function interactive<T>(path: string, init?: RequestInit): Promise<T> {
  const m = await resolveMode()
  if (m === 'demo') {
    throw new ApiError(
      'This action runs a computation on the backend and needs the API. ' +
        'The console is reading a recorded run, so there is nothing to run it on.',
      503,
      path,
    )
  }
  return json<T>(path, init)
}

/* -- reads ----------------------------------------------------------------- */

export const api = {
  runs: () => read<RunListing[]>('/api/runs', 'runs.json'),

  summary: (run: string) =>
    read<SummaryResponse>(`/api/runs/${run}/summary`, `${run}/summary.json`),

  identify: (run: string) =>
    read<IdentifyResponse>(`/api/runs/${run}/identify`, `${run}/identify.json`),

  fidelity: (run: string) =>
    read<FidelityResponse>(`/api/runs/${run}/fidelity`, `${run}/fidelity.json`),

  defend: (run: string) =>
    read<DefendResponse>(`/api/runs/${run}/defend`, `${run}/defend.json`),

  loop: (run: string) => read<LoopResponse>(`/api/runs/${run}/loop`, `${run}/loop.json`),

  report: (run: string) =>
    read<ReportResponse>(`/api/runs/${run}/report`, `${run}/report.json`),

  /**
   * A page of the scored test window.
   *
   * Filtering is a server concern live - the file is ~280k rows and shipping it
   * to the browser to filter would be absurd. The recording holds one
   * pre-filtered slice big enough to stream from and to scroll convincingly,
   * and says so, because a filter control that silently does nothing is worse
   * than one that is visibly unavailable.
   */
  transactions: async (run: string, params: TxnQuery = {}): Promise<TransactionsResponse> => {
    const m = await resolveMode()
    if (m === 'live') {
      const q = new URLSearchParams()
      for (const [k, v] of Object.entries(params)) {
        if (v !== undefined && v !== null && v !== '') q.set(k, String(v))
      }
      try {
        return await json<TransactionsResponse>(`/api/runs/${run}/transactions?${q}`)
      } catch (err) {
        if (err instanceof ApiError && err.status === 404) throw err
        fallToDemo(err instanceof Error ? err.message : String(err))
      }
    }
    return demoTransactions(run, params)
  },

  transaction: (run: string, txnId: string) =>
    read<TransactionDetail>(
      `/api/runs/${run}/transactions/${encodeURIComponent(txnId)}`,
      `${run}/txn/${txnId}.json`,
    ),

  /* -- live only ----------------------------------------------------------- */

  health: () => interactive<Health>('/api/health'),

  score: (payload: ScoreRequest) =>
    interactive<ScoreResponse>('/api/score', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    }),

  attack: (payload: Record<string, unknown>) =>
    interactive<AttackResponse>('/api/attack', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    }),
}

export interface TxnQuery {
  cursor?: number
  limit?: number
  rail?: string
  vector?: string
  decision?: 'alert' | 'approve' | 'fraud' | 'missed' | 'false_positive'
  hard_negative?: boolean
  min_score?: number
  max_score?: number
  min_amount?: number
  max_amount?: number
}

/* -- the recorded transaction slice ---------------------------------------- */

/**
 * The demo payload holds one slice of the scored test set, already sampled to
 * keep every decision class and every simulated vector represented. Paging and
 * the filters that are cheap on an array are applied here so the table, the
 * stream and the filter chips all behave; the rest are disabled in the UI.
 */
let slice: Promise<TransactionsResponse> | null = null

async function demoTransactions(
  run: string,
  params: TxnQuery,
): Promise<TransactionsResponse> {
  slice ??= json<TransactionsResponse>(`${DEMO_ROOT}/${run}/transactions.json`)
  const all = await slice

  let rows = all.rows
  if (params.rail) rows = rows.filter((r) => r.rail === params.rail)
  if (params.vector) rows = rows.filter((r) => r.attack_vector_id === params.vector)
  if (params.hard_negative !== undefined) {
    rows = rows.filter((r) => Boolean(r.is_hard_negative) === params.hard_negative)
  }
  switch (params.decision) {
    case 'alert':
      rows = rows.filter((r) => r.decision === 1)
      break
    case 'approve':
      rows = rows.filter((r) => r.decision === 0)
      break
    case 'fraud':
      rows = rows.filter((r) => r.is_fraud === 1)
      break
    case 'missed':
      rows = rows.filter((r) => r.is_fraud === 1 && r.decision === 0)
      break
    case 'false_positive':
      rows = rows.filter((r) => r.is_fraud === 0 && r.decision === 1)
      break
  }
  if (params.min_score !== undefined) rows = rows.filter((r) => r.score >= params.min_score!)
  if (params.max_score !== undefined) rows = rows.filter((r) => r.score <= params.max_score!)
  if (params.min_amount !== undefined) rows = rows.filter((r) => r.amount >= params.min_amount!)
  if (params.max_amount !== undefined) rows = rows.filter((r) => r.amount <= params.max_amount!)

  const cursor = params.cursor ?? 0
  const limit = params.limit ?? 100
  const page = rows.slice(cursor, cursor + limit)
  const next = cursor + limit

  return {
    ...all,
    rows: page,
    total: rows.length,
    cursor,
    next_cursor: next < rows.length ? next : null,
  }
}

/* -- artefact links -------------------------------------------------------- */

/**
 * A URL that opens the artefact a panel's numbers came from, or null when there
 * is nothing to open. Demo Mode has no file server, so `SourceLink` renders the
 * filename as text rather than as a link that would 404.
 */
export function fileUrl(run: string, file: string): string | null {
  if (currentMode() !== 'live') return null
  return `/api/runs/${run}/file/${encodeURIComponent(file)}`
}

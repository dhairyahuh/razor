/**
 * TanStack Query bindings for the artefact endpoints.
 *
 * The defaults here matter more than the hooks. A completed run's artefacts are
 * immutable - the files on disk do not change while someone reads them - so
 * refetching on window focus, on reconnect, or after any staleness interval is
 * pure cost. Turning all of that off is what makes navigating between routes
 * instant after the first visit, which is most of what "feels fast" means for a
 * four-minute demo.
 *
 * The exception is `RUN A ROUND NOW`, which does change the run on disk. That
 * path invalidates explicitly rather than being covered by a polling interval.
 */

import { QueryClient, useQuery } from '@tanstack/react-query'
import { api, ApiError, type TxnQuery } from './client'
import { useRunId } from '@/state/session'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: Infinity,
      gcTime: Infinity,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
      // One retry, not three. A 404 for a run that does not exist should show
      // its empty state immediately; three exponential retries would leave a
      // spinner up for several seconds first.
      retry: (attempts, error) => {
        if (error instanceof ApiError && error.status === 404) return false
        return attempts < 1
      },
    },
  },
})

export const keys = {
  runs: ['runs'] as const,
  summary: (run: string) => ['summary', run] as const,
  identify: (run: string) => ['identify', run] as const,
  fidelity: (run: string) => ['fidelity', run] as const,
  defend: (run: string) => ['defend', run] as const,
  loop: (run: string) => ['loop', run] as const,
  report: (run: string) => ['report', run] as const,
  transactions: (run: string, q: TxnQuery) => ['transactions', run, q] as const,
  transaction: (run: string, id: string) => ['transaction', run, id] as const,
}

export function useRuns() {
  return useQuery({ queryKey: keys.runs, queryFn: api.runs })
}

export function useSummary() {
  const run = useRunId()
  return useQuery({ queryKey: keys.summary(run), queryFn: () => api.summary(run) })
}

/**
 * `enabled` exists for the command palette, which is mounted in the shell on
 * every route but only needs the vector and metric lists once someone opens it.
 * Fetching them eagerly would put a quarter of a megabyte in front of the
 * landing route's first paint to populate a list nobody has asked for yet.
 */
export function useIdentify(enabled = true) {
  const run = useRunId()
  return useQuery({ queryKey: keys.identify(run), queryFn: () => api.identify(run), enabled })
}

export function useFidelity() {
  const run = useRunId()
  return useQuery({ queryKey: keys.fidelity(run), queryFn: () => api.fidelity(run) })
}

export function useDefend(enabled = true) {
  const run = useRunId()
  return useQuery({ queryKey: keys.defend(run), queryFn: () => api.defend(run), enabled })
}

export function useLoop() {
  const run = useRunId()
  return useQuery({ queryKey: keys.loop(run), queryFn: () => api.loop(run) })
}

export function useReport() {
  const run = useRunId()
  return useQuery({ queryKey: keys.report(run), queryFn: () => api.report(run) })
}

export function useTransactions(query: TxnQuery) {
  const run = useRunId()
  return useQuery({
    queryKey: keys.transactions(run, query),
    queryFn: () => api.transactions(run, query),
    // Paging through a virtualised table should not blank the rows already on
    // screen while the next page arrives.
    placeholderData: (prev) => prev,
  })
}

export function useTransaction(txnId: string | null) {
  const run = useRunId()
  return useQuery({
    queryKey: keys.transaction(run, txnId ?? ''),
    queryFn: () => api.transaction(run, txnId as string),
    enabled: Boolean(txnId),
  })
}

/**
 * Everything derived from a run's files, dropped. Called after a loop round
 * writes new artefacts, which is the only thing in the application that makes
 * the cached copies wrong.
 */
export function invalidateRun(run: string) {
  return queryClient.invalidateQueries({
    predicate: (q) => q.queryKey.length > 1 && q.queryKey[1] === run,
  })
}

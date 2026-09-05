/**
 * Session state: which run is being read, and how the console is being shown.
 *
 * Deliberately small. Everything that is *data* belongs to TanStack Query,
 * which already handles caching, staleness and request de-duplication better
 * than a store would. What lives here is the handful of facts that are about
 * the viewing session rather than the run - and every one of them survives a
 * reload, because a demo that loses Presentation Mode when someone hits refresh
 * in front of an audience is a demo that has to be re-set-up on stage.
 */

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { DEMO_RUN, type Mode } from '@/api/client'

type Theme = 'dark' | 'light'

interface Session {
  /** The run every read is scoped to. */
  runId: string
  /** Resolved by probing on boot; null until then. */
  mode: Mode | null
  theme: Theme
  /** Type up a quarter, strokes thickened, contrast raised. For projectors. */
  presentation: boolean

  setRun: (run: string) => void
  setMode: (mode: Mode) => void
  setTheme: (theme: Theme) => void
  toggleTheme: () => void
  setPresentation: (on: boolean) => void
  togglePresentation: () => void
}

export const useSession = create<Session>()(
  persist(
    (set, get) => ({
      runId: DEMO_RUN,
      mode: null,
      theme: 'dark',
      presentation: false,

      setRun: (runId) => set({ runId }),
      setMode: (mode) => set({ mode }),

      setTheme: (theme) => {
        applyTheme(theme)
        set({ theme })
      },
      toggleTheme: () => get().setTheme(get().theme === 'dark' ? 'light' : 'dark'),

      setPresentation: (presentation) => {
        applyPresentation(presentation)
        set({ presentation })
      },
      togglePresentation: () => get().setPresentation(!get().presentation),
    }),
    {
      name: 'rt.session',
      // `mode` is excluded: it is a fact about the current environment, not a
      // preference. Persisting it would leave a judge who opened the packaged
      // demo first stuck in Demo Mode when they later start the API.
      partialize: (s) => ({
        runId: s.runId,
        theme: s.theme,
        presentation: s.presentation,
      }),
      onRehydrateStorage: () => (state) => {
        if (!state) return
        applyTheme(state.theme)
        applyPresentation(state.presentation)
      },
    },
  ),
)

/**
 * Both of these write to `data-*` on the root element rather than to a class,
 * because that is where the token layer switches. Keeping the DOM write here
 * means no component has to remember to do it, and the inline script in
 * index.html applies the same attributes before first paint so there is no
 * flash of the wrong theme on load.
 */
function applyTheme(theme: Theme) {
  if (typeof document === 'undefined') return
  document.documentElement.dataset.theme = theme
  try {
    localStorage.setItem('rt.theme', theme)
  } catch {
    /* private browsing */
  }
}

function applyPresentation(on: boolean) {
  if (typeof document === 'undefined') return
  if (on) document.documentElement.dataset.presentation = 'on'
  else delete document.documentElement.dataset.presentation
  try {
    localStorage.setItem('rt.presentation', on ? 'on' : 'off')
  } catch {
    /* private browsing */
  }
}

/** The run id alone, for the many components that need only that. */
export const useRunId = () => useSession((s) => s.runId)
export const useMode = () => useSession((s) => s.mode)

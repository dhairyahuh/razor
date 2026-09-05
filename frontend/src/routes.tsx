import { lazy, Suspense, type ReactNode } from 'react'
import { createBrowserRouter, type RouteObject } from 'react-router-dom'
import { AppShell } from './components/shell/AppShell'
import { RouteSkeleton } from './components/ui/RouteSkeleton'

/**
 * Routes.
 *
 * The console is eager and everything else is lazy. That split is about the
 * four-minute demo: the landing route has to paint fast on a conference laptop,
 * and the analysis routes carry the chart code, so bundling them together would
 * make the first screen pay for six screens nobody has navigated to yet.
 */

const Identify = lazy(() => import('./pages/Identify'))
const Generate = lazy(() => import('./pages/Generate'))
const Defend = lazy(() => import('./pages/Defend'))
const Loop = lazy(() => import('./pages/Loop'))
const Deploy = lazy(() => import('./pages/Deploy'))
const Evidence = lazy(() => import('./pages/Evidence'))

function deferred(node: ReactNode) {
  return <Suspense fallback={<RouteSkeleton />}>{node}</Suspense>
}

const children: RouteObject[] = [
  { index: true, lazy: async () => ({ Component: (await import('./pages/Console')).default }) },
  { path: 'identify', element: deferred(<Identify />) },
  { path: 'generate', element: deferred(<Generate />) },
  { path: 'defend', element: deferred(<Defend />) },
  { path: 'loop', element: deferred(<Loop />) },
  { path: 'deploy', element: deferred(<Deploy />) },
  { path: 'evidence', element: deferred(<Evidence />) },
]

export const router = createBrowserRouter([
  { path: '/', element: <AppShell />, children },
])

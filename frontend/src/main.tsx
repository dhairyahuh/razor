import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClientProvider } from '@tanstack/react-query'
import { RouterProvider } from 'react-router-dom'

import './styles/index.css'
import { router } from './routes'
import { queryClient } from './api/queries'
import { resolveMode } from './api/client'
import { useSession } from './state/session'

// Decide live-versus-demo before React mounts. Doing it here rather than in an
// effect means the shell's first paint already knows which one it is, so the
// mode badge does not flicker from "demo" to "live" in front of an audience.
resolveMode().then((mode) => useSession.getState().setMode(mode))

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
)

/**
 * Tour state.
 *
 * Two properties are non-negotiable and shape everything here.
 *
 * **It must complete unattended.** No step may wait on something that can fail
 * silently. A step whose action does not resolve advances on its hold anyway, so
 * a dead backend costs one under-illustrated step rather than a stalled demo in
 * front of a judge.
 *
 * **It must be interruptible at any point, into free exploration at that spot.**
 * Not "back to the beginning", not a confirmation dialog - any click anywhere
 * outside the tour's own controls stops it and leaves the judge exactly where
 * the tour had got to. That is the behaviour that makes it safe to start.
 */

import { create } from 'zustand'
import { SCRIPT } from './script'

interface TourState {
  running: boolean
  /** Index into SCRIPT. Meaningful only while running. */
  step: number
  /** Set by the console and loop routes so the tour can drive them. */
  handlers: Partial<Record<'open-inspector' | 'run-round', () => void>>

  start: () => void
  stop: () => void
  next: () => void
  back: () => void
  register: (name: 'open-inspector' | 'run-round', fn: () => void) => () => void
}

export const useTour = create<TourState>((set, get) => ({
  running: false,
  step: 0,
  handlers: {},

  start: () => set({ running: true, step: 0 }),
  stop: () => set({ running: false }),

  next: () => {
    const next = get().step + 1
    if (next >= SCRIPT.length) set({ running: false, step: 0 })
    else set({ step: next })
  },

  back: () => set({ step: Math.max(0, get().step - 1) }),

  /**
   * Routes hand the tour a way to drive them, and take it back on unmount. The
   * tour cannot reach into a route's local state - opening the inspector means
   * setting a `useState` inside `Console` - and a store of callbacks is a
   * smaller price than lifting that state up for the tour's benefit alone.
   */
  register: (name, fn) => {
    set((s) => ({ handlers: { ...s.handlers, [name]: fn } }))
    return () => {
      set((s) => {
        const rest = { ...s.handlers }
        delete rest[name]
        return { handlers: rest }
      })
    }
  },
}))

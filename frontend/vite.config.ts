import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'

// The API runs as a separate process (uvicorn on 8000). Proxying rather than pointing the
// client at an absolute origin keeps one base URL in the client for both dev and the
// packaged container, so nothing has to know which one it is running in.
const API = process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api': { target: API, changeOrigin: true },
      // The interactive endpoints kept their bare paths so the existing tests and CLI
      // still work; the client reaches them through the same proxy.
      '/score': { target: API, changeOrigin: true },
      '/attack': { target: API, changeOrigin: true },
      '/jobs': { target: API, changeOrigin: true },
      '/health': { target: API, changeOrigin: true },
    },
  },
  build: {
    // A judge on a tired laptop waits for this once. Splitting the heavy chart library out
    // of the entry chunk is what keeps the landing route fast, since only /defend and /loop
    // draw anything.
    //
    // This was previously three named groups - react, charts and data - and that produced
    // `Circular chunk: react -> data -> react` at build time. The warning was not cosmetic:
    // @tanstack/react-query imports React while React's own chunk ended up importing back
    // out of the data chunk, so the emitted modules initialised in an order where
    // `React.Children` was assigned onto an undefined export. The dev server never chunks,
    // so it worked locally and every production bundle - the container's included - threw
    // "Cannot set properties of undefined" before the first paint.
    //
    // The function form fixes it by construction rather than by reordering a list: charts
    // depends on vendor and nothing depends on charts, so the graph cannot cycle.
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (!id.includes('node_modules')) return undefined
          // recharts pulls a large tree of d3 packages; they belong with it, not in vendor.
          if (/[\\/]node_modules[\\/](recharts|d3-|victory-|internmap|delaunator|robust-predicates)/.test(id)) {
            return 'charts'
          }
          return 'vendor'
        },
      },
    },
    chunkSizeWarningLimit: 900,
  },
})

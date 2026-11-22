/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The FastAPI server: uvicorn app.main:app --reload. 127.0.0.1 rather than
// localhost, which can resolve to IPv6 (::1), where uvicorn doesn't listen.
const api = 'http://127.0.0.1:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // The page calls the API on its own origin and Vite passes these paths
    // on, so the browser never makes a cross-origin request and FastAPI
    // needs no CORS settings. `vite preview` uses the same proxy.
    proxy: {
      '/ask': api,
      '/health': api,
      '/documents': api,
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    unstubGlobals: true, // undo vi.stubGlobal('fetch', ...) after each test
  },
})

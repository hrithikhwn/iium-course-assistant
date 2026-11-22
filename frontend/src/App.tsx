import { useEffect, useState } from 'react'
import { health } from './api.ts'

// A placeholder that only checks the API through Vite's proxy. The real page replaces it.
export default function App() {
  const [status, setStatus] = useState('Checking the API…')

  useEffect(() => {
    const controller = new AbortController()
    health(controller.signal)
      .then((h) => setStatus(`API ok, model ${h.model}`))
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setStatus(e instanceof Error ? e.message : String(e))
      })
    // in development StrictMode runs each effect twice; the first request is cancelled here
    return () => controller.abort()
  }, [])

  return (
    <main>
      <h1>IIUM Course Assistant</h1>
      <p role="status">{status}</p>
    </main>
  )
}

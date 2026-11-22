import { render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import { NOT_RUNNING } from './api.ts'
import App from './App.tsx'

test('shows the model the API runs', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => Response.json({ status: 'ok', model: 'qwen2.5:7b' })))
  render(<App />)
  expect(await screen.findByText('API ok, model qwen2.5:7b')).toBeInTheDocument()
})

test('says how to start the API when it is not running', async () => {
  // Vite's dev proxy answers 502 while uvicorn isn't running
  vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 502, statusText: 'Bad Gateway' })))
  render(<App />)
  expect(await screen.findByText(NOT_RUNNING)).toBeInTheDocument()
})

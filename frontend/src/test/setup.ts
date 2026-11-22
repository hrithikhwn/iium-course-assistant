// Runs before every test file (vite.config.ts: test.setupFiles).
import '@testing-library/jest-dom/vitest' // expect(...).toBeInTheDocument() and the like
import { cleanup } from '@testing-library/react'
import { afterEach } from 'vitest'

// Remove what each test rendered, so the next test starts from an empty page.
afterEach(() => cleanup())

// jsdom doesn't lay pages out, so it can't scroll (it would log "Not
// implemented"): the page's scroll to the newest message does nothing in tests.
window.scrollTo = () => {}

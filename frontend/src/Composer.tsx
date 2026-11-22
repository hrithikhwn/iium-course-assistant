import type { FormEvent, KeyboardEvent } from 'react'

interface Props {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  busy: boolean // the API is still answering: no new question until it's done
}

/** The message box at the bottom of the page. */
export default function Composer({ value, onChange, onSend, busy }: Props) {
  const ready = !busy && value.trim().length >= 3 // the API takes 3 to 1000 characters

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault() // a form's default is to load a new page
    if (ready) onSend()
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends and Shift+Enter starts a new line. While an input method is
    // composing a word, Enter picks the word instead.
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      event.currentTarget.form?.requestSubmit() // the same path as clicking the button
    }
  }

  return (
    <form className="composer" onSubmit={onSubmit}>
      <textarea
        aria-label="Your question"
        placeholder="Ask about rules, dates or your study plan"
        rows={1}
        maxLength={1000}
        autoFocus
        value={value}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={onKeyDown}
      />
      <button type="submit" aria-label="Ask" disabled={!ready}>
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path
            d="M12 19V5M5 12l7-7 7 7"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </button>
    </form>
  )
}

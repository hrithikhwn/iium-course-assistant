import { Fragment, useEffect, useState } from 'react'
import { documentUrl, type AskResponse, type Source, type Step } from './api.ts'
import { splitCitations } from './citations.ts'
import StudentPicker, { type Student } from './StudentPicker.tsx'

/** One exchange in the conversation: what the student sent, and what came back. */
export interface Turn {
  id: string
  said: string // the student's message as shown: the question, or "BCS, batch 241" after the picker
  question: string // what was sent to /ask
  reply?: AskResponse // not there yet while the API works, nor after an error
  error?: string
}

interface Props {
  turn: Turn
  latest: boolean // only the latest turn offers the picker and "Try again"
  student: Student | null
  onChooseStudent: (who: Student) => void
  onRetry: () => void
}

/** The assistant's side of a turn: waiting, an error, a question back, or the answer. */
export default function Reply({ turn, latest, student, onChooseStudent, onRetry }: Props) {
  const { reply, error } = turn

  if (error) {
    return (
      <div className="reply error" role="alert">
        <p>{error}</p>
        {latest && (
          <button type="button" className="link" onClick={onRetry}>
            Try again
          </button>
        )}
      </div>
    )
  }

  if (!reply) return <Waiting />

  if (reply.needs_student_info) {
    // The agent's reason, when it was given a batch it has no study plan for
    // (the "student" step in app/agent.py; this is its text when nothing was given).
    const why = reply.steps?.find((step) => step.name === 'student')?.detail
    return (
      <div className="reply">
        <p>{reply.answer}</p>
        {why && why !== 'missing batch/programme' && <p className="why">{why}</p>}
        {latest && <StudentPicker student={student} action="Continue" onChoose={onChooseStudent} />}
      </div>
    )
  }

  return <Answer id={turn.id} reply={reply} />
}

/** The answer, each [n] linked to source n in the list under it. */
function Answer({ id, reply }: { id: string; reply: AskResponse }) {
  const sourceId = (ref: number) => `${id}-source-${ref}`

  return (
    <div className="reply">
      <p className="answer">
        {splitCitations(reply.answer, reply.sources.length).map((part, i) =>
          typeof part === 'number' ? (
            <a key={i} className="cite" href={`#${sourceId(part)}`} title={reply.sources[part - 1].citation}>
              {part}
            </a>
          ) : (
            <Fragment key={i}>{part}</Fragment>
          ),
        )}
      </p>

      {reply.sources.length > 0 && (
        <ol className="sources" aria-label="Sources">
          {reply.sources.map((source) => (
            <SourceItem key={source.ref} id={sourceId(source.ref)} source={source} />
          ))}
        </ol>
      )}

      <div className="meta">
        <span>
          {reply.seconds.toFixed(1)} s · {reply.model}
        </span>
        {reply.steps && <Steps steps={reply.steps} />}
      </div>
    </div>
  )
}

/** A cited excerpt: its document, opened at the page, and the excerpt itself on request. */
function SourceItem({ id, source }: { id: string; source: Source }) {
  return (
    <li id={id}>
      <span className="n">{source.ref}</span>
      <div>
        <a href={documentUrl(source.source, source.page)} target="_blank" rel="noreferrer">
          {source.source.replace(/\.pdf$/i, '')}
          {source.page !== null && `, p. ${source.page}`}
        </a>
        <details>
          <summary>Excerpt</summary>
          <blockquote>{source.excerpt}</blockquote>
        </details>
      </div>
    </li>
  )
}

/** The agent's steps (the trace), folded away until asked for. */
function Steps({ steps }: { steps: Step[] }) {
  return (
    <details className="steps">
      <summary>How it found this</summary>
      <ol>
        {steps.map((step, i) => (
          <li key={i}>
            <b>{step.name}</b> {step.detail}
          </li>
        ))}
      </ol>
    </details>
  )
}

/** Seconds since it appeared. It's shown only while the API works, so each
 * question gets a new one, and the count starts again from 0. */
function Waiting() {
  const [seconds, setSeconds] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => setSeconds((s) => s + 1), 1000)
    return () => clearInterval(timer) // the reply came: stop counting
  }, [])

  return (
    <p className="reply waiting">
      <span className="dot" aria-hidden="true" />
      Searching the documents… {seconds} s
    </p>
  )
}

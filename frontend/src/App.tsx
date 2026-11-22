import { useEffect, useState } from 'react'
import { ask, health } from './api.ts'
import Composer from './Composer.tsx'
import Reply, { type Turn } from './Reply.tsx'
import StudentPicker, { type Student } from './StudentPicker.tsx'

// Offered on an empty page. All three come from the eval set, and qwen2.5:7b
// answered them right in both runs of the final eval.
const EXAMPLES = [
  "What do I need to make the Dean's List?",
  'When is the final exam period for Semester I 2026/2027?',
  'How many credits is the Final Year Project, and when can I register for FYP I?',
]

export default function App() {
  const [turns, setTurns] = useState<Turn[]>([]) // the conversation, oldest first
  const [draft, setDraft] = useState('') // the text in the message box
  const [student, setStudent] = useState<Student | null>(null) // kept until the page reloads
  const [changingStudent, setChangingStudent] = useState(false)

  const latest = turns.at(-1)
  const busy = latest !== undefined && !latest.reply && !latest.error // the API is still on the latest question

  // keep the newest message in view
  useEffect(() => {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' })
  }, [turns])

  /** Show the student's message, ask the API, and put its reply (or what went wrong) under it. */
  async function send(question: string, said = question, who = student) {
    const id = crypto.randomUUID()
    setTurns((previous) => [...previous, { id, said, question }])
    // Find the turn by id when the reply comes: after "New chat" it's gone, and the reply is dropped.
    const finish = (result: Pick<Turn, 'reply' | 'error'>) =>
      setTurns((previous) => previous.map((turn) => (turn.id === id ? { ...turn, ...result } : turn)))
    try {
      finish({ reply: await ask({ question, programme: who?.programme, batch: who?.batch, trace: true }) })
    } catch (e) {
      finish({ error: e instanceof Error ? e.message : String(e) })
    }
  }

  function sendDraft() {
    send(draft.trim())
    setDraft('')
  }

  function chooseStudent(who: Student, question: string) {
    setStudent(who)
    // `student` keeps its old value until the next render, so pass the new one along
    send(question, `${who.programme}, batch ${who.batch}`, who)
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>IIUM Course Assistant</h1>
        {turns.length > 0 && (
          <button type="button" className="new-chat" onClick={() => setTurns([])}>
            New chat
          </button>
        )}
      </header>

      <main className="thread">
        {turns.length === 0 && <Welcome onPick={(question) => send(question)} />}
        {turns.map((turn) => (
          <article key={turn.id} className="turn">
            <p className="said">{turn.said}</p>
            <Reply
              turn={turn}
              latest={turn.id === latest?.id}
              student={student}
              onChooseStudent={(who) => chooseStudent(who, turn.question)}
              onRetry={() => send(turn.question, turn.said)}
            />
          </article>
        ))}
      </main>

      <footer className="bottom">
        {changingStudent ? (
          <StudentPicker
            student={student}
            action="Save"
            onChoose={(who) => {
              setStudent(who)
              setChangingStudent(false)
            }}
            onCancel={() => setChangingStudent(false)}
          />
        ) : (
          student && (
            <p className="student">
              For {student.programme} · batch {student.batch} ·{' '}
              <button type="button" className="link" onClick={() => setChangingStudent(true)}>
                Change
              </button>
            </p>
          )
        )}
        <Composer value={draft} onChange={setDraft} onSend={sendDraft} busy={busy} />
        <p className="note">
          <ApiStatus /> · Answers come only from official KICT and IIUM documents, and earlier questions aren't
          remembered.
        </p>
      </footer>
    </div>
  )
}

/** The empty page: what the assistant does, and questions to try. */
function Welcome({ onPick }: { onPick: (question: string) => void }) {
  return (
    <section className="welcome">
      <h2>What would you like to know?</h2>
      <p>
        Ask about KICT and IIUM rules, the academic calendar or your study plan. Every answer cites the document
        and page it comes from.
      </p>
      <ul aria-label="Example questions">
        {EXAMPLES.map((question) => (
          <li key={question}>
            <button type="button" onClick={() => onPick(question)}>
              {question}
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}

/** The model the API answers with, or why the page can't reach the API. */
function ApiStatus() {
  const [status, setStatus] = useState('Checking the API…')

  useEffect(() => {
    const controller = new AbortController()
    health(controller.signal)
      .then((h) => setStatus(`Model: ${h.model}`))
      .catch((e: unknown) => {
        if (!controller.signal.aborted) setStatus(e instanceof Error ? e.message : String(e))
      })
    // in development StrictMode runs each effect twice; the first request is cancelled here
    return () => controller.abort()
  }, [])

  return <span role="status">{status}</span>
}

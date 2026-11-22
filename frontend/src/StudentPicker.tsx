import { useState, type FormEvent } from 'react'
import type { Programme } from './api.ts'

/** Who is asking: the agent searches their programme's documents and study plan. */
export interface Student {
  programme: Programme
  batch: number // 241; from a matric number, its first 3 digits
}

const PROGRAMMES: Programme[] = ['BCS', 'BIT']

/** The batch in what the student typed: "241", or a matric number that starts with it. */
function batchIn(text: string): number | null {
  const digits = text.replace(/\D/g, '')
  return digits.length >= 3 ? Number(digits.slice(0, 3)) : null
}

interface Props {
  student: Student | null // the earlier choice, if any, to start from
  action: string // the button's label
  onChoose: (student: Student) => void
  onCancel?: () => void // given: a Cancel button too
}

/** Programme and batch: when the agent asks for them, or to change them. */
export default function StudentPicker({ student, action, onChoose, onCancel }: Props) {
  const [programme, setProgramme] = useState<Programme | null>(student?.programme ?? null)
  const [batchText, setBatchText] = useState(student ? String(student.batch) : '')
  const batch = batchIn(batchText)

  function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (programme && batch) onChoose({ programme, batch })
  }

  return (
    <form className="picker" aria-label="Programme and batch" onSubmit={onSubmit}>
      <div className="choice" role="group" aria-label="Programme">
        {PROGRAMMES.map((p) => (
          <button key={p} type="button" aria-pressed={programme === p} onClick={() => setProgramme(p)}>
            {p}
          </button>
        ))}
      </div>
      <input
        aria-label="Batch"
        placeholder="Batch: 241, or your matric number"
        inputMode="numeric"
        value={batchText}
        onChange={(event) => setBatchText(event.target.value)}
      />
      <button type="submit" disabled={!programme || !batch}>
        {action}
      </button>
      {onCancel && (
        <button type="button" className="link" onClick={onCancel}>
          Cancel
        </button>
      )}
    </form>
  )
}

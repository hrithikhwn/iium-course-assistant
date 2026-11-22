import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, test, vi } from 'vitest'
import { NOT_RUNNING, documentUrl, type Programme } from './api.ts'
import App from './App.tsx'
import { emptyVectorStore, found, needsStudentInfo, notFound } from './test/fixtures.ts'

/** Swap fetch for a fake API: /health names the model, and each /ask gets the next reply. */
function fakeApi(...replies: (Response | Promise<Response>)[]) {
  const fake = vi.fn<typeof fetch>(async (url) => {
    if (url === '/health') return Response.json({ status: 'ok', model: 'qwen2.5:7b' })
    const reply = replies.shift()
    if (reply === undefined) throw new Error('the test has no reply for this /ask')
    return reply
  })
  vi.stubGlobal('fetch', fake)
  return fake
}

/** What the page sent to /ask, in order. */
function asked(fake: ReturnType<typeof fakeApi>): unknown[] {
  return fake.mock.calls.filter(([url]) => url === '/ask').map(([, init]) => JSON.parse(String(init?.body)))
}

/** Show the page and wait for its health check, as a student would see it. */
async function renderApp() {
  const user = userEvent.setup()
  render(<App />)
  await screen.findByText('Model: qwen2.5:7b')
  return { user, box: screen.getByRole('textbox', { name: 'Your question' }) }
}

const question = 'When can I register for FYP I?'
const answered = /consists of 7 credit hours/ // part of found.answer

describe('asking', () => {
  test('shows the model the API runs', async () => {
    fakeApi()
    render(<App />)
    expect(await screen.findByText('Model: qwen2.5:7b')).toBeInTheDocument()
  })

  test('says how to start the API when it is not running', async () => {
    // Vite's dev proxy answers 502 while uvicorn isn't running
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 502, statusText: 'Bad Gateway' })))
    render(<App />)
    expect(await screen.findByText(NOT_RUNNING)).toBeInTheDocument()
  })

  test('Enter sends the question, and the answer comes under it', async () => {
    const fake = fakeApi(Response.json(found))
    const { user, box } = await renderApp()
    await user.type(box, `${question}{Enter}`)

    expect(await screen.findByText(answered)).toBeInTheDocument()
    expect(screen.getByText(question)).toHaveClass('said') // the student's bubble
    expect(asked(fake)).toEqual([{ question, trace: true }])
    expect(box).toHaveValue('') // ready for the next question
  })

  test('Shift+Enter starts a new line instead', async () => {
    const fake = fakeApi()
    const { user, box } = await renderApp()
    await user.type(box, 'First line{Shift>}{Enter}{/Shift}second line')

    expect(box).toHaveValue('First line\nsecond line')
    expect(asked(fake)).toEqual([])
  })

  test('a question needs at least 3 characters', async () => {
    const fake = fakeApi()
    const { user, box } = await renderApp()
    await user.type(box, ' hi {Enter}')

    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled()
    expect(asked(fake)).toEqual([])
  })

  test('counts the seconds while the API works on the question', async () => {
    let reply!: (response: Response) => void
    fakeApi(new Promise<Response>((resolve) => (reply = resolve)))
    const { user, box } = await renderApp()
    await user.type(box, `${question}{Enter}`)

    expect(await screen.findByText('Searching the documents… 0 s')).toBeInTheDocument()
    expect(await screen.findByText('Searching the documents… 1 s', {}, { timeout: 2000 })).toBeInTheDocument()
    await user.type(box, 'Next question')
    expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled() // one question at a time

    reply(Response.json(found))
    expect(await screen.findByText(answered)).toBeInTheDocument()
    expect(screen.queryByText(/Searching the documents/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Ask' })).toBeEnabled()
  })

  test('says why the API failed, and Try again asks again', async () => {
    const fake = fakeApi(Response.json(emptyVectorStore, { status: 503 }), Response.json(found))
    const { user, box } = await renderApp()
    await user.type(box, `${question}{Enter}`)
    expect(await screen.findByRole('alert')).toHaveTextContent(emptyVectorStore.detail)

    await user.click(screen.getByRole('button', { name: 'Try again' }))
    expect(await screen.findByText(answered)).toBeInTheDocument()
    expect(asked(fake)).toEqual([
      { question, trace: true },
      { question, trace: true },
    ])
  })

  test('asks an example question in one click', async () => {
    const fake = fakeApi(Response.json(needsStudentInfo))
    const { user } = await renderApp()
    const examples = within(screen.getByRole('list', { name: 'Example questions' })).getAllByRole('button')
    expect(examples).toHaveLength(3)

    await user.click(examples[0])
    expect(await screen.findByText(needsStudentInfo.answer)).toBeInTheDocument()
    expect(asked(fake)).toEqual([{ question: examples[0].textContent, trace: true }])
    // the examples are only for an empty page
    expect(screen.queryByRole('list', { name: 'Example questions' })).not.toBeInTheDocument()
  })

  test('New chat clears the conversation', async () => {
    fakeApi(Response.json(found))
    const { user, box } = await renderApp()
    await user.type(box, `${question}{Enter}`)
    await screen.findByText(answered)

    await user.click(screen.getByRole('button', { name: 'New chat' }))
    expect(screen.queryByText(answered)).not.toBeInTheDocument()
    expect(screen.getByRole('list', { name: 'Example questions' })).toBeInTheDocument()
  })
})

describe('programme and batch', () => {
  /** Ask, get asked for programme and batch, and give them. */
  async function introduce(user: ReturnType<typeof userEvent.setup>, box: HTMLElement, programme: Programme, batch: string) {
    await user.type(box, `${question}{Enter}`)
    await screen.findByText(needsStudentInfo.answer)
    await user.click(screen.getByRole('button', { name: programme }))
    await user.type(screen.getByRole('textbox', { name: 'Batch' }), batch)
    await user.click(screen.getByRole('button', { name: 'Continue' }))
  }

  test('when the agent asks for them, the same question goes again with them', async () => {
    const fake = fakeApi(Response.json(needsStudentInfo), Response.json(found))
    const { user, box } = await renderApp()
    await introduce(user, box, 'BIT', '2610123') // a matric number: the batch is its first 3 digits

    expect(await screen.findByText(answered)).toBeInTheDocument()
    expect(asked(fake)).toEqual([
      { question, trace: true },
      { question, programme: 'BIT', batch: 261, trace: true },
    ])
    expect(screen.getByText('BIT, batch 261')).toHaveClass('said')
    expect(screen.getByText(/For BIT · batch 261/)).toBeInTheDocument()
  })

  test('Continue needs a programme and a batch of 3 digits', async () => {
    fakeApi(Response.json(needsStudentInfo))
    const { user, box } = await renderApp()
    await user.type(box, `${question}{Enter}`)
    await screen.findByText(needsStudentInfo.answer)
    const go = screen.getByRole('button', { name: 'Continue' })
    const batch = screen.getByRole('textbox', { name: 'Batch' })

    expect(go).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'BCS' }))
    await user.type(batch, '24')
    expect(go).toBeDisabled()
    await user.type(batch, '1')
    expect(go).toBeEnabled()
  })

  test('later questions take them along without asking', async () => {
    const fake = fakeApi(Response.json(needsStudentInfo), Response.json(found), Response.json(notFound))
    const { user, box } = await renderApp()
    await introduce(user, box, 'BCS', '241')
    await screen.findByText(answered)

    await user.type(box, 'What time does the KICT library open?{Enter}')
    await screen.findByText(notFound.answer)
    expect(asked(fake)[2]).toEqual({
      question: 'What time does the KICT library open?',
      programme: 'BCS',
      batch: 241,
      trace: true,
    })
  })

  test('Change sets them for the next question', async () => {
    const fake = fakeApi(Response.json(needsStudentInfo), Response.json(found), Response.json(found))
    const { user, box } = await renderApp()
    await introduce(user, box, 'BCS', '241')
    await screen.findByText(answered)

    await user.click(screen.getByRole('button', { name: 'Change' }))
    await user.click(screen.getByRole('button', { name: 'BIT' }))
    await user.clear(screen.getByRole('textbox', { name: 'Batch' }))
    await user.type(screen.getByRole('textbox', { name: 'Batch' }), '251')
    await user.click(screen.getByRole('button', { name: 'Save' }))
    expect(screen.getByText(/For BIT · batch 251/)).toBeInTheDocument()

    await user.type(box, `${question}{Enter}`)
    await waitFor(() => expect(screen.getAllByText(answered)).toHaveLength(2))
    expect(asked(fake)[2]).toEqual({ question, programme: 'BIT', batch: 251, trace: true })
  })

  test('says why when a batch has no study plan, and asks again', async () => {
    const reason = 'batch 191 is older than the 201 plan; no study plan for it'
    const noPlan = { ...needsStudentInfo, steps: [{ name: 'student', detail: reason }] }
    fakeApi(Response.json(needsStudentInfo), Response.json(noPlan))
    const { user, box } = await renderApp()
    await introduce(user, box, 'BCS', '191')

    expect(await screen.findByText(reason)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Continue' })).toBeInTheDocument()
  })
})

describe('answers', () => {
  test('link each [n] to its source, and each source to its page', async () => {
    fakeApi(Response.json(found))
    const { user, box } = await renderApp()
    await user.type(box, `${question}{Enter}`)
    const answer = await screen.findByText(answered)

    const sources = within(screen.getByRole('list', { name: 'Sources' })).getAllByRole('listitem')
    const cites = within(answer).getAllByRole('link')
    expect(cites.map((cite) => cite.textContent)).toEqual(['1', '2'])
    expect(cites[1]).toHaveAttribute('href', `#${sources[1].id}`)

    const page = within(sources[1]).getByRole('link', { name: 'KICT-Course-Registration-FAQ-BCS-BIT, p. 7' })
    expect(page).toHaveAttribute('href', documentUrl('KICT-Course-Registration-FAQ-BCS-BIT.pdf', 7))
    expect(screen.getByText('26.0 s · qwen2.5:7b')).toBeInTheDocument()
  })

  test("say so when the documents don't have the answer", async () => {
    fakeApi(Response.json(notFound))
    const { user, box } = await renderApp()
    await user.type(box, 'What time does the KICT library open?{Enter}')

    expect(await screen.findByText(notFound.answer)).toBeInTheDocument()
    expect(screen.queryByRole('list', { name: 'Sources' })).not.toBeInTheDocument()
  })

  test('show how the agent found the answer', async () => {
    fakeApi(Response.json(found))
    const { user, box } = await renderApp()
    await user.type(box, `${question}{Enter}`)
    await screen.findByText(answered)
    const step = screen.getByText('sufficient=True — Prerequisites and credit hours for FYP I are provided.')

    expect(step).not.toBeVisible() // folded away
    await user.click(screen.getByText('How it found this'))
    expect(step).toBeVisible()
  })
})

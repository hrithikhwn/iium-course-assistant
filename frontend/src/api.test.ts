import { describe, expect, test, vi } from 'vitest'
import { ApiError, NOT_RUNNING, ask, documentUrl, health } from './api.ts'
import { emptyVectorStore, found } from './test/fixtures.ts'

/** Swap fetch for a stand-in that returns `response`, or throws it if it's an error. */
function stubFetch(response: Response | Error) {
  const fake = vi.fn<typeof fetch>(async () => {
    // not `instanceof Error`: jsdom's DOMException comes from another realm and fails that check
    if (response instanceof Response) return response
    throw response
  })
  vi.stubGlobal('fetch', fake)
  return fake
}

/** What a promise rejects with. */
function rejection(promise: Promise<unknown>): Promise<unknown> {
  return promise.then(
    () => {
      throw new Error('expected the request to fail')
    },
    (e: unknown) => e,
  )
}

const question = 'When can I register for FYP I?'

describe('ask', () => {
  test('posts the question as JSON and returns the answer', async () => {
    const fake = stubFetch(Response.json(found))
    const req = { question, programme: 'BCS', batch: 241, trace: true } as const
    expect(await ask(req)).toEqual(found)

    const [url, init] = fake.mock.calls[0]
    expect(url).toBe('/ask')
    expect(init?.method).toBe('POST')
    expect(JSON.parse(String(init?.body))).toEqual(req)
  })

  test("a 503 carries FastAPI's reason", async () => {
    stubFetch(Response.json(emptyVectorStore, { status: 503 }))
    const error = await rejection(ask({ question }))
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 503, message: emptyVectorStore.detail })
  })

  test('a 422 lists what was wrong with the request', async () => {
    const detail = [{ msg: 'String should have at least 3 characters' }, { msg: 'Input should be a valid integer' }]
    stubFetch(Response.json({ detail }, { status: 422 }))
    expect(await rejection(ask({ question: 'hi' }))).toMatchObject({
      status: 422,
      message: 'String should have at least 3 characters; Input should be a valid integer',
    })
  })

  test('an error without a reason gives the status', async () => {
    stubFetch(new Response('Internal Server Error', { status: 500, statusText: 'Internal Server Error' }))
    expect(await rejection(ask({ question }))).toMatchObject({
      status: 500,
      message: 'The API answered 500 Internal Server Error',
    })
  })

  test('says how to start the API when nothing answers', async () => {
    stubFetch(new TypeError('Failed to fetch'))
    expect(await rejection(ask({ question }))).toMatchObject({ status: 0, message: NOT_RUNNING })
  })

  test("says how to start the API when Vite's proxy can't reach it", async () => {
    // what `npm run dev` sends while uvicorn isn't running: 502, empty body
    stubFetch(new Response('', { status: 502, statusText: 'Bad Gateway' }))
    expect(await rejection(ask({ question }))).toMatchObject({ status: 502, message: NOT_RUNNING })
  })

  test('a cancelled request stays cancelled, not an API error', async () => {
    const controller = new AbortController()
    controller.abort()
    stubFetch(new DOMException('The operation was aborted.', 'AbortError'))
    const error = await rejection(ask({ question }, controller.signal))
    expect(error).not.toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ name: 'AbortError' })
  })
})

test('health asks /health', async () => {
  const fake = stubFetch(Response.json({ status: 'ok', model: 'qwen2.5:7b' }))
  expect(await health()).toEqual({ status: 'ok', model: 'qwen2.5:7b' })
  expect(fake.mock.calls[0][0]).toBe('/health')
})

describe('documentUrl', () => {
  test('opens the document at the cited page', () => {
    expect(documentUrl('BCS Study Plan (Batch 241) Rev_3.pdf', 1)).toBe(
      '/documents/BCS%20Study%20Plan%20(Batch%20241)%20Rev_3.pdf#page=1',
    )
  })

  test('links to the whole document when it has no pages', () => {
    expect(documentUrl('notes.txt', null)).toBe('/documents/notes.txt')
  })
})

// The FastAPI backend (app/main.py). These types mirror AskRequest, Source
// and AskResponse there: change them together.

export type Programme = 'BCS' | 'BIT'

export interface AskRequest {
  question: string // 3 to 1000 characters
  programme?: Programme
  batch?: number | string // 241, or a matric number that starts with it, like "2410123"
  trace?: boolean // true: the response lists the agent's steps
}

export interface Source {
  ref: number // the n of [n] in the answer
  citation: string // "file.pdf, p.42"
  source: string // the document's file name; documentUrl() links to it
  page: number | null // from 1; null for files without pages
  excerpt: string // the whole excerpt, up to about 900 characters
}

export interface Step {
  name: string // student, plan, retrieve, grade or answer
  detail: string
}

export interface AskResponse {
  answer: string // with [1], [2], ... after the facts they support
  sources: Source[] // only the excerpts the answer cites, numbered as in the answer
  needs_student_info: boolean // true: ask for programme and batch, then send the question again
  found: boolean // false: the documents don't answer it; `answer` says so and there are no sources
  steps: Step[] | null // null unless the request had trace: true
  seconds: number
  model: string
}

export interface Health {
  status: string
  model: string
}

/** The API answered with an error, or didn't answer at all: status 0 without a dev proxy, 502 with one. */
export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export function ask(req: AskRequest, signal?: AbortSignal): Promise<AskResponse> {
  return request<AskResponse>('/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  })
}

export function health(signal?: AbortSignal): Promise<Health> {
  return request<Health>('/health', { signal })
}

/** A link that opens a cited document at its page in the browser's PDF viewer. */
export function documentUrl(source: string, page: number | null): string {
  const url = `/documents/${encodeURIComponent(source)}`
  return page ? `${url}#page=${page}` : url
}

export const NOT_RUNNING = "Can't reach the API. Start it with: uvicorn app.main:app --reload"

async function request<T>(path: string, init: RequestInit): Promise<T> {
  let resp: Response
  try {
    resp = await fetch(path, init)
  } catch (e) {
    if (init.signal?.aborted) throw e // cancelled by the caller: not an API problem
    throw new ApiError(NOT_RUNNING, 0)
  }
  if (!resp.ok) throw new ApiError(await errorMessage(resp), resp.status)
  return (await resp.json()) as T
}

interface ErrorBody {
  detail?: string | { msg: string }[]
}

/** FastAPI's reason: {"detail": "..."} with a 404 or 503, a list of problems with a 422. */
async function errorMessage(resp: Response): Promise<string> {
  const body = (await resp.json().catch(() => null)) as ErrorBody | null
  const detail = body?.detail
  if (typeof detail === 'string') return detail
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join('; ')
  // FastAPI never sends a 502 itself: Vite's dev proxy does, when nothing listens on the API's port
  if (resp.status === 502) return NOT_RUNNING
  return `The API answered ${resp.status} ${resp.statusText}`.trim()
}

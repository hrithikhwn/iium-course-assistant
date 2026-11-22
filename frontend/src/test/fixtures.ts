// Saved /ask responses for tests, one for each state the page shows. The
// answers, steps and timings are real: the eval's qwen2.5:7b run with the
// default settings (eval/results/2026-11-22_0305_qwen2.5-7b_no-rewrite.csv),
// asked as BCS batch 241. One excerpt is shortened.
import type { AskResponse } from '../api.ts'

// q13: "BCS batch 241: how many credits is the Final Year Project, and when can I register for FYP I?"
export const found: AskResponse = {
  answer:
    'The Final Year Project for BCS batch 241 consists of 7 credit hours, split into FYP I (3 credit hours) ' +
    'and FYP II (4 credit hours) [1]. You can register for FYP I after completing at least 90 credit hours ' +
    'and having a proposal approved by a supervisor [2].',
  sources: [
    {
      ref: 1,
      citation: 'BCS Study Plan (Batch 241) Rev_3.pdf, p.1',
      source: 'BCS Study Plan (Batch 241) Rev_3.pdf',
      page: 1,
      excerpt:
        'BCS study plan 241, Final Year Project:\n' +
        '- BICS 4321 Final Year Project I: 3 credit hours; prerequisite: >=90 Credits; offered by DCS\n' +
        '- BICS 4421 Final Year Project II: 4 credit hours; prerequisite: BICS 4321; offered by DCS',
    },
    {
      ref: 2,
      citation: 'KICT-Course-Registration-FAQ-BCS-BIT.pdf, p.7',
      source: 'KICT-Course-Registration-FAQ-BCS-BIT.pdf',
      page: 7,
      // the real excerpt starts with the industrial training contacts
      excerpt:
        'What are the prerequisites for registering FYP? (BCS + BIT) \n' +
        'Students must have completed at least 90 credit hours for study plan Batch 201 & Batch 241, and at least \n' +
        '78 credit hours for study plan Batch 261. All students must pass the relevant core courses before registering \n' +
        'for the Final Year Project, and must have a proposal approved by a supervisor.',
    },
  ],
  needs_student_info: false,
  found: true,
  steps: [
    { name: 'student', detail: 'BCS 241 -> plan 241' },
    { name: 'plan', detail: 'rewrite off: searching the question only' },
    {
      name: 'retrieve',
      detail:
        "5 chunks for 'BCS batch 241: how many credits is the Final Year Project, and when can I register " +
        "for FYP I?' (plan 241, BCS); 5 in context",
    },
    { name: 'grade', detail: 'sufficient=False — Missing registration prerequisites for FYP I' },
    { name: 'retrieve', detail: "5 chunks for 'BCS batch 241 FYP registration prerequisites' (plan 241, BCS); 6 in context" },
    { name: 'grade', detail: 'sufficient=True — Prerequisites and credit hours for FYP I are provided.' },
    { name: 'answer', detail: '6 excerpts in context; told the study plan; cited excerpts [6, 1]' },
  ],
  seconds: 26.0,
  model: 'qwen2.5:7b',
}

// q39: "What time does the KICT library open?" (the documents don't say)
export const notFound: AskResponse = {
  answer:
    "I couldn't find that in the official KICT and IIUM documents I have. " +
    'Please check with your academic advisor or the KICT academic office.',
  sources: [],
  needs_student_info: false,
  found: false,
  steps: [
    { name: 'student', detail: 'BCS 241 -> plan 241' },
    { name: 'plan', detail: 'rewrite off: searching the question only' },
    { name: 'retrieve', detail: "5 chunks for 'What time does the KICT library open?' (plan 241, BCS); 5 in context" },
    { name: 'grade', detail: 'sufficient=False — No information about library opening time' },
    { name: 'retrieve', detail: "5 chunks for 'KICT library hours' (plan 241, BCS); 7 in context" },
    { name: 'grade', detail: 'sufficient=False — No information about KICT library hours' },
    { name: 'answer', detail: 'model replied NOT_FOUND from 7 excerpts' },
  ],
  seconds: 22.5,
  model: 'qwen2.5:7b',
}

// Any question sent without programme and batch. Note `found` is still true:
// check needs_student_info first.
export const needsStudentInfo: AskResponse = {
  answer: "Which programme are you in (BCS or BIT), and what's your batch (e.g. 241)?",
  sources: [],
  needs_student_info: true,
  found: true,
  steps: [{ name: 'student', detail: 'missing batch/programme' }],
  seconds: 0,
  model: 'qwen2.5:7b',
}

/** The body of the 503 that /ask sends before `python -m app.ingest` has run. */
export const emptyVectorStore = { detail: 'Vector store is empty — run `python -m app.ingest` first.' }

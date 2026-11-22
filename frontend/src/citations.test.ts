import { expect, test } from 'vitest'
import { splitCitations } from './citations.ts'
import { found } from './test/fixtures.ts'

test('cuts a real answer at its citations, keeping the numbers', () => {
  expect(splitCitations(found.answer, found.sources.length)).toEqual([
    'The Final Year Project for BCS batch 241 consists of 7 credit hours, split into FYP I (3 credit hours) ' +
      'and FYP II (4 credit hours)',
    1,
    '. You can register for FYP I after completing at least 90 credit hours and having a proposal approved ' +
      'by a supervisor',
    2,
    '.',
  ])
})

test('reads citations side by side, and two in one pair of brackets', () => {
  expect(splitCitations('Approved by a supervisor [1][2].', 2)).toEqual(['Approved by a supervisor', 1, 2, '.'])
  expect(splitCitations('Approved by a supervisor [1, 2].', 2)).toEqual(['Approved by a supervisor', 1, 2, '.'])
})

test('leaves a number with no source as text', () => {
  expect(splitCitations('Rule 35 [3] says so [1].', 1)).toEqual(['Rule 35 [3] says so', 1, '.'])
})

test('leaves brackets that are not citations alone', () => {
  const answer = 'Lectures start at [8:00], see [the timetable].'
  expect(splitCitations(answer, 2)).toEqual([answer])
})

test('an answer without citations stays in one piece', () => {
  expect(splitCitations("I couldn't find that.", 0)).toEqual(["I couldn't find that."])
})

test('a citation can open the answer', () => {
  expect(splitCitations('[1] 20 credit hours at most.', 1)).toEqual([1, ' 20 credit hours at most.'])
})

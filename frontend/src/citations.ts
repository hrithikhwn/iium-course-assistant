// How the API marks the source of a fact: "[1]", or "[1, 2]" for two. The API
// renumbers them in order of use (renumber_citations in app/text.py), so [n]
// always means sources[n - 1].
const CITATION = /\s*\[(\d+(?:\s*,\s*\d+)*)\]/g

/**
 * The answer cut into text and source numbers, so each [n] can become a link:
 * "7 credit hours [1]. After 90 [1, 2]." -> ["7 credit hours", 1, ". After 90", 1, 2, "."]
 * The space before a citation is dropped, so its link sits against the word.
 * A number with no source (there are `sourceCount`) stays as plain text.
 */
export function splitCitations(answer: string, sourceCount: number): (string | number)[] {
  const parts: (string | number)[] = []
  let done = 0 // how much of the answer is in `parts` already
  for (const match of answer.matchAll(CITATION)) {
    const refs = match[1].split(',').map(Number)
    if (!refs.every((n) => n >= 1 && n <= sourceCount)) continue // not a source: leave it as text
    parts.push(answer.slice(done, match.index), ...refs)
    done = match.index + match[0].length
  }
  parts.push(answer.slice(done))
  return parts.filter((part) => part !== '')
}

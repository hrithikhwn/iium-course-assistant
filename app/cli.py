"""Ask a question from the terminal.

    python -m app.cli "What is the minimum CGPA to avoid probation?" --programme BCS --batch 241 --trace
"""
from __future__ import annotations

import argparse
import sys

from .agent import CourseAgent
from .llm import OpenAICompatibleLLM
from .retriever import ChromaRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the IIUM Course Assistant a question.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--programme", help="BCS or BIT")
    parser.add_argument("--batch", help="your batch, e.g. 241 (or your matric number)")
    parser.add_argument("--trace", action="store_true", help="show the agent's steps")
    args = parser.parse_args()
    # Windows writes piped or redirected output in its ANSI code page (cp1252),
    # which has no "≤" (SAPER writes "1.67 ≤ CGPA"), and printing it crashed
    sys.stdout.reconfigure(encoding="utf-8")

    agent = CourseAgent(OpenAICompatibleLLM(), ChromaRetriever())
    result = agent.run(" ".join(args.question), batch=args.batch, programme=args.programme)

    if result.needs_student_info:
        # ask once in the terminal (Enter keeps what was passed as a flag), then run again
        print(result.answer)
        programme = input(f"Programme (BCS/BIT) [{args.programme or ''}]: ").strip() or args.programme
        batch = input(f"Batch, e.g. 241 [{args.batch or ''}]: ").strip() or args.batch
        result = agent.run(" ".join(args.question), batch=batch, programme=programme)
        if result.needs_student_info:
            raise SystemExit(f"Sorry, I can't tell which study plan that is: {result.steps[-1].detail}")

    if args.trace:
        print("--- trace ---")
        for s in result.steps:
            print(f"[{s.name}] {s.detail}")
        print()

    print(result.answer)
    if result.sources:
        print("\nSources:")
        for i, c in enumerate(result.sources, 1):
            print(f"  [{i}] {c.citation()}")
    print(f"\n({result.seconds:.1f}s)")


if __name__ == "__main__":
    main()

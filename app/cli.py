"""Ask a question from the terminal.

    python -m app.cli "What is the minimum CGPA to avoid probation?" --trace
"""
from __future__ import annotations

import argparse

from .agent import CourseAgent
from .llm import OpenAICompatibleLLM
from .retriever import ChromaRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the IIUM Course Assistant a question.")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--trace", action="store_true", help="show the agent's steps")
    args = parser.parse_args()

    agent = CourseAgent(OpenAICompatibleLLM(), ChromaRetriever())
    result = agent.run(" ".join(args.question))

    if args.trace:
        print("--- trace ---")
        for s in result.steps:
            print(f"[{s.name}] {s.detail}")
        print()

    print(result.answer)
    print("\nSources:")
    for i, c in enumerate(result.sources, 1):
        print(f"  [{i}] {c.citation()}")
    print(f"\n({result.seconds:.1f}s)")


if __name__ == "__main__":
    main()

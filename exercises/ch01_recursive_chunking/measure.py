"""Show your chunks for both parts, then which chunk each question retrieves.

Uses the local Nomic encoder (about 9 s to load). No API key, no Docker.
"""

import numpy as np

from exercises.common import implementation
from modules.c01_embeddings.local_embeddings import LocalEncoder

# Each question, and the wording its chunk must contain to answer it correctly.
PART1_QUESTIONS = [
    ("Can I get a full refund if I already used the service?", ["provided they have not used the service"]),
    ("What happens if I cancel after 14 days?", ["No refund is issued for the remaining days"]),
]
PART2_QUESTIONS = [
    (
        "Can a customer who used the service receive a full refund within 14 days?",
        ["full refund if they cancel within 14 days", "already used the service are not eligible"],
    ),
]


def flat(text):
    return " ".join(text.split())


def retrieve(model, texts, questions):
    """Print the best chunk for every question; return how many hold all needed wording."""
    vectors = model.encode(["search_document: " + text for text in texts])
    complete = 0
    for question, needed in questions:
        best = int(np.argmax(vectors @ model.encode(["search_query: " + question])[0]))
        missing = [phrase for phrase in needed if phrase not in flat(texts[best])]
        complete += not missing
        print(f"\n  Q: {question}")
        print(f"     chunk {best + 1}: {flat(texts[best])[:150]}")
        print("     complete: the rule and its condition are in the chunk" if not missing
              else f"     INCOMPLETE: the chunk is missing \"{missing[0]}\"")
    return complete


def main():
    lab = implementation("exercises.ch01_recursive_chunking")
    model = LocalEncoder(allow_download=True).load()

    chunks = lab.make_chunks()
    print(f"=== Part 1 · chunk_size={lab.CHUNK_SIZE} · chunk_overlap={lab.CHUNK_OVERLAP} · {len(chunks)} chunks")
    for number, chunk in enumerate(chunks, 1):
        print(f"  {number}. {len(chunk):3} chars · {flat(chunk)}")
    part1 = retrieve(model, chunks, PART1_QUESTIONS)

    sections = lab.make_sections()
    print(f"\n=== Part 2 · split on {[h for h, _ in lab.HEADERS_TO_SPLIT_ON]} · {len(sections)} chunks")
    for number, chunk in enumerate(sections, 1):
        print(f"  {number}. {chunk.metadata}")
    part2 = retrieve(model, [chunk.page_content for chunk in sections], PART2_QUESTIONS)

    print(f"\nPart 1: {part1}/{len(PART1_QUESTIONS)} · Part 2: {part2}/{len(PART2_QUESTIONS)} "
          "questions retrieve a chunk with the complete rule.")


if __name__ == "__main__":
    main()

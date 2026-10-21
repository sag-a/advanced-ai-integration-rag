"""Compare fixed, recursive, and section-based splitting on the same document."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .data import load_support_text


def split_text(text: str, chunk_size: int, overlap: int = 0) -> list[str]:
    """Cut text into character windows, repeating `overlap` characters."""
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Choose a positive chunk size and 0 <= overlap < chunk size.")

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        # Example: size 300 and overlap 50 means the next chunk starts at 250.
        start = end - overlap
    return chunks


def split_sections(text: str) -> list[str]:
    """Use this guide's format: a title, then heading/body blocks with blank lines."""
    blocks = text.strip().split("\n\n")
    if len(blocks) < 2:
        return [text.strip()] if text.strip() else []
    # Keep the document title with the first section, not as a tiny extra chunk.
    first_section = blocks[0] + "\n\n" + blocks[1]
    return [first_section] + blocks[2:]


def main() -> None:
    text = load_support_text()
    chunk_size = 300  # Characters, not tokens. Try 600 after the first run.
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=0,
        length_function=len,
    )
    examples = [
        (
            "1. Fixed characters",
            "Cut at the size limit, even inside a word.",
            split_text(text, chunk_size),
        ),
        (
            "2. Recursive",
            "Try paragraphs, lines, and spaces before cutting characters.",
            recursive_splitter.split_text(text),
        ),
        (
            "3. Sections",
            "Keep each heading with its body; sections can exceed the size limit.",
            split_sections(text),
        ),
    ]
    print(f"Same document. Fixed/recursive limit: {chunk_size} characters; overlap: 0.")
    print("Sections follow document structure instead of enforcing that limit.")

    for title, explanation, chunks in examples:
        print(f"\n=== {title} ===")
        print(f"{explanation}\nChunks: {len(chunks)}")
        for number, chunk in enumerate(chunks, start=1):
            print(f"\n--- Chunk {number}: {len(chunk)} characters ---")
            print(chunk)

    print("\nDiscuss: Which chunks separate a heading from its instructions?")
    print(
        "Recursive splitting can still separate related text. Sections may be too long."
    )
    print("These boundaries alone do not prove better retrieval or better answers.")


if __name__ == "__main__":
    main()

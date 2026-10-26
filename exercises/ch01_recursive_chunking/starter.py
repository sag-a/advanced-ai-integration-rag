"""Chapter 1 exercise: fix the chunking settings in Part 1 and Part 2."""

from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Part 1: size-based chunking. Keep each refund rule with its condition.
# ---------------------------------------------------------------------------

TEXT = """
Customers can receive a full refund if they cancel within 14 days of
purchase, provided they have not used the service. Customers who have
already used the service are not eligible for a full refund.

After 14 days, customers can still cancel their subscription, but the
cancellation takes effect at the end of the current billing period.
No refund is issued for the remaining days.
"""

# These settings are intentionally poor. Fix them!
CHUNK_SIZE = 300
CHUNK_OVERLAP = 0


def make_chunks(text=TEXT):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Part 2: header-based chunking. One chunk per main section of the manual.
# ---------------------------------------------------------------------------

MANUAL = Path(__file__).with_name("support_manual.md").read_text(encoding="utf-8")

# Change ONLY this configuration. These settings are intentionally poor.
HEADERS_TO_SPLIT_ON = [
    ("#", "Document"),
    ("###", "Subsection"),
]


def make_sections(text=MANUAL):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,
    )
    return splitter.split_text(text)


if __name__ == "__main__":
    print("=== Part 1: size-based chunking ===")
    for i, chunk in enumerate(make_chunks(), start=1):
        print(f"\n--- Chunk {i} ({len(chunk)} characters) ---")
        print(chunk)

    sections = make_sections()
    print(f"\n=== Part 2: header-based chunking · {len(sections)} chunks ===")
    for i, chunk in enumerate(sections, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Metadata: {chunk.metadata}")
        print(chunk.page_content)

# Chapter 1 exercise: fix the chunk boundaries

After lesson 1 (chunking, slides 24–26) · about 20 minutes · no API key, no Docker

Both parts use the same refund policy. A chatbot answers from **one** retrieved chunk,
so a rule cut away from its condition becomes a wrong answer. You fix the splitter
settings in [`starter.py`](starter.py), never the documents.

```bash
uv run --all-extras python -m exercises.ch01_recursive_chunking.starter    # print the chunks of both parts
uv run --all-extras python -m exercises.ch01_recursive_chunking.measure    # which chunk each question retrieves
uv run --all-extras pytest exercises/ch01_recursive_chunking               # 10 tests, 5 per part
```

## Part 1: size-based chunking

`RecursiveCharacterTextSplitter` cuts plain text by size. With a chunk size that is too
small, the refund rule is cut from its condition:

```text
--- Chunk 1 (43 characters) ---
Customers can receive a full refund if they
--- Chunk 2 (24 characters) ---
cancel within 14 days of
--- Chunk 3 (41 characters) ---
purchase, provided they have not used the
```

**Task:** change `CHUNK_SIZE` (and, if you like, `CHUNK_OVERLAP`) until every rule keeps
its condition, no chunk ends mid-sentence, and the two rules stay in separate chunks.

## Part 2: Markdown header chunking

[`support_manual.md`](support_manual.md) has `#`, `##` and `###` headings.
`MarkdownHeaderTextSplitter` cuts at the heading levels you list and records each
heading in the chunk's metadata. The chatbot question is:

> Can a customer who used the service receive a full refund within 14 days?

The starter splits at `#` and `###` but not at `##`. The **Eligibility** rule and its
**Exceptions** land in different chunks, the main section is missing from the metadata,
and the `## Cancellations` heading ends up at the bottom of the Exceptions chunk.
Retrieval returns the Eligibility chunk alone, and the chatbot would promise a refund.

**Task:** change only `HEADERS_TO_SPLIT_ON`. Leave the Markdown unchanged.

**Success criteria:**

- Exactly three chunks: Refunds, Cancellations and Payments.
- The Refunds chunk contains both Eligibility and Exceptions.
- Each chunk's metadata includes `Document` and `Section`, with the correct values.
- No chunk mixes content from different main sections.

## Done when

`measure` reports `Part 1: 2/2 · Part 2: 1/1` and all 10 tests pass.

## Discuss

- Part 1: which chunk sizes pass? Find the smallest and the largest. Why does a size above
  that fail, even though nothing is cut? Why does `CHUNK_OVERLAP` change nothing here?
- Part 2: why does splitting at **every** heading level (`#`, `##`, `###`) make this
  example worse? Which single chunk now contains all the evidence the chatbot needs?
- Both parts fail the same way: a rule separated from its exception. How does the
  course's PDF pipeline avoid it for the Magenta terms? (It splits by clause number and
  repeats the section heading in every chunk: `rag_core/corpus.py`.)

## For instructors

Answers: Part 1 passes for any `CHUNK_SIZE` from 200 to 383 (the first paragraph is 199
characters, both together 382); the reference uses 250. Part 2 is
`[("#", "Document"), ("##", "Section")]`; adding `("###", "Subsection")` splits the rule
from its exception again. Grade the reference with
`LAB_TARGET=solution uv run --all-extras pytest exercises/ch01_recursive_chunking`.

Reference: [LangChain MarkdownHeaderTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter)

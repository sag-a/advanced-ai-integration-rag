"""Split the support document with LangChain, then embed its chunks locally."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from modules.c01_embeddings.local_embeddings import (
    EMBEDDING_MODEL as MODEL,
    LocalEncoder,
)

from .data import load_support_text
from .display import print_vectors


def main() -> None:
    model = LocalEncoder(allow_download=True).load()
    text = load_support_text()
    # Use the same support document as the boundary and search demonstrations.
    # 300 CHARACTERS is roughly 70 English tokens. Overlap is only a target.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50, length_function=len
    )
    chunks = splitter.split_text(text)
    inputs = ["search_document: " + chunk for chunk in chunks]
    print(
        "300 characters is roughly 70 English tokens; inspect the actual counts below."
    )
    print(
        "Production chunks are commonly several times larger and should be tuned to the content."
    )
    for index, (chunk, embedding_input) in enumerate(
        zip(chunks, inputs, strict=True), start=1
    ):
        content_tokens = len(model.tokenizer.encode(chunk, add_special_tokens=False))
        input_tokens = len(
            model.tokenizer.encode(embedding_input, add_special_tokens=True)
        )
        print(
            f"\nChunk {index} | Characters: {len(chunk)} | Content tokens: {content_tokens} | Input tokens (prefix + specials): {input_tokens}\n{chunk}"
        )

    vectors = model.encode(inputs)
    print_vectors(
        MODEL, [f"Chunk {index}" for index in range(1, len(chunks) + 1)], vectors
    )


if __name__ == "__main__":
    main()

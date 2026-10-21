"""Clause boundaries, physical pages, and the actual Nomic token budget."""

import hashlib
import json
from pathlib import Path
import re

from langchain_core.documents import Document
from pypdf import PdfReader

from .embeddings import (
    CONTRACT,
    MAX_TOKENS,
    PREFIXES,
    EMBEDDING_MODEL,
    REVISION,
)
from .config import AppError

CHUNKING = {
    "version": "clauses-v1",
    "max_tokens": MAX_TOKENS,
    "overlap": 0,
    "whitespace": "line-strip",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def identity(path: Path):
    if not path.is_file():
        raise AppError(f"PDF not found: {path}. Check --corpus and its source PDFs.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tokenizer():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    try:
        snapshot = snapshot_download(
            EMBEDDING_MODEL,
            revision=REVISION,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.model"],
        )
        return AutoTokenizer.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False
        )
    except Exception as error:
        raise AppError(
            "Pinned Nomic tokenizer is unavailable locally. Run python -m rag_core.embeddings --download first."
        ) from error


def clauses(path):
    """Retain duplicate labels and page-spanning clauses as distinct occurrences."""
    try:
        pages = PdfReader(path).pages
        groups, current, label, heading = [], [], "", ""
        for page, pdf_page in enumerate(pages, 1):
            text = pdf_page.extract_text() or ""
            if not text.strip():
                raise AppError(
                    f"Physical PDF page {page} has no extractable text; inspect it before ingestion."
                )
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                match = re.match(r"^(\d+\.\d+)(?:\s|$)", line)
                section = re.match(r"^(\d+)\.\s*(?=\D)", line)
                if match or section:
                    if current:
                        groups.append((label, current))
                    label = match[1] if match else section[1]
                    if section:
                        heading = line
                        current = [(line, page)]
                    else:
                        current = [(heading, page), (line, page)] if heading else [(line, page)]
                else:
                    current.append((line, page))
        if current:
            groups.append((label, current))
        return groups
    except AppError:
        raise
    except Exception as error:
        raise AppError("Could not extract the selected PDF with pypdf.") from error


def _document_chunks(path: Path, encoder_tokenizer=None):
    pdf_hash = identity(path)
    tok = encoder_tokenizer or tokenizer()
    chunks = []
    for label, lines in clauses(path):
        # Do not index a title or bare section heading as standalone evidence.
        # Section headings are already carried with the numbered clauses.
        if not label or ("." not in label and len(lines) == 1):
            continue
        text, spans = "", []
        for line, page in lines:
            start = len(text)
            text += line + "\n"
            spans.append((start, len(text), page))
        offset = 0
        while offset < len(text):
            remaining = text[offset:]

            def length(part):
                return len(
                    tok(PREFIXES["document"] + part, add_special_tokens=True)["input_ids"]
                )

            if length(remaining) <= MAX_TOKENS:
                end = len(text)
            else:
                low, high = 1, len(remaining)
                while low < high:
                    mid = (low + high + 1) // 2
                    if length(remaining[:mid]) <= MAX_TOKENS:
                        low = mid
                    else:
                        high = mid - 1
                end = offset + low
                boundary = text.rfind(" ", offset, end)
                if boundary > offset:
                    end = boundary
            part = text[offset:end].strip()
            pages = [page for start, stop, page in spans if start < end and stop > offset]
            if not part or length(part) > MAX_TOKENS:
                raise AppError("Could not split this clause within the Nomic token budget.")
            index = len(chunks)
            chunk_id = digest([path.name, pdf_hash, CHUNKING, index])
            metadata = {
                "source": path.name,
                "pdf_sha256": pdf_hash,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "page_start": min(pages),
                "page_end": max(pages),
                "clause": label,
                "language": "el",
                "token_count": length(part),
                "chunk_config_id": digest(CHUNKING),
                "embedding_contract_id": digest(CONTRACT),
            }
            chunks.append(Document(id=chunk_id, page_content=part, metadata=metadata))
            offset = end
            while offset < len(text) and text[offset].isspace():
                offset += 1
    return chunks


def source_manifest(directory: Path):
    """Read source identity and physical page counts without loading embeddings."""
    if not directory.is_dir():
        raise AppError(f"Corpus directory not found: {directory}.")
    paths = sorted(directory.glob("*.pdf"))
    if not paths:
        raise AppError("Corpus has no PDF documents. Select --corpus with at least one PDF.")
    sources = []
    for path in paths:
        try:
            pages = len(PdfReader(path).pages)
        except Exception as error:
            raise AppError(f"Could not inspect PDF: {path.name}.") from error
        if not pages:
            raise AppError(f"PDF has no pages: {path.name}.")
        sources.append({"source": path.name, "pdf_sha256": identity(path), "pages": pages})
    return sources


def corpus_identity(directory: Path):
    return digest(source_manifest(directory))


def build_chunks(directory: Path, encoder_tokenizer=None):
    manifest = source_manifest(directory)
    corpus_id = digest(manifest)
    tok = encoder_tokenizer or tokenizer()
    chunks = []
    for source in manifest:
        for document in _document_chunks(directory / source["source"], tok):
            document.metadata.update(corpus_id=corpus_id, chunk_index=len(chunks))
            chunks.append(document)
    return chunks

"""Build the knowledge graph once: Claude reads each passage and lists entities and facts.

The result is saved to outputs/graph_rag/graph.json and reused by every question.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import unicodedata

from rag_core.config import AppError
from rag_core.corpus import digest
from rag_core.model import accepts_temperature
from rag_core.store import read_documents as read_documents
from rag_core.templating import for_settings
from .schema import ChunkExtraction, Extraction


def normalized(value):
    """Make names comparable: same Unicode form, case-folded, single spaces."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def index_path(settings):
    return Path(settings.extras.get("graph_index", "outputs/graph_rag/graph.json"))


def contract(documents, settings):
    """Everything the graph depends on. If any of it changes, the graph is stale."""
    return {
        "version": 2,
        "schema_id": digest(ChunkExtraction.model_json_schema()),
        "collection": settings.collection,
        "corpus_id": documents[0].metadata["corpus_id"],
        "sources": [
            list(pair)
            for pair in sorted({(d.metadata["source"], d.metadata["pdf_sha256"]) for d in documents})
        ],
        "embedding_contract_id": documents[0].metadata["embedding_contract_id"],
        "chunk_config_id": documents[0].metadata["chunk_config_id"],
        "extraction_model": settings.model,
        "prompt_id": for_settings("extraction", settings).prompt_id,
        "temperature": 0 if accepts_temperature(settings.model) else None,
        "passages": digest(
            [{"id": d.id, "text": d.page_content, "metadata": d.metadata} for d in documents]
        ),
    }


def validate_extraction(extracted, document):
    """Every fact must connect declared entities and quote the passage exactly."""
    ids = [normalized(entity.id) for entity in extracted.entities]
    if len(ids) != len(set(ids)) or not all(ids):
        raise AppError("Duplicate or empty canonical entity identifiers.")
    for relation in extracted.relationships:
        missing = {normalized(relation.source), normalized(relation.target)} - set(ids)
        if missing:
            raise AppError(
                "Extracted relationship has a missing endpoint: " + ", ".join(sorted(missing))
                + ". Add these exact identifiers to this response's entities array. "
                + "Declared identifiers: " + ", ".join(ids)
            )
        if not relation.quote.strip() or relation.quote not in document.page_content:
            raise AppError("Extracted quotation is not an exact substring of its PDF passage.")
    return extracted


def extract(document, chat, settings, catalog):
    """Ask Claude for one passage's entities and facts. Retry twice with the error as feedback."""
    lines = document.page_content.split("\n")
    payload = {
        "chunk_id": document.id,
        "source_lines": [{"line": number, "text": text} for number, text in enumerate(lines, 1)],
        "entity_catalog": catalog,
    }
    for attempt in range(3):
        try:
            selected, _ = chat.call(ChunkExtraction, for_settings("extraction", settings).text, payload)
            relationships = []
            for relation in selected.relationships:
                if not 1 <= relation.first_line <= relation.last_line <= len(lines):
                    raise AppError("Selected quotation line range is invalid.")
                # Claude picks line numbers; the app copies the exact text.
                quote = "\n".join(lines[relation.first_line - 1 : relation.last_line])
                relationships.append(
                    {**relation.model_dump(exclude={"first_line", "last_line"}), "quote": quote}
                )
            extracted = Extraction(entities=selected.entities, relationships=relationships)
            return validate_extraction(extracted, document)
        except AppError as error:
            if attempt == 2:
                raise
            payload["validation_feedback"] = (
                str(error) + " Return entities and relationships as JSON arrays of objects."
            )


def assemble(documents, manifest, extractions):
    """Merge every passage's entities by ID and number every fact."""
    entities, relationships = {}, []
    for document, extracted in zip(documents, extractions, strict=True):
        for entity in extracted.entities:
            key = normalized(entity.id)
            merged = entities.setdefault(
                key,
                {"id": key, "label": entity.label, "type": normalized(entity.type), "aliases": [], "chunk_ids": []},
            )
            if merged["type"] != normalized(entity.type):
                raise AppError(f"Canonical entity {key!r} has incompatible types.")
            merged["aliases"] = sorted(set(merged["aliases"] + entity.aliases + [entity.label, key]))
            merged["chunk_ids"] = list(dict.fromkeys(merged["chunk_ids"] + [document.id]))
        for relation in extracted.relationships:
            relationships.append(
                {
                    **relation.model_dump(),
                    "id": f"r{len(relationships)}",
                    "source": normalized(relation.source),
                    "target": normalized(relation.target),
                    "chunk_id": document.id,
                }
            )
    return {
        "complete": True,
        "contract": manifest,
        "fingerprint": digest({"contract": manifest, "entities": entities, "relationships": relationships}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entities": entities,
        "relationships": relationships,
    }


def prepare(documents, settings, chat, *, progress=lambda message: None):
    """Extract every passage in order, then save the graph. About one Claude call per passage."""
    if index_path(settings).exists():
        return load_index(documents, settings)
    extractions = []
    for number, document in enumerate(documents, 1):
        progress(f"Extracting passage {number}/{len(documents)}")
        # Show Claude the entities found so far, so it reuses "hearts" instead of inventing a new ID.
        catalog = list(
            {
                normalized(entity.id): {"id": normalized(entity.id), "label": entity.label, "type": entity.type, "aliases": entity.aliases}
                for extracted in extractions
                for entity in extracted.entities
            }.values()
        )
        extractions.append(extract(document, chat, settings, catalog))
    graph = assemble(documents, contract(documents, settings), extractions)
    index_path(settings).parent.mkdir(parents=True, exist_ok=True)
    index_path(settings).write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    return graph


def load_index(documents, settings):
    """Load the saved graph, refusing it if the PDF, model or prompt changed."""
    path = index_path(settings)
    if not path.exists():
        raise AppError("Graph index is missing. Run ./rag prepare --architecture graph_rag first.")
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise AppError("Graph index is malformed. Delete it and run prepare again.") from error
    if graph.get("contract") != contract(documents, settings):
        raise AppError(
            "Graph index is incompatible with this PDF, model or prompt. "
            "Use --option graph_index=<new path> and prepare again."
        )
    texts = {document.id: document.page_content for document in documents}
    for relation in graph["relationships"]:
        if relation["quote"] not in texts.get(relation["chunk_id"], ""):
            raise AppError(f"Graph fact {relation['id']} quotes text that is not in its PDF passage.")
    return graph

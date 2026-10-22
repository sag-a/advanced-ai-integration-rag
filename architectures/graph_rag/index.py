"""Explicit resumable extraction; never changes the Chroma collection."""

from contextlib import contextmanager
from datetime import datetime, timezone
from filelock import FileLock, Timeout
from pathlib import Path
import json
import os
import unicodedata

from rag_core.config import AppError
from rag_core.templating import for_settings, render
from rag_core.corpus import digest
from rag_core.model import accepts_temperature
from rag_core.store import read_documents as read_documents
from .schema import ChunkExtraction, Extraction

PROMPT = render("extraction").text


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def contract(documents, settings):
    return {
        "version": 2,
        "schema_id": digest(ChunkExtraction.model_json_schema()),
        "collection": settings.collection,
        "corpus_id": documents[0].metadata["corpus_id"],
        "sources": [
            list(pair)
            for pair in sorted(
                {(d.metadata["source"], d.metadata["pdf_sha256"]) for d in documents}
            )
        ],
        "embedding_contract_id": documents[0].metadata["embedding_contract_id"],
        "chunk_config_id": documents[0].metadata["chunk_config_id"],
        "extraction_model": settings.model,
        "prompt_id": for_settings("extraction", settings).prompt_id,
        # Recorded as sent: newer models no longer accept a temperature at all.
        "temperature": 0 if accepts_temperature(settings.model) else None,
        "passages": digest(
            [{"id": d.id, "text": d.page_content, "metadata": d.metadata} for d in documents]
        ),
    }


def validate_extraction(extracted, document):
    ids = [normalized(e.id) for e in extracted.entities]
    if len(ids) != len(set(ids)) or any(not i for i in ids):
        raise AppError("Duplicate or empty canonical entity identifiers.")
    for relation in extracted.relationships:
        missing = {
            normalized(endpoint) for endpoint in (relation.source, relation.target)
        } - set(ids)
        if missing:
            raise AppError(
                "Extracted relationship has a missing endpoint: "
                + ", ".join(sorted(missing))
                + ". Add these exact identifiers to this response's entities array. "
                + "Declared identifiers: "
                + ", ".join(ids)
            )
        if not relation.quote.strip() or relation.quote not in document.page_content:
            raise AppError("Extracted quotation is not an exact substring of its PDF passage.")
    return extracted


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def index_path(settings):
    return Path(settings.extras.get("graph_index", "outputs/graph_rag/graph.json"))


@contextmanager
def build_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(path.with_suffix(path.suffix + ".lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as error:
        raise AppError("Another graph preparation is running for this index.") from error
    try:
        yield
    finally:
        lock.release()


def assemble(documents, manifest, rows):
    entities, relations, aliases = {}, [], {}
    usage = []
    for document, row in zip(documents, rows, strict=True):
        extracted = validate_extraction(Extraction.model_validate(row["extraction"]), document)
        usage.append(row["observation"])
        for entity in extracted.entities:
            key, kind = normalized(entity.id), normalized(entity.type)
            if key in entities and entities[key]["type"] != kind:
                raise AppError(
                    f"Canonical entity {key!r} has incompatible types; inspect extraction cache before continuing."
                )
            target = entities.setdefault(
                key,
                {
                    "id": key,
                    "label": entity.label,
                    "type": kind,
                    "aliases": [],
                    "chunk_ids": [],
                },
            )
            target["aliases"] = sorted(
                set(target["aliases"] + entity.aliases + [entity.label, key])
            )
            target["chunk_ids"] = list(dict.fromkeys(target["chunk_ids"] + [document.id]))
        for relation in extracted.relationships:
            relations.append(
                {
                    **relation.model_dump(),
                    "id": f"r{len(relations)}",
                    "source": normalized(relation.source),
                    "target": normalized(relation.target),
                    "chunk_id": document.id,
                }
            )
    for key, entity in entities.items():
        for alias in entity["aliases"]:
            aliases.setdefault(normalized(alias), set()).add(key)
    return {
        "complete": True,
        "contract": manifest,
        "fingerprint": digest(
            {"contract": manifest, "entities": entities, "relationships": relations}
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "entities": entities,
        "relationships": relations,
        "ambiguous_aliases": {a: sorted(keys) for a, keys in aliases.items() if len(keys) > 1},
        "indexing_observations": usage,
        "extraction_schema_id": digest(ChunkExtraction.model_json_schema()),
    }


def load_index(documents, settings):
    try:
        graph = json.loads(index_path(settings).read_text(encoding="utf-8"))
        if not graph.get("complete") or graph["contract"] != contract(documents, settings):
            raise AppError(
                "Graph index is incompatible. Prepare a new --index path for changed inputs or extraction configuration."
            )
        expected = digest(
            {
                "contract": graph["contract"],
                "entities": graph["entities"],
                "relationships": graph["relationships"],
            }
        )
        if graph["fingerprint"] != expected:
            raise AppError(
                "Graph index fingerprint is invalid; inspect or prepare a new index."
            )
        by_id = {d.id: d for d in documents}
        for key, entity in graph["entities"].items():
            if key != normalized(entity["id"]) or not set(entity["chunk_ids"]) <= by_id.keys():
                raise AppError("Graph contains foreign entity source IDs.")
        aliases = {}
        for key, entity in graph["entities"].items():
            for alias in entity["aliases"]:
                aliases.setdefault(normalized(alias), set()).add(key)
        expected_ambiguous = {a: sorted(keys) for a, keys in aliases.items() if len(keys) > 1}
        if graph["ambiguous_aliases"] != expected_ambiguous:
            raise AppError("Graph ambiguous-alias map is inconsistent with its entities.")
        relationship_ids = [r["id"] for r in graph["relationships"]]
        if len(relationship_ids) != len(set(relationship_ids)):
            raise AppError("Graph contains duplicate relationship IDs.")
        for relation in graph["relationships"]:
            if relation["chunk_id"] not in by_id:
                raise AppError("Graph contains foreign chunk IDs.")
            if (
                relation["source"] not in graph["entities"]
                or relation["target"] not in graph["entities"]
            ):
                raise AppError("Graph contains a missing endpoint.")
            if (
                not relation["quote"].strip()
                or relation["quote"] not in by_id[relation["chunk_id"]].page_content
            ):
                raise AppError("Graph contains an invalid quotation.")
            if relation["polarity"] not in {"positive", "negative", "conditional"}:
                raise AppError("Graph contains an invalid polarity.")
        return graph
    except AppError:
        raise
    except FileNotFoundError as error:
        raise AppError(
            "Graph index is missing. Run ./rag prepare --architecture graph_rag first."
        ) from error
    except Exception as error:
        raise AppError(
            "Graph index is malformed. Inspect it or prepare a new --index path."
        ) from error


def prepare(documents, settings, chat, *, resume=False, progress=lambda message: None):

    manifest = contract(documents, settings)
    cache = index_path(settings).with_name(index_path(settings).name + ".cache")
    with build_lock(index_path(settings)):
        if index_path(settings).exists():
            return load_index(documents, settings)
        marker = cache / "manifest.json"
        if marker.exists():
            if json.loads(marker.read_text(encoding="utf-8")) != manifest:
                raise AppError("Extraction cache is incompatible. Select a new --index path.")
            if not resume:
                raise AppError(
                    "Interrupted preparation found. Use prepare --resume explicitly."
                )
        else:
            write_json(marker, manifest)
        rows, catalog = [], {}
        for document in documents:
            path = cache / (digest(document.id) + ".json")
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                progress(
                    f"Extracting passage {document.metadata['chunk_index'] + 1}/{len(documents)}"
                )
                lines = document.page_content.split("\n")
                payload = {
                    "chunk_id": document.id,
                    "source_lines": [
                        {"line": i, "text": text} for i, text in enumerate(lines, 1)
                    ],
                    "entity_catalog": list(catalog.values()),
                }
                attempts = []
                for attempt in range(3):
                    try:
                        selected, observation = chat.call(
                            ChunkExtraction, for_settings("extraction", settings).text, payload
                        )
                        attempts.append(observation)
                        relationships = []
                        for relation in selected.relationships:
                            if not 1 <= relation.first_line <= relation.last_line <= len(lines):
                                raise AppError("Selected quotation line range is invalid.")
                            relationships.append(
                                {
                                    **relation.model_dump(exclude={"first_line", "last_line"}),
                                    "quote": "\n".join(
                                        lines[relation.first_line - 1 : relation.last_line]
                                    ),
                                }
                            )
                        extracted = Extraction(
                            entities=selected.entities, relationships=relationships
                        )
                        validate_extraction(extracted, document)
                        break
                    except AppError as error:
                        attempts.append({"validation_error": str(error), "usage": None})
                        # Failed work is not an accepted extraction, but its available
                        # usage must remain inspectable after interruption/exhaustion.
                        write_json(
                            cache / (digest(document.id) + ".failed.json"),
                            {"chunk_id": document.id, "attempts": attempts},
                        )
                        if attempt == 2:
                            raise
                        progress(
                            f"Retrying invalid/incomplete extraction ({attempt + 1}/2); accepted passages remain cached."
                        )
                        payload["validation_feedback"] = (
                            str(error)
                            + " Return BOTH entities and relationships as actual JSON arrays of objects, NEVER as JSON-encoded strings. Select valid source line ranges from the numbered input."
                        )
                row = {
                    "extraction": extracted.model_dump(),
                    "observation": {"attempts": attempts},
                }
                # Validate cross-passage identifiers BEFORE accepting the cache row.
                assemble(documents[: len(rows) + 1], manifest, rows + [row])
                write_json(path, row)
            rows.append(row)
            partial = assemble(documents[: len(rows)], manifest, rows)
            catalog = {
                key: {k: e[k] for k in ("id", "label", "type", "aliases")}
                for key, e in partial["entities"].items()
            }
        graph = assemble(documents, manifest, rows)
        write_json(index_path(settings), graph)
        return graph

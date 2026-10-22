"""Graph RAG retrieval in four steps, with no LLM call:

1. Seeds: the 3 passages closest in meaning to the question (a normal vector search).
2. Start: the entities those passages mention, entities named in the question first.
3. Walk: follow relationships for up to 2 hops. Each one points to the passage it came from.
4. Rank: keep the seeds, then add the graph's passages, closest in meaning first.

Claude later reads the original PDF passages. The graph only helps find them.
"""

from dataclasses import replace
import re
import time
from uuid import uuid4

from rag_core.callbacks import report_stage
from rag_core.retrieval import RetrievalResult
from rag_core.store import retrieve
from .index import load_index, normalized, read_documents

SEEDS = 3  # passages from the first vector search
MAX_START = 4  # entities the walk starts from
MAX_HOPS = 2  # relationship steps away from them
MAX_FACTS = 30  # relationships followed in total
MAX_PASSAGES = 12  # seeds plus graph passages, before the final ranking


def mentioned(alias, question):
    """True when the alias appears in the question as a whole word."""
    pattern = r"(?<!\w)" + re.escape(normalized(alias)) + r"(?!\w)"
    return re.search(pattern, normalized(question)) is not None


class GraphRetriever:
    def __init__(self, collection, settings, *, reporter=None):
        self.collection, self.settings, self.reporter = collection, settings, reporter
        documents = read_documents(collection)
        self.passages = {document.id: document for document in documents}
        self.graph = load_index(documents, settings)
        self.request_id = ""

    def log(self, event, **data):
        """Feed the classroom panels (--process and the UI), when they are attached."""
        if self.reporter:
            self.reporter(event, {"request_id": self.request_id, **data})

    def __call__(self, question):
        started = time.perf_counter()
        self.request_id = uuid4().hex[:8]
        self.log("retrieval_started", top_k=self.settings.top_k)

        # 1. Seeds
        first = retrieve(self.collection, replace(self.settings, top_k=min(SEEDS, self.settings.top_k)), question)
        seeds = first.documents
        report_stage("graph-vector-seeds", f"{len(seeds)} seed passages")
        self.log("seeds_selected", documents=seeds, elapsed_seconds=first.elapsed_seconds)

        # 2. Start
        start = self.start_entities(seeds, question)
        self.log("anchors_selected", anchors=[self.graph["entities"][key]["label"] for key in start], ambiguous_aliases=[])

        # 3. Walk
        facts = self.walk(start)
        seed_ids = [document.id for document in seeds]
        added = list(dict.fromkeys(f["chunk_id"] for f in facts if f["chunk_id"] not in seed_ids))
        added = added[: MAX_PASSAGES - len(seed_ids)]
        report_stage("graph-traversal", f"{len(start)} start entities, {len(facts)} facts, {len(added)} new passages")
        known = set(seed_ids)
        for hop in range(1, MAX_HOPS + 1):
            hop_facts = [fact for fact in facts if fact["hop"] == hop]
            if hop_facts:
                new = {fact["chunk_id"] for fact in hop_facts} - known
                known |= new
                self.log("hop_completed", hop=hop, relationships=hop_facts, new_passages=len(new))

        # 4. Rank
        documents = self.rank(question, seeds, added)
        report_stage("graph-passage-selection", f"{len(documents)} passages selected")
        details = self.details(seeds, start, facts, added, documents)
        elapsed = time.perf_counter() - started
        self.log(
            "passages_selected",
            documents=documents,
            elapsed_seconds=elapsed,
            candidate_count=len(seeds) + len(added),
            hops_used=max((fact["hop"] for fact in facts), default=0),
            **{key: details[key] for key in ("seed_ids", "graph_used", "selected_expansion_count", "traversed_count", "index_fingerprint", "limits")},
        )
        return RetrievalResult(documents, elapsed, details)

    def start_entities(self, seeds, question):
        """Entities mentioned in the seed passages. Named in the question first, then by seed rank."""
        seed_ids = [document.id for document in seeds]
        entities = self.graph["entities"]

        def order(key):
            named = any(mentioned(alias, question) for alias in entities[key]["aliases"])
            best_seed = min(seed_ids.index(i) for i in entities[key]["chunk_ids"] if i in seed_ids)
            return (not named, best_seed, key)

        linked = [key for key, entity in entities.items() if set(entity["chunk_ids"]) & set(seed_ids)]
        return sorted(linked, key=order)[:MAX_START]

    def walk(self, start):
        """Breadth-first: hop 1 takes every fact touching a start entity, hop 2 every fact touching those."""
        facts, followed, reached, frontier = [], set(), set(start), set(start)
        for hop in range(1, MAX_HOPS + 1):
            next_frontier = set()
            for fact in self.graph["relationships"]:
                if len(facts) == MAX_FACTS:
                    break
                if fact["id"] in followed or not {fact["source"], fact["target"]} & frontier:
                    continue
                followed.add(fact["id"])
                passage = self.passages[fact["chunk_id"]].metadata
                facts.append(
                    {**fact, "hop": hop, "page_start": passage["page_start"], "page_end": passage["page_end"], "clause": passage.get("clause", "")}
                )
                next_frontier |= {fact["source"], fact["target"]} - reached
            reached |= next_frontier
            frontier = next_frontier
        return facts

    def rank(self, question, seeds, added):
        """Seeds first. Then graph passages, closest in meaning first. Top up with vector results if short."""
        chosen = list(seeds)
        if added:
            result = self.collection.query(
                query_texts=[question],
                n_results=len(added),
                where={"$and": [{"corpus_id": {"$eq": self.graph["contract"]["corpus_id"]}}, {"chunk_id": {"$in": added}}]},
                include=["distances"],
            )
            for identifier, distance in zip(result["ids"][0], result["distances"][0], strict=True):
                passage = self.passages[identifier]
                chosen.append(passage.model_copy(update={"metadata": {**passage.metadata, "distance": distance}}))
        if seeds and len(chosen) < self.settings.top_k:
            ids = {document.id for document in chosen}
            chosen += [d for d in retrieve(self.collection, self.settings, question).documents if d.id not in ids]
        return chosen[: self.settings.top_k]

    def details(self, seeds, start, facts, added, documents):
        """What the UI graph panel and the evaluation report show."""
        seed_ids = [document.id for document in seeds]
        selected_ids = [document.id for document in documents]
        selected_added = [i for i in selected_ids if i in added]
        touched = sorted({key for fact in facts for key in (fact["source"], fact["target"])})
        return {
            "architecture": "graph_rag",
            "index_fingerprint": self.graph["fingerprint"],
            "seed_ids": seed_ids,
            "anchors": start,
            "relationships": facts,
            "entities": [self.graph["entities"][key] for key in touched],
            "candidate_ids": seed_ids + added,
            "expansion_count": len(added),
            "selected_expansion_count": len(selected_added),
            "selected_graph_ids": selected_added,
            "traversed_count": len(facts),
            "graph_used": bool(added),
            "explanation": f"Graph added {len(added)} candidate passages via {len(facts)} relationships."
            if added
            else "Graph expansion contributed no additional passages; using vector results.",
            "limits": {"anchors": MAX_START, "hops": MAX_HOPS, "relationships": MAX_FACTS, "candidates": MAX_PASSAGES},
            "selected_ids": selected_ids,
            "seed_passages": [document.model_dump() for document in seeds],
            "selected_passages": [document.model_dump() for document in documents],
        }

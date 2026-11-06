"""Grade the capstone RAG app step by step. Offline: no model download, no API call."""

import pytest

from exercises.fakes import WordOverlap

SOURCES = ["billing.md", "cancellations.md", "internet_support.md", "refunds.md", "refunds_2023.md"]


def flat(text):
    return " ".join(text.split())


@pytest.fixture
def chunks(lab):
    return lab.chunk_documents(lab.load_documents())


@pytest.fixture
def collection(lab, chunks, tmp_path):
    collection = lab.open_collection(path=tmp_path / "db", embedding_function=WordOverlap())
    lab.index_chunks(collection, chunks)
    return collection


# Step 2: load the files


def test_step2_every_file_is_loaded_with_its_source(lab):
    documents = lab.load_documents()
    assert sorted(d["metadata"]["source"] for d in documents) == SOURCES


def test_step2_front_matter_becomes_metadata(lab):
    documents = {d["metadata"]["source"]: d for d in lab.load_documents()}
    old = documents["refunds_2023.md"]
    assert old["metadata"].get("status") == "superseded", "put the front matter lines in the metadata"
    assert old["metadata"]["version"] == 1, "store version as an int, not a string"
    assert documents["internet_support.md"]["metadata"]["category"] == "technical"
    assert old["text"].startswith("# "), "the front matter must not stay in the document text"


# Step 3: create chunks


def test_step3_chunks_have_ids_sources_and_sections(chunks):
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids)), "chunk IDs must be unique"
    assert all(c["metadata"]["source"] in SOURCES and c["metadata"]["section"] for c in chunks)
    assert "refunds.md#0" in ids


def test_step3_a_rule_stays_with_its_exception(chunks):
    rule = next(c for c in chunks if "within 14 days of purchase" in flat(c["text"]) and "refunds.md" in c["id"])
    assert "already used the service" in flat(rule["text"]), (
        "the refund rule and its exception are in different chunks: split at # and ## only"
    )


def test_step3_long_sections_are_split_and_keep_their_heading(chunks):
    wifi = [c for c in chunks if c["metadata"]["section"] == "Wi-Fi troubleshooting"]
    assert len(wifi) >= 2, "the Wi-Fi section is longer than CHUNK_SIZE: split it by size"
    assert all("## Wi-Fi troubleshooting" in c["text"] for c in wifi), "every piece needs its section heading"


def test_step3_small_chunks_separate_the_rule_from_its_exception(lab):
    small = lab.chunk_documents(lab.load_documents(), chunk_size=150)
    rule = next(c for c in small if "within 14 days of purchase" in flat(c["text"]) and "refunds.md" in c["id"])
    assert "already used the service" not in flat(rule["text"])


# Step 4: store embeddings in ChromaDB


def test_step4_reindexing_adds_no_duplicates(lab, chunks, collection):
    assert lab.index_chunks(collection, chunks) == len(chunks)
    assert collection.count() == len(chunks)


def test_step4_records_are_stored_under_their_chunk_ids(chunks, collection):
    record = collection.get(ids=["refunds_2023.md#0"])
    assert record["ids"] == ["refunds_2023.md#0"], "store each chunk under its own ID, not a random one"
    assert record["metadatas"][0]["source"] == "refunds_2023.md"


def test_step4_rebuild_removes_old_chunks(lab, tmp_path):
    first = lab.open_collection(path=tmp_path / "db", embedding_function=WordOverlap())
    lab.index_chunks(first, lab.chunk_documents(lab.load_documents(), chunk_size=150))
    rebuilt = lab.open_collection(path=tmp_path / "db", embedding_function=WordOverlap(), rebuild=True)
    assert rebuilt.count() == 0


# Step 5: retrieve relevant chunks


def test_step5_retrieve_returns_n_results_with_distances(lab, collection):
    found = lab.retrieve(collection, "How many days do I have to get a full refund?", n_results=2)
    assert len(found) == 2
    assert {"id", "text", "metadata", "distance"} <= set(found[0])
    assert found[0]["distance"] <= found[1]["distance"]


def test_step5_a_filter_limits_the_results(lab, tmp_path):
    collection = lab.open_collection(path=tmp_path / "db", embedding_function=WordOverlap())
    collection.upsert(
        ids=["old", "new"],
        documents=["full refund within 30 days", "full refund within 14 days"],
        metadatas=[{"status": "superseded"}, {"status": "current"}],
    )
    found = lab.retrieve(collection, "full refund", n_results=2, where={"status": "current"})
    assert [c["id"] for c in found] == ["new"], "pass the where filter on to collection.query"


# Step 6: generate a grounded answer


def test_step6_the_prompt_numbers_its_sources(lab, collection):
    found = lab.retrieve(collection, "How many days do I have to get a full refund?", n_results=2)
    prompt = lab.build_prompt("How many days do I have to get a full refund?", found)
    assert "[1]" in prompt and "[2]" in prompt
    assert found[0]["metadata"]["source"] in prompt
    assert "How many days do I have to get a full refund?" in prompt


def test_step6_the_instructions_require_citations_and_the_refusal(lab):
    assert lab.REFUSAL in lab.SYSTEM_PROMPT, "tell the model the exact sentence to use when the answer is missing"
    assert "[" in lab.SYSTEM_PROMPT and ".md" in lab.SYSTEM_PROMPT, "show the citation format, e.g. [refunds.md]"


def test_step6_dry_run_calls_no_model(lab, collection, monkeypatch):
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: pytest.fail("--dry-run must not call the model"))
    found = lab.retrieve(collection, "red light", n_results=1)
    shown = lab.generate_answer("What does a red light mean?", found, dry_run=True)
    assert "What does a red light mean?" in shown and found[0]["text"].split("\n")[0] in shown


# Step 7: evaluate and improve


def test_step7_citations_are_extracted(lab):
    assert lab.cited_sources("Yes [refunds.md], but see [billing.md] and [refunds.md].") == ["billing.md", "refunds.md"]


def test_step7_the_evaluation_runs(lab, collection):
    rows = lab.evaluate(collection, n_results=3)
    assert len(rows) == len(lab.EVAL_QUESTIONS)
    assert rows[-1]["hit"] is None, "the unanswerable question has no evidence to find"


def test_step7_you_added_a_question_with_real_evidence(lab):
    assert len(lab.EVAL_QUESTIONS) >= 7, "add one of your own questions to EVAL_QUESTIONS"
    docs = flat(" ".join(d["text"] for d in lab.load_documents()))
    for question, evidence in lab.EVAL_QUESTIONS:
        for phrase in evidence or []:
            assert phrase in docs, f"{question!r}: {phrase!r} is not in the documents (copy it exactly)"


def test_step7_the_in_memory_collection_starts_empty_every_time(lab, chunks):
    first = lab.open_collection(path=None, embedding_function=WordOverlap(), rebuild=True)
    lab.index_chunks(first, chunks)
    again = lab.open_collection(path=None, embedding_function=WordOverlap(), rebuild=True)
    assert again.count() == 0


# Step 8: the pipeline as a LangGraph graph


def test_step8_route_picks_the_next_node(lab):
    assert lab.route({"chunks": [{"distance": 0.2}]}) == "answer"
    assert lab.route({"chunks": [{"distance": 0.9}]}) == "refuse"
    assert lab.route({"chunks": []}) == "refuse"


def test_step8_a_far_question_is_refused_without_the_model(lab, collection, monkeypatch):
    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: pytest.fail(
        "the graph called the model for a question no chunk is close to: use route() after retrieve"))
    path, state = lab.ask_graph(collection, "What are your opening hours?")
    assert path == ["retrieve", "refuse"] and state["answer"] == lab.REFUSAL


def test_step8_a_close_question_goes_to_the_model(lab, collection):
    question = "Customers can receive a full refund if they cancel within 14 days of purchase."
    path, state = lab.ask_graph(collection, question, dry_run=True)
    assert path == ["retrieve", "answer"] and question in state["answer"]

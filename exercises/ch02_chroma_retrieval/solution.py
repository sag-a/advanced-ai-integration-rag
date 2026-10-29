"""Chapter 2 exercise: reference solution for the three Chroma queries."""

from uuid import uuid4

import chromadb
from chromadb.config import Settings

# One chunk per policy statement. Metadata says where a chunk belongs and whether it
# is still valid: version 1 of the refund rule was replaced by version 2.
POLICIES = [
    ("refund-rule-v2", "Customers can receive a full refund if they cancel within 14 days of purchase.",
     {"category": "billing", "version": 2, "status": "current"}),
    ("refund-exception-v2", "Customers who have already used the service are not eligible for a full refund.",
     {"category": "billing", "version": 2, "status": "current"}),
    ("refund-rule-v1", "Customers can receive a full refund if they cancel within 30 days of purchase.",
     {"category": "billing", "version": 1, "status": "superseded"}),
    ("duplicate-charge", "If you were charged twice, record both payment dates and contact the billing team.",
     {"category": "billing", "version": 2, "status": "current"}),
    ("payment-methods", "Customers can pay by debit card, credit card or bank transfer.",
     {"category": "billing", "version": 2, "status": "current"}),
    ("wifi", "If Wi-Fi stops working, restart the router and check that Wi-Fi is enabled.",
     {"category": "technical", "version": 2, "status": "current"}),
    ("router-light", "A red router light may indicate a connection problem.",
     {"category": "technical", "version": 2, "status": "current"}),
    ("cancel-steps", "Submit a cancellation request through your online account.",
     {"category": "account", "version": 2, "status": "current"}),
]

# ---------------------------------------------------------------------------
# The three settings, fixed.
# ---------------------------------------------------------------------------

# Exercise 1: 1 returns only the exception. 3 returns the rule and its exception
# (with Nomic, 2 already does). 5 works too, but adds unrelated chunks to read.
N_RESULTS = 3

# Exercise 2: search the category the question belongs to.
BILLING_FILTER = {"category": "billing"}

# Exercise 3: keep the category and add the version status. {"version": {"$gte": 2}}
# instead of the status works too.
POLICY_FILTER = {"$and": [{"category": "billing"}, {"status": "current"}]}


# ---------------------------------------------------------------------------
# The questions. You don't need to change anything below.
# ---------------------------------------------------------------------------


def make_collection(embedding_function=None):
    """An in-memory Chroma collection holding POLICIES. No Docker server needed."""
    if embedding_function is None:
        from modules.c02_vector_databases.nomic_embedding_function import NomicEmbeddingFunction

        embedding_function = NomicEmbeddingFunction()
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection(
        "policies-" + uuid4().hex[:8],
        embedding_function=embedding_function,
        configuration={"hnsw": {"space": "cosine"}},
    )
    collection.add(
        ids=[policy_id for policy_id, _, _ in POLICIES],
        documents=[text for _, text, _ in POLICIES],
        metadatas=[metadata for _, _, metadata in POLICIES],
    )
    return collection


def search(collection, question, n_results, where=None):
    """Return the retrieved chunks as dicts with id, text and metadata."""
    result = collection.query(query_texts=[question], n_results=n_results, where=where)
    return [
        {"id": chunk_id, "text": text, "metadata": metadata}
        for chunk_id, text, metadata in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], strict=True
        )
    ]


def missing_information(collection):
    return search(collection, "Can I get a full refund within 14 days if I already used the service?", N_RESULTS)


def wrong_category(collection):
    return search(collection, "I was charged twice on my card", 2, where=BILLING_FILTER)


def outdated_information(collection):
    return search(collection, "How many days do I have to get a full refund?", 3, where=POLICY_FILTER)

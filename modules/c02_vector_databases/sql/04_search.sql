SELECT id, content, metadata,
       embedding <=> %(query)s AS cosine_distance,
       1 - (embedding <=> %(query)s) AS cosine_similarity
FROM rag_pgvector_demo_nomic_v2.documents
-- Order by the distance operator directly, ascending, for HNSW eligibility.
-- ORDER BY cosine_similarity DESC would not be the index-compatible form.
ORDER BY embedding <=> %(query)s
LIMIT %(top_k)s;

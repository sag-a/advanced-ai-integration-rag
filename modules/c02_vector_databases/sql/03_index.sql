-- Match this operator class to cosine-distance searches with <=>.
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
ON rag_pgvector_demo_nomic_v2.documents USING hnsw (embedding vector_cosine_ops);

ANALYZE rag_pgvector_demo_nomic_v2.documents;

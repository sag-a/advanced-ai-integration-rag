-- Match this operator class to cosine-distance searches with <=>.
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
ON rag_pgvector_demo_nomic_v2.documents USING hnsw (embedding vector_cosine_ops);

-- The keyword index: GIN maps each stemmed word to the rows that contain it.
CREATE INDEX IF NOT EXISTS documents_content_tsv_gin
ON rag_pgvector_demo_nomic_v2.documents USING gin (content_tsv);

ANALYZE rag_pgvector_demo_nomic_v2.documents;

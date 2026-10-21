-- Psycopg named placeholders: values are bound, never interpolated into SQL.
INSERT INTO rag_pgvector_demo_nomic_v2.documents (id, content, metadata, embedding)
VALUES (%(id)s, %(content)s, %(metadata)s, %(embedding)s)
ON CONFLICT (id) DO UPDATE SET
    content = EXCLUDED.content,
    metadata = EXCLUDED.metadata,
    embedding = EXCLUDED.embedding;

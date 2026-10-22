-- Execute once per database; rerunning does not drop existing objects.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS rag_pgvector_demo_nomic_v2;

-- One compatible embedding space per table: normalized Nomic v2 MoE, 768D.
CREATE TABLE IF NOT EXISTS rag_pgvector_demo_nomic_v2.documents (
    id integer PRIMARY KEY,
    content text NOT NULL,
    metadata jsonb NOT NULL,
    embedding vector(768) NOT NULL
);

-- Keyword search: PostgreSQL keeps stemmed English words beside each passage.
-- A generated column stays in sync with content; inserts never set it.
ALTER TABLE rag_pgvector_demo_nomic_v2.documents
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

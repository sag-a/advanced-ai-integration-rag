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

-- Hybrid search in one query: a dense ranking and a keyword ranking, fused with RRF (k = 60).
-- %(query)s is the 768D question vector; %(text)s is the question text.
WITH terms AS (
    -- plainto_tsquery stems the words and joins them with AND (&). Switching to
    -- OR (|) lets a passage match on any one word, like TF-IDF or BM25.
    SELECT replace(plainto_tsquery('english', %(text)s)::text, '&', '|')::tsquery AS q
),
dense AS (
    SELECT id, row_number() OVER (ORDER BY distance) AS rank
    FROM (
        SELECT id, embedding <=> %(query)s AS distance
        FROM rag_pgvector_demo_nomic_v2.documents
        ORDER BY embedding <=> %(query)s
        LIMIT %(candidate_k)s
    ) nearest
),
keyword AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(content_tsv, q) DESC, id) AS rank
    FROM rag_pgvector_demo_nomic_v2.documents, terms
    WHERE content_tsv @@ q
    ORDER BY ts_rank_cd(content_tsv, q) DESC, id
    LIMIT %(candidate_k)s
)
SELECT id,
       coalesce(1.0 / (60 + dense.rank), 0) + coalesce(1.0 / (60 + keyword.rank), 0) AS rrf_score,
       dense.rank AS dense_rank,
       keyword.rank AS keyword_rank,
       ts_headline('english', content, q, 'StartSel=[, StopSel=], HighlightAll=true') AS highlighted
FROM dense
FULL OUTER JOIN keyword USING (id)
JOIN rag_pgvector_demo_nomic_v2.documents USING (id)
CROSS JOIN terms
ORDER BY rrf_score DESC, id
LIMIT %(top_k)s;

-- pgvector HNSW index and semantic search RPC for conversation_turns
-- Requires: pgvector extension (enabled in 001_init.sql)

CREATE INDEX IF NOT EXISTS idx_turns_embedding_hnsw
ON conversation_turns
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- RPC function for similarity search, callable via supabase.rpc()
CREATE OR REPLACE FUNCTION match_conversation_turns(
    query_embedding vector(1536),
    match_user_id text,
    match_count int DEFAULT 3
)
RETURNS TABLE (
    id bigint,
    user_id text,
    raw_text text,
    intent_type text,
    tool_name text,
    assistant_speak text,
    context_json jsonb,
    created_at timestamptz,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        ct.id,
        ct.user_id,
        ct.raw_text,
        ct.intent_type,
        ct.tool_name,
        ct.assistant_speak,
        ct.context_json,
        ct.created_at,
        1 - (ct.embedding <=> query_embedding) AS similarity
    FROM conversation_turns ct
    WHERE ct.user_id = match_user_id
      AND ct.embedding IS NOT NULL
    ORDER BY ct.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

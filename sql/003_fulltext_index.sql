-- Optional full-text search index on conversation_turns as a lightweight
-- alternative/complement to Elasticsearch.  Uses PostgreSQL built-in
-- tsvector and GIN indexing.

ALTER TABLE conversation_turns
  ADD COLUMN IF NOT EXISTS tsv tsvector
  GENERATED ALWAYS AS (
    to_tsvector('english', coalesce(raw_text, '') || ' ' || coalesce(assistant_speak, ''))
  ) STORED;

CREATE INDEX IF NOT EXISTS idx_turns_tsv ON conversation_turns USING gin(tsv);

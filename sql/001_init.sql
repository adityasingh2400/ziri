-- Ziri initial schema for Supabase/Postgres
create extension if not exists vector;

create table if not exists sessions (
  id bigserial primary key,
  user_id text not null,
  device_id text not null,
  room text not null,
  raw_text text not null,
  request_ts timestamptz not null,
  status text not null default 'RECEIVED',
  route_json jsonb,
  tool_json jsonb,
  response_json jsonb,
  processed_at timestamptz,
  created_at timestamptz not null default now()
);

create index if not exists idx_sessions_user_created on sessions(user_id, created_at desc);

create table if not exists conversation_turns (
  id bigserial primary key,
  user_id text not null,
  raw_text text not null,
  intent_type text not null,
  tool_name text not null,
  assistant_speak text,
  private_note text,
  context_json jsonb not null default '{}'::jsonb,
  embedding vector(1536),
  created_at timestamptz not null default now()
);

create index if not exists idx_turns_user_created on conversation_turns(user_id, created_at desc);

create table if not exists user_preferences (
  user_id text primary key,
  preferences jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

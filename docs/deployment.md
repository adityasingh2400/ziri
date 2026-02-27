# Deployment Notes

## AWS EC2 (t3.medium)

1. Provision EC2 instance.
2. Install Docker or Python runtime.
3. Configure `.env` with Bedrock/Supabase/Spotify values.
4. Run `docker compose up -d --build`.
5. Put API behind ALB or Nginx with TLS.

## AWS Lambda (Alternative)

- Package with Mangum adapter (not included in this starter).
- Place behind API Gateway.
- Keep warm if low-latency voice responses are required.

## Cost Guidance

- Keep routing/model calls on Sonnet class models for real-time turns.
- Run long transcript cleanup jobs asynchronously (nightly) if using larger models.
- Prefer AWS Polly neural voices for low per-request TTS cost.

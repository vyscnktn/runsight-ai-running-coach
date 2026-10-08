# n8n workflow

`runsight_workflow.json` is a sanitised export of the production workflow (80 nodes): post-workout flow, onboarding, RAG ingestion and the weekly learning loop. See [docs/architecture.md](../docs/architecture.md) for what each flow does.

## What was removed from the export

Credential IDs (names kept), webhook IDs, Google Sheet/Drive IDs and URLs, e-mail addresses, the Lambda URL, execution sample data (`pinData`) and the instance ID. [`scripts/sanitize_workflow.py`](../scripts/sanitize_workflow.py) does this and CI re-checks the result.

## Import

1. Import `runsight_workflow.json` into n8n (2.x).
2. Create credentials and bind them to the nodes: Strava OAuth2, Telegram Bot, Google (Sheets, Drive, Gmail), Notion, OpenAI (agent + embeddings), Google Gemini, Pinecone.
3. Replace the placeholders: `<GOOGLE_FILE_ID_n>` (create the sheets/folder yourself — users, trainings, weekly summaries — and select them in the Google nodes), and `https://<your-lambda-function-url>…` in the GRU HTTP Request node.
4. The three `HTTP Request` nodes that fetch system prompts point to `raw.githubusercontent.com/vyscnktn/runsight-ai-running-coach/main/prompts/…`. Change them to your own fork/branch if you modify the prompts.
5. OAuth redirect: add `https://<your-n8n-host>/rest/oauth2-credential/callback` in Google Cloud; in Strava enter only the domain as "Authorization Callback Domain". A Google app in "Testing" mode expires tokens after 7 days — publish it.
6. Fill the Pinecone index once through the form trigger (sources: [rag/SOURCES.md](../rag/SOURCES.md)).
7. **Publish** the workflow (in n8n 2.x, triggers run only on the published version).

Self-hosted environment variables: see [`.env.example`](../.env.example). Self-hosted n8n has no Python runner, hence JavaScript code nodes.

## Note on naming

Node names are partly Turkish (the product's user-facing language). They are referenced by expressions and connections, so they were left unchanged rather than risk breaking the flow.

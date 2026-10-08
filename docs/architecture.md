# Architecture

## Components

| Component | Technology | Responsibility |
|---|---|---|
| Trigger | Strava webhook, Telegram bot, weekly cron | Starts the three flows |
| Orchestration | n8n (self-hosted, Docker, Hetzner VPS, reverse proxy + TLS) | Control flow, branching, retries, persistence |
| Inference | AWS Lambda, container image (ECR), TensorFlow/Keras | Classify one workout, return probability + HR percentiles |
| Agents | n8n LangChain Agent nodes; Gemini and OpenAI chat models | Onboarding, post-workout report, weekly review |
| Knowledge base | Pinecone (serverless, 1536-d, cosine), OpenAI embeddings | RAG over sports-science sources, used as an agent tool |
| State | Google Sheets (users, trainings, weekly summaries), Notion (training plans), Drive (log files) | Operational data store |
| Delivery | Telegram, Gmail | Reports, human-in-the-loop questions, plan approval |

## Flows

### Post-workout
1. `Strava Trigger` fires for a new activity; the activity streams are fetched.
2. Non-runs are filtered out.
3. The streams are POSTed unchanged to the Lambda → `training_zone`, `anaerobic_probability`, `confidence`, `hr_p10`, `hr_p95`.
4. A code node computes **cardiac drift** (efficiency factor = speed / heart rate, first half vs second half of the run; `null` for missing data or average speed < 3 km/h).
5. A code node computes **plan week / weekday** from the user's `plan_start_date`, so the LLM never does date arithmetic.
6. **Routing on confidence:** `≥ 0.6` → stored as is; `< 0.6` → Telegram `sendAndWait` form asks the athlete, and the corrected label is stored.
7. The post-workout system prompt is fetched from this repository; the agent (tools: Sheets history, Notion profile, Pinecone knowledge base) returns a schema-validated JSON report.
8. Report → Telegram; input/output → Drive log.

### Onboarding
Telegram message → collect profile/goal data (form, `sendAndWait`) → compute Karvonen zones in a code node (age + resting HR) → save to the users sheet → agent drafts a training plan → plan is written to Notion → coach approval by e-mail → approved plan is sent to the athlete.

### RAG ingestion (manual)
Form upload (PDF/MD) → Default Data Loader → OpenAI embeddings → Pinecone insert. Retrieval uses the same embedding model (mandatory for consistent vectors).

### Weekly learning loop
Cron (Sunday 20:00) → all users → filter to those with an active plan → per user: last 7 days of trainings, totals and zone split (code node) → if no training: motivational message; otherwise read the Notion plan → agent compares plan vs actual → adherence score (0–100), recommendations, next-week adjustment → Telegram report → summary stored in a sheet.

## Design properties

* **Prompts live in git.** Agents fetch their system prompt over HTTP at run time; changing behaviour needs no workflow edit.
* **Structured output.** Each agent has a JSON Schema ([`schemas/`](../schemas)); the parser enforces the fields downstream nodes rely on.
* **Deterministic where possible.** Date arithmetic, zone maths and drift are code, not LLM output.
* **Treat the model result as given.** The post-workout prompt tells the agent to take the GRU label as certain; uncertainty is handled *before* the agent, by the confidence router.
* **No Python in n8n.** Self-hosted n8n has no Python runner, so all code nodes are JavaScript (numerically checked against the Python version of the zone calculation).

## Known gaps
See "Limitations and next steps" in the [README](../README.md).

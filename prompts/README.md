# Agent prompts

Three system prompts, one per agent. They are written in **Turkish** because the coaching conversation with the target user happens in Turkish; the matching output schemas are in [`../schemas/`](../schemas).

| File | Agent | Output schema |
|---|---|---|
| `onboarding_prompt.md` | Onboarding: turns the collected profile and goal into a training plan | `onboarding_schema.json` |
| `post_workout_prompt.md` | Post-workout coach: interprets the GRU result, history and RAG sources | `post_workout_schema.json` |
| `weekly_review_prompt.md` | Weekly review: plan-vs-actual adherence and next-week adjustment | `weekly_review_schema.json` |

The n8n workflow downloads these files from the `main` branch at run time, so a merged change takes effect without editing n8n.

Design notes: the post-workout prompt instructs the agent to treat the GRU label as certain (uncertainty is resolved earlier by confidence routing) and to interpret heart-rate drift cautiously on progression runs.

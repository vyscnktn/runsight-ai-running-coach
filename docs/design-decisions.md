# Design decisions

Short records of the choices that shaped the project, with the trade-off each one accepted.

### 1. Train on a rule-generated label (distillation)
*Why:* no ground-truth intensity labels exist for the dataset, and the rule is interpretable. *Trade-off:* metrics measure fidelity to the rule. *Gain:* the rule needs the athlete's full history; the GRU does not, which makes cold-start use possible.

### 2. Per-workout train/test split
*Why:* the number of workouts per user is very uneven, so a split by user would make the test set hinge on a few heavy users and give an unstable estimate; a per-workout split gives a stable, representative measurement of "can the model reproduce the rule on unseen workouts". Leakage is controlled where it is most dangerous — scalers are fitted on the training split only. *Trade-off:* the same athlete can appear on both sides, so performance on **new users** is unmeasured. *Follow-up:* add a user-grouped evaluation. The split decides which question you are answering; this one answers the first.

### 3. Threshold at 70 % of the personal HR range, not 85 %
See [labelling method](labeling-method.md). 0.85 mislabels sustained tempo efforts as aerobic.

### 4. Cutoff 0.02 on `anaerobic_time_frac`
A small noise margin: pure easy runs scored exactly 0 in synthetic tests. The class balance is a consequence.

### 5. No clustering
Duration separates long from easy runs trivially; the useful split is threshold-based and can be computed directly.

### 6. GRU rather than LSTM
Fewer parameters and a lighter artifact, which suits a pay-per-use serverless deployment.

### 7. Pinecone for the vector store
Persistent (the first version used n8n's in-memory store, which was wiped on every restart) and a generous free tier.

### 8. Confidence routing with a human in the loop
Below 0.6 the model's own uncertainty is surfaced to the athlete instead of being silently passed to the agent. The stored label is the human's, producing a verified dataset for later fine-tuning. The 0.6 threshold is a pragmatic setting, not a calibrated one — the model's probabilities have not been calibration-tested.

### 9. Prompts as files fetched at run time
Versioned in git, reviewable in diffs, and changeable without touching the workflow. *Trade-off:* a runtime dependency on the raw-content URL of this repository.

### 10. Determinism outside the LLM
Plan week/day, Karvonen zones and cardiac drift are code. The LLM explains and recommends; it does not compute.

### 11. One agent with tools, not a multi-agent system
The tasks (report, plan, weekly review) are separate flows with their own prompts and schemas; within a flow a single tool-using agent is enough. Sorting out *which* data to trust (confidence routing) is handled by workflow logic rather than by agents debating.

### 12. Self-hosted n8n
Moved from the n8n Cloud trial to a Hetzner VPS (Docker, reverse proxy, TLS) to keep running after the trial and to control data. Consequence: no built-in AI credits, so the OpenAI and Gemini keys are the operator's own, and Code nodes are JavaScript.

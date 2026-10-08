# Knowledge base sources

The Pinecone index (`n8n`, 1536 dimensions, cosine) was filled with the documents below. The files themselves are **not** included (copyright); fetch them from the publishers if you want to rebuild the index.

| Source | Type |
|---|---|
| Seiler, S. (2010). *What is best practice for training intensity and duration distribution in endurance athletes?* Int J Sports Physiol Perform 5(3), 276–291 | Paper |
| Smyth, Lawlor, Berndsen, Feely (2022). *Recommendations for marathon runners: on the application of recommender systems and machine learning to support recreational marathon runners.* User Model User-Adap Inter 32, 787–838. doi:10.1007/s11257-021-09299-3 | Paper |
| Andrade et al. (2020). *Is the Acute:Chronic Workload Ratio (ACWR) Associated with Risk of Time-Loss Injury in Professional Team Sports?* Sports Med 50, 1613–1635. doi:10.1007/s40279-020-01308-6 | Systematic review |
| Three short practitioner guides on setting heart-rate zones: from max heart rate, from functional threshold heart rate, and from lactate measurements | Guides |

The agent uses the index as a `knowledge_base` tool and is expected to cite the sources it used (`sources_used` in the output schema).

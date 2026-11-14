| Pipeline | Hit@1 | Evidence Hit@3 | Hit@5 | MRR@10 | Context Evidence Rate (top-3) | Avg context chars | Median latency ms | Cost USD (all queries) | Errors |
|---|---|---|---|---|---|---|---|---|---|
| dense +neighbours(top2) | 14/20 (70%) | 18/20 (90%) | 20/20 (100%) | 0.814 | 19/20 (95%) | 3737 | 920 | 0.00002 | 0 |
| rewrite_dense +neighbours(top2) | 16/20 (80%) | 19/20 (95%) | 19/20 (95%) | 0.874 | 20/20 (100%) | 3830 | 2152 | 0.00078 | 0 |
| rewrite_hybrid_rerank +neighbours(top2) | 17/20 (85%) | 19/20 (95%) | 19/20 (95%) | 0.900 | 20/20 (100%) | 3757 | 3291 | 0.00754 | 0 |

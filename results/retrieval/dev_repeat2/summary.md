| Pipeline | Hit@1 | Evidence Hit@3 | Hit@5 | MRR@10 | Context Evidence Rate (top-3) | Avg context chars | Median latency ms | Cost USD (all queries) | Errors |
|---|---|---|---|---|---|---|---|---|---|
| rewrite_bm25 | 13/20 (65%) | 17/20 (85%) | 19/20 (95%) | 0.764 | 17/20 (85%) | 2158 | 1263 | 0.00077 | 0 |
| rewrite_dense | 17/20 (85%) | 19/20 (95%) | 19/20 (95%) | 0.900 | 19/20 (95%) | 2111 | 2231 | 0.00078 | 0 |
| rewrite_hybrid | 14/20 (70%) | 19/20 (95%) | 19/20 (95%) | 0.825 | 19/20 (95%) | 2142 | 2169 | 0.00078 | 0 |
| rewrite_hybrid_rerank | 18/20 (90%) | 19/20 (95%) | 19/20 (95%) | 0.925 | 19/20 (95%) | 2131 | 3496 | 0.00754 | 0 |

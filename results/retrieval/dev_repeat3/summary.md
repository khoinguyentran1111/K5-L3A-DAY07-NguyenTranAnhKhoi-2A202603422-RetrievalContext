| Pipeline | Hit@1 | Evidence Hit@3 | Hit@5 | MRR@10 | Context Evidence Rate (top-3) | Avg context chars | Median latency ms | Cost USD (all queries) | Errors |
|---|---|---|---|---|---|---|---|---|---|
| rewrite_bm25 | 14/20 (70%) | 18/20 (90%) | 19/20 (95%) | 0.793 | 18/20 (90%) | 2176 | 1186 | 0.00077 | 0 |
| rewrite_dense | 16/20 (80%) | 19/20 (95%) | 19/20 (95%) | 0.880 | 19/20 (95%) | 2093 | 2114 | 0.00078 | 0 |
| rewrite_hybrid | 15/20 (75%) | 19/20 (95%) | 19/20 (95%) | 0.847 | 19/20 (95%) | 2111 | 2165 | 0.00075 | 0 |
| rewrite_hybrid_rerank | 16/20 (80%) | 19/20 (95%) | 19/20 (95%) | 0.867 | 19/20 (95%) | 2178 | 3450 | 0.00748 | 0 |

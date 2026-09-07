# RAG membership inference attack (Anderson et al., 2024)

- generator: `Qwen/Qwen3.5-9B`  |  device: `cuda`
- members: 1000  |  non-members: 1000  |  targets/class: 250  |  seeds: [42, 43, 44]
- target epsilons: ['inf', '0.5', '1', '2', '4', '8']  |  delta: 1e-06  |  answer length: 32 tokens  |  modes: ['unified']

AUC 0.5 = no better than chance; higher = membership leaks.

**Plot against `realized_epsilon`.** It is the total guarantee each pipeline's own accountant reports for the configuration actually run, covering retrieval and generation of an entire answer. The requested value is kept only for traceability.

**Rows with status `INFEASIBLE`** are targets the mechanism cannot deliver at any parameter setting — for Koga, budgets so tight that the Limited-Domain selection needs more agreeing voters than exist. They were not run. Had they been, they would have produced an AUC of 0.5 that looks exactly like a successful defence. The reachable epsilon range is itself a result.

**Mode `unified`** is the only one comparable across pipelines: P(the deployed system answers 'Yes'), computed exactly where the output distribution is closed-form and sampled where it is not. `upper_bound` describes an attacker reading pre-noise internals and is flat in epsilon by construction; the gap between the two is how much leakage the mechanism removes.

| pipeline | mode | variant | params | target eps | realized eps | seed | status | AUC | TPR@1% | TPR@5% | n_split | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive | unified | original | - | inf | inf | 42 | OK | 0.984 | 0.448 | 0.936 |  | 147.0 |
| grislain | unified | original | - | inf | inf | 42 | OK | 0.713 | 0.060 | 0.200 |  | 185.1 |
| grislain | unified | original | - | 0.5 | 0.500 | 42 | OK | 0.575 | 0.028 | 0.048 |  | 249.9 |
| grislain | unified | original | - | 1 | 1.000 | 42 | OK | 0.612 | 0.040 | 0.052 |  | 245.5 |
| grislain | unified | original | - | 2 | 2.002 | 42 | OK | 0.610 | 0.048 | 0.056 |  | 244.2 |
| grislain | unified | original | - | 4 | 3.998 | 42 | OK | 0.603 | 0.064 | 0.100 |  | 238.7 |
| grislain | unified | original | - | 8 | 8.000 | 42 | OK | 0.576 | 0.060 | 0.128 |  | 236.0 |
| naive | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | inf | inf | 42 | OK | 0.997 | 0.916 | 0.996 |  | 145.1 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | inf | inf | 42 | OK | 0.713 | 0.060 | 0.200 |  | 183.6 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 0.5 | 0.500 | 42 | OK | 0.585 | 0.036 | 0.064 |  | 247.1 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 1 | 1.000 | 42 | OK | 0.622 | 0.048 | 0.076 |  | 244.6 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 2 | 2.002 | 42 | OK | 0.609 | 0.040 | 0.064 |  | 242.7 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 4 | 3.998 | 42 | OK | 0.634 | 0.056 | 0.144 |  | 243.0 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 8 | 8.000 | 42 | OK | 0.592 | 0.084 | 0.128 |  | 235.3 |
| naive | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | inf | inf | 42 | OK | 0.980 | 0.548 | 0.856 |  | 156.3 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | inf | inf | 42 | OK | 0.765 | 0.152 | 0.352 |  | 238.0 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 0.5 | 0.500 | 42 | OK | 0.624 | 0.040 | 0.076 |  | 352.8 |
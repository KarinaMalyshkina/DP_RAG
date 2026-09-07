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
| naive | unified | original | - | inf | inf | 42 | OK | 0.984 | 0.448 | 0.936 |  | 137.3 |
| grislain | unified | original | - | inf | inf | 42 | OK | 0.713 | 0.036 | 0.200 |  | 185.1 |
| grislain | unified | original | - | 0.5 | 0.500 | 42 | OK | 0.587 | 0.044 | 0.068 |  | 250.7 |
| grislain | unified | original | - | 1 | 1.000 | 42 | OK | 0.617 | 0.040 | 0.064 |  | 390.5 |
| grislain | unified | original | - | 2 | 2.002 | 42 | OK | 0.619 | 0.044 | 0.068 |  | 245.6 |
| grislain | unified | original | - | 4 | 3.998 | 42 | OK | 0.624 | 0.064 | 0.092 |  | 239.8 |
| grislain | unified | original | - | 8 | 8.000 | 42 | OK | 0.604 | 0.092 | 0.128 |  | 235.0 |
| naive | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | inf | inf | 42 | OK | 0.997 | 0.916 | 0.996 |  | 135.8 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | inf | inf | 42 | OK | 0.826 | 0.296 | 0.540 |  | 178.3 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 0.5 | 0.500 | 42 | OK | 0.631 | 0.032 | 0.036 |  | 251.8 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 1 | 1.000 | 42 | OK | 0.680 | 0.072 | 0.132 |  | 249.9 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 2 | 2.002 | 42 | OK | 0.679 | 0.048 | 0.088 |  | 244.6 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 4 | 3.998 | 42 | OK | 0.673 | 0.068 | 0.100 |  | 243.4 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 8 | 8.000 | 42 | OK | 0.649 | 0.100 | 0.144 |  | 239.5 |
| naive | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | inf | inf | 42 | OK | 0.980 | 0.548 | 0.856 |  | 145.4 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | inf | inf | 42 | OK | 0.765 | 0.152 | 0.352 |  | 237.3 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 0.5 | 0.500 | 42 | OK | 0.638 | 0.044 | 0.096 |  | 347.6 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 1 | 1.000 | 42 | OK | 0.653 | 0.028 | 0.096 |  | 344.9 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 2 | 2.002 | 42 | OK | 0.626 | 0.056 | 0.092 |  | 335.5 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 4 | 3.998 | 42 | OK | 0.651 | 0.064 | 0.108 |  | 333.4 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 8 | 8.000 | 42 | OK | 0.627 | 0.060 | 0.068 |  | 325.3 |
| naive | unified | canary | n_canaries=10,n_repeats=1 | inf | inf | 42 | OK | 0.981 | 0.444 | 0.928 |  | 130.0 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | inf | inf | 42 | OK | 0.712 | 0.036 | 0.196 |  | 184.2 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 0.5 | 0.500 | 42 | OK | 0.588 | 0.040 | 0.056 |  | 248.6 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 1 | 1.000 | 42 | OK | 0.614 | 0.060 | 0.084 |  | 249.3 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 2 | 2.002 | 42 | OK | 0.616 | 0.052 | 0.088 |  | 240.3 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 4 | 3.998 | 42 | OK | 0.592 | 0.060 | 0.088 |  | 240.3 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 8 | 8.000 | 42 | OK | 0.603 | 0.064 | 0.120 |  | 234.4 |
| naive | unified | original | - | inf | inf | 43 | OK | 0.971 | 0.280 | 0.880 |  | 134.3 |
| grislain | unified | original | - | inf | inf | 43 | OK | 0.687 | 0.080 | 0.164 |  | 185.0 |
| grislain | unified | original | - | 0.5 | 0.500 | 43 | OK | 0.587 | 0.044 | 0.060 |  | 253.9 |
| grislain | unified | original | - | 1 | 1.000 | 43 | OK | 0.589 | 0.040 | 0.064 |  | 251.6 |
| grislain | unified | original | - | 2 | 2.002 | 43 | OK | 0.608 | 0.048 | 0.100 |  | 247.6 |
| grislain | unified | original | - | 4 | 3.998 | 43 | OK | 0.602 | 0.044 | 0.080 |  | 243.9 |
| grislain | unified | original | - | 8 | 8.000 | 43 | OK | 0.600 | 0.048 | 0.092 |  | 239.2 |
| naive | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | inf | inf | 43 | OK | 0.983 | 0.744 | 0.924 |  | 131.5 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | inf | inf | 43 | OK | 0.825 | 0.296 | 0.468 |  | 178.9 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 0.5 | 0.500 | 43 | OK | 0.666 | 0.068 | 0.120 |  | 255.7 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 1 | 1.000 | 43 | OK | 0.641 | 0.076 | 0.084 |  | 252.8 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 2 | 2.002 | 43 | OK | 0.636 | 0.072 | 0.108 |  | 251.4 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 4 | 3.998 | 43 | OK | 0.642 | 0.092 | 0.124 |  | 246.2 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 8 | 8.000 | 43 | OK | 0.646 | 0.100 | 0.172 |  | 242.4 |
| naive | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | inf | inf | 43 | OK | 0.966 | 0.316 | 0.820 |  | 142.0 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | inf | inf | 43 | OK | 0.759 | 0.172 | 0.268 |  | 220.0 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 0.5 | 0.500 | 43 | OK | 0.660 | 0.044 | 0.136 |  | 342.8 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 1 | 1.000 | 43 | OK | 0.609 | 0.060 | 0.092 |  | 329.7 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 2 | 2.002 | 43 | OK | 0.658 | 0.056 | 0.128 |  | 329.4 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 4 | 3.998 | 43 | OK | 0.603 | 0.088 | 0.124 |  | 319.6 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 8 | 8.000 | 43 | OK | 0.654 | 0.092 | 0.176 |  | 313.8 |
| naive | unified | canary | n_canaries=10,n_repeats=1 | inf | inf | 43 | OK | 0.974 | 0.348 | 0.872 |  | 137.2 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | inf | inf | 43 | OK | 0.684 | 0.080 | 0.164 |  | 187.0 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 0.5 | 0.500 | 43 | OK | 0.616 | 0.052 | 0.064 |  | 258.4 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 1 | 1.000 | 43 | OK | 0.588 | 0.040 | 0.084 |  | 257.0 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 2 | 2.002 | 43 | OK | 0.611 | 0.036 | 0.088 |  | 249.0 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 4 | 3.998 | 43 | OK | 0.601 | 0.052 | 0.120 |  | 248.5 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 8 | 8.000 | 43 | OK | 0.587 | 0.056 | 0.112 |  | 244.3 |
| naive | unified | original | - | inf | inf | 44 | OK | 0.959 | 0.120 | 0.732 |  | 135.2 |
| grislain | unified | original | - | inf | inf | 44 | OK | 0.670 | 0.088 | 0.188 |  | 184.3 |
| grislain | unified | original | - | 0.5 | 0.500 | 44 | OK | 0.538 | 0.024 | 0.068 |  | 258.2 |
| grislain | unified | original | - | 1 | 1.000 | 44 | OK | 0.567 | 0.036 | 0.080 |  | 255.9 |
| grislain | unified | original | - | 2 | 2.002 | 44 | OK | 0.604 | 0.036 | 0.052 |  | 247.8 |
| grislain | unified | original | - | 4 | 3.998 | 44 | OK | 0.621 | 0.044 | 0.092 |  | 244.5 |
| grislain | unified | original | - | 8 | 8.000 | 44 | OK | 0.610 | 0.052 | 0.136 |  | 242.5 |
| naive | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | inf | inf | 44 | OK | 0.987 | 0.728 | 0.900 |  | 130.6 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | inf | inf | 44 | OK | 0.815 | 0.164 | 0.488 |  | 176.2 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 0.5 | 0.500 | 44 | OK | 0.622 | 0.052 | 0.056 |  | 257.2 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 1 | 1.000 | 44 | OK | 0.651 | 0.044 | 0.112 |  | 256.9 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 2 | 2.002 | 44 | OK | 0.642 | 0.080 | 0.100 |  | 250.8 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 4 | 3.998 | 44 | OK | 0.622 | 0.104 | 0.112 |  | 246.8 |
| grislain | unified | full_duplicate | duplication_rate=0.3,n_copies=1 | 8 | 8.000 | 44 | OK | 0.645 | 0.084 | 0.116 |  | 249.3 |
| naive | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | inf | inf | 44 | OK | 0.962 | 0.252 | 0.676 |  | 137.6 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | inf | inf | 44 | OK | 0.733 | 0.176 | 0.280 |  | 224.4 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 0.5 | 0.500 | 44 | OK | 0.606 | 0.036 | 0.100 |  | 342.1 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 1 | 1.000 | 44 | OK | 0.592 | 0.048 | 0.072 |  | 334.2 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 2 | 2.002 | 44 | OK | 0.610 | 0.044 | 0.116 |  | 330.8 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 4 | 3.998 | 44 | OK | 0.600 | 0.068 | 0.108 |  | 330.6 |
| grislain | unified | partial_overlap | n_hosts=3,overlap_rate=0.3 | 8 | 8.000 | 44 | OK | 0.604 | 0.096 | 0.136 |  | 315.9 |
| naive | unified | canary | n_canaries=10,n_repeats=1 | inf | inf | 44 | OK | 0.959 | 0.124 | 0.776 |  | 128.8 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | inf | inf | 44 | OK | 0.672 | 0.088 | 0.188 |  | 187.3 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 0.5 | 0.500 | 44 | OK | 0.565 | 0.032 | 0.068 |  | 258.9 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 1 | 1.000 | 44 | OK | 0.579 | 0.040 | 0.068 |  | 256.8 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 2 | 2.002 | 44 | OK | 0.569 | 0.060 | 0.084 |  | 252.3 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 4 | 3.998 | 44 | OK | 0.570 | 0.052 | 0.100 |  | 247.6 |
| grislain | unified | canary | n_canaries=10,n_repeats=1 | 8 | 8.000 | 44 | OK | 0.575 | 0.060 | 0.084 |  | 246.4 |
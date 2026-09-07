# RAG membership inference attack (Anderson et al., 2024)

- generator: `Qwen/Qwen3.5-9B`  |  device: `cuda`
- members: 1000  |  non-members: 1000  |  targets/class: 25  |  seeds: [42]
- target epsilons: ['1', '4']  |  delta: 1e-06  |  answer length: 32 tokens  |  modes: ['unified']

AUC 0.5 = no better than chance; higher = membership leaks.

**Plot against `realized_epsilon`.** It is the total guarantee each pipeline's own accountant reports for the configuration actually run, covering retrieval and generation of an entire answer. The requested value is kept only for traceability.

**Rows with status `INFEASIBLE`** are targets the mechanism cannot deliver at any parameter setting — for Koga, budgets so tight that the Limited-Domain selection needs more agreeing voters than exist. They were not run. Had they been, they would have produced an AUC of 0.5 that looks exactly like a successful defence. The reachable epsilon range is itself a result.

**Mode `unified`** is the only one comparable across pipelines: P(the deployed system answers 'Yes'), computed exactly where the output distribution is closed-form and sampled where it is not. `upper_bound` describes an attacker reading pre-noise internals and is flat in epsilon by construction; the gap between the two is how much leakage the mechanism removes.

| pipeline | mode | variant | params | target eps | realized eps | seed | status | AUC | TPR@1% | TPR@5% | n_split | secs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| grislain | unified | original | - | 1 | 1.000 | 42 | OK | 0.621 | 0.080 | 0.080 |  | 23.6 |
| koga | unified | original | - | 1 | 1.000 | 42 | INFEASIBLE | nan | nan | nan | 128 | 0.0 |
| grislain | unified | original | - | 4 | 3.998 | 42 | OK | 0.682 | 0.200 | 0.200 |  | 22.7 |
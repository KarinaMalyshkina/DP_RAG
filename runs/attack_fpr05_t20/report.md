# RAG membership inference attack (Anderson et al., 2024)

- mode: `gray_box`
- generator: `Qwen/Qwen2.5-0.5B-Instruct`  |  embedder: `sentence-transformers/all-MiniLM-L6-v2`  |  device: `cpu`
- members indexed: 40  |  non-members held out: 40  |  targets/class: 20  |  seed: 42

AUC 0.5 = no better than chance; higher = membership leaks. Each row = one (pipeline, variant).

**Caveat (gray_box mode):** `grislain` applies `DPLogitsAggregator` before scoring — the score reflects the actual DP output distribution. `koga` averages raw voter logits before DP-Gumbel noise — the score is an adversarial upper bound, not the DP-protected output. `naive` has no DP and serves as the epsilon=inf baseline.

| pipeline | variant | status | AUC | TPR@FPR=0.01 | TPR@FPR=0.05 | accuracy | TPR | FPR | secs |
|---|---|---|---|---|---|---|---|---|---|
| naive | original | OK | 0.586 | 0.050 | 0.200 | 0.525 | 0.950 | 0.900 | 1662.7 |
| grislain | original | OK | 0.497 | 0.100 | 0.150 | 0.475 | 0.800 | 0.850 | 4026.2 |
| naive | full_duplicate | OK | 0.594 | 0.050 | 0.200 | 0.525 | 1.000 | 0.950 | 1266.6 |
| grislain | full_duplicate | OK | 0.450 | 0.050 | 0.100 | 0.525 | 0.850 | 0.800 | 3515.1 |

Note: smoke-scale (tiny CPU model, few targets) — AUC is noisy. The intended reading is the *trend*: members should be easier to detect than non-members, and duplication (`full_duplicate`) should raise AUC relative to `original`.
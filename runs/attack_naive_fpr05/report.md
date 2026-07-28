# RAG membership inference attack (Anderson et al., 2024)

- mode: `gray_box`
- generator: `Qwen/Qwen2.5-0.5B-Instruct`  |  embedder: `sentence-transformers/all-MiniLM-L6-v2`  |  device: `cpu`
- members indexed: 40  |  non-members held out: 40  |  targets/class: 10  |  seed: 42

AUC 0.5 = no better than chance; higher = membership leaks. Each row = one (pipeline, variant).

**Caveat (gray_box mode):** `grislain` applies `DPLogitsAggregator` before scoring — the score reflects the actual DP output distribution. `koga` averages raw voter logits before DP-Gumbel noise — the score is an adversarial upper bound, not the DP-protected output. `naive` has no DP and serves as the epsilon=inf baseline.

| pipeline | variant | status | AUC | TPR@FPR=0.01 | TPR@FPR=0.05 | accuracy | TPR | FPR | secs |
|---|---|---|---|---|---|---|---|---|---|
| naive | original | OK | 0.710 | 0.200 | 0.200 | 0.550 | 0.900 | 0.800 | 611.4 |
| naive | full_duplicate | OK | 0.755 | 0.300 | 0.300 | 0.600 | 1.000 | 0.800 | 857.5 |
| naive | partial_overlap | OK | 0.645 | 0.000 | 0.000 | 0.550 | 0.900 | 0.800 | 1202.9 |
| naive | canary | OK | 0.675 | 0.100 | 0.100 | 0.600 | 0.900 | 0.700 | 1008.5 |

Note: smoke-scale (tiny CPU model, few targets) — AUC is noisy. The intended reading is the *trend*: members should be easier to detect than non-members, and duplication (`full_duplicate`) should raise AUC relative to `original`.
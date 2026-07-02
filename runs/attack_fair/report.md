# RAG membership inference attack (Anderson et al., 2024)

- mode: `gray_box`
- generator: `Qwen/Qwen2.5-0.5B-Instruct`  |  embedder: `sentence-transformers/all-MiniLM-L6-v2`  |  device: `cpu`
- members indexed: 60  |  non-members held out: 60  |  targets/class: 5  |  seed: 42

AUC 0.5 = no better than chance; higher = membership leaks. Each row = one (pipeline, variant).

**Caveat (gray_box mode):** `grislain` applies `DPLogitsAggregator` before scoring — the score reflects the actual DP output distribution. `koga` averages raw voter logits before DP-Gumbel noise — the score is an adversarial upper bound, not the DP-protected output. `naive` has no DP and serves as the epsilon=inf baseline.

| pipeline | variant | status | AUC | TPR@FPR=0.01 | accuracy | TPR | FPR | secs |
|---|---|---|---|---|---|---|---|---|
| naive | original | OK | 0.720 | 0.200 | 0.500 | 1.000 | 1.000 | 242.9 |
| koga | original | OK | 0.520 | 0.200 | 0.500 | 1.000 | 1.000 | 717.7 |
| grislain | original | OK | 0.840 | 0.600 | 0.700 | 0.800 | 0.400 | 673.0 |
| naive | full_duplicate | OK | 0.620 | 0.200 | 0.500 | 1.000 | 1.000 | 254.2 |
| koga | full_duplicate | OK | 0.520 | 0.200 | 0.500 | 1.000 | 1.000 | 714.2 |
| grislain | full_duplicate | OK | 0.600 | 0.200 | 0.600 | 1.000 | 0.800 | 706.4 |

Note: smoke-scale (tiny CPU model, few targets) — AUC is noisy. The intended reading is the *trend*: members should be easier to detect than non-members, and duplication (`full_duplicate`) should raise AUC relative to `original`.
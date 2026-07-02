# RAG membership inference attack (Anderson et al., 2024)

- mode: `gray_box`
- generator: `Qwen/Qwen2.5-0.5B-Instruct`  |  embedder: `sentence-transformers/all-MiniLM-L6-v2`  |  device: `cpu`
- members indexed: 40  |  non-members held out: 40  |  targets/class: 10  |  seed: 42

AUC 0.5 = no better than chance; higher = membership leaks. Each row = one (pipeline, variant).

| pipeline | variant | status | AUC | TPR@FPR=0.01 | accuracy | TPR | FPR | secs |
|---|---|---|---|---|---|---|---|---|
| naive | original | OK | 0.710 | 0.200 | 0.550 | 0.900 | 0.800 | 527.2 |
| naive | full_duplicate | OK | 0.755 | 0.300 | 0.600 | 1.000 | 0.800 | 548.0 |
| naive | partial_overlap | OK | 0.645 | 0.000 | 0.550 | 0.900 | 0.800 | 775.6 |
| naive | canary | OK | 0.675 | 0.100 | 0.600 | 0.900 | 0.700 | 591.6 |

Note: smoke-scale (tiny CPU model, few targets) — AUC is noisy. The intended reading is the *trend*: members should be easier to detect than non-members, and duplication (`full_duplicate`) should raise AUC relative to `original`.
# RAG membership inference attack (Anderson et al., 2024)

- mode: `gray_box`
- generator: `Qwen/Qwen2.5-0.5B-Instruct`  |  embedder: `sentence-transformers/all-MiniLM-L6-v2`  |  device: `cpu`
- members indexed: 30  |  non-members held out: 30  |  targets/class: 8  |  seed: 42

AUC 0.5 = no better than chance; higher = membership leaks. Each row = one (pipeline, variant).

| pipeline | variant | status | AUC | accuracy | TPR | FPR | secs |
|---|---|---|---|---|---|---|---|
| koga | original | OK | 0.781 | 0.562 | 0.250 | 0.125 | 516.4 |
| grislain | original | OK | 0.734 | 0.625 | 0.625 | 0.375 | 1244.5 |
| koga | full_duplicate | OK | 0.953 | 0.750 | 0.625 | 0.125 | 700.4 |
| grislain | full_duplicate | OK | 0.781 | 0.688 | 0.750 | 0.375 | 1140.6 |
| koga | partial_overlap | OK | 0.875 | 0.688 | 0.375 | 0.000 | 719.2 |
| grislain | partial_overlap | OK | 0.828 | 0.688 | 0.750 | 0.375 | 1537.6 |
| koga | canary | OK | 0.844 | 0.562 | 0.125 | 0.000 | 589.4 |
| grislain | canary | OK | 0.719 | 0.625 | 0.750 | 0.500 | 1541.6 |

Note: smoke-scale (tiny CPU model, few targets) — AUC is noisy. The intended reading is the *trend*: members should be easier to detect than non-members, and duplication (`full_duplicate`) should raise AUC relative to `original`.
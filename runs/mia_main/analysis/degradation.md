# Duplication degradation vs. the zero-duplication control

- sources: runs/mia_main
- metrics: auc, tpr_at_fpr_01, tpr_at_fpr_05  |  control variant: `original`
- rows read: 84  |  paired cells: 189

Each duplicated cell is compared with the `original` cell at the **same** pipeline, scoring mode, epsilon target and seed. `delta = value - control`; `ratio = value / control`. For attack metrics a positive delta means duplication increased leakage — the gap between the formal per-document guarantee and the protection actually delivered.

**Compare pipelines only within the `unified` mode**, and only at epsilon values both mechanisms can reach. Rows the mechanism could not deliver were recorded `INFEASIBLE` by the runner and are excluded here rather than being counted as zero leakage.

## Per-seed comparisons

| metric | pipeline | mode | eps | seed | variant | params | axis | strength | canary repeats | value | control | delta | ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auc | grislain | unified | 0.5 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6314 | 0.5872 | 0.0442 | 1.075 |
| auc | grislain | unified | 0.5 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6378 | 0.5872 | 0.0506 | 1.086 |
| auc | grislain | unified | 0.5 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.5884 | 0.5872 | 0.0012 | 1.002 |
| auc | grislain | unified | 0.5 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6664 | 0.5873 | 0.0791 | 1.135 |
| auc | grislain | unified | 0.5 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6604 | 0.5873 | 0.0731 | 1.125 |
| auc | grislain | unified | 0.5 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.6165 | 0.5873 | 0.0292 | 1.050 |
| auc | grislain | unified | 0.5 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6224 | 0.5380 | 0.0844 | 1.157 |
| auc | grislain | unified | 0.5 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6058 | 0.5380 | 0.0678 | 1.126 |
| auc | grislain | unified | 0.5 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.5653 | 0.5380 | 0.0273 | 1.051 |
| auc | grislain | unified | 1 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6796 | 0.6167 | 0.0629 | 1.102 |
| auc | grislain | unified | 1 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6525 | 0.6167 | 0.0358 | 1.058 |
| auc | grislain | unified | 1 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.6144 | 0.6167 | -0.0023 | 0.996 |
| auc | grislain | unified | 1 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6414 | 0.5893 | 0.0520 | 1.088 |
| auc | grislain | unified | 1 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6088 | 0.5893 | 0.0194 | 1.033 |
| auc | grislain | unified | 1 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.5877 | 0.5893 | -0.0017 | 0.997 |
| auc | grislain | unified | 1 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6508 | 0.5673 | 0.0835 | 1.147 |
| auc | grislain | unified | 1 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.5923 | 0.5673 | 0.0250 | 1.044 |
| auc | grislain | unified | 1 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.5786 | 0.5673 | 0.0113 | 1.020 |
| auc | grislain | unified | 2 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6794 | 0.6188 | 0.0606 | 1.098 |
| auc | grislain | unified | 2 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6259 | 0.6188 | 0.0071 | 1.011 |
| auc | grislain | unified | 2 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.6160 | 0.6188 | -0.0028 | 0.995 |
| auc | grislain | unified | 2 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6358 | 0.6083 | 0.0275 | 1.045 |
| auc | grislain | unified | 2 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6577 | 0.6083 | 0.0494 | 1.081 |
| auc | grislain | unified | 2 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.6111 | 0.6083 | 0.0028 | 1.005 |
| auc | grislain | unified | 2 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6422 | 0.6040 | 0.0383 | 1.063 |
| auc | grislain | unified | 2 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6104 | 0.6040 | 0.0065 | 1.011 |
| auc | grislain | unified | 2 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.5689 | 0.6040 | -0.0351 | 0.942 |
| auc | grislain | unified | 4 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6727 | 0.6239 | 0.0487 | 1.078 |
| auc | grislain | unified | 4 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6511 | 0.6239 | 0.0272 | 1.044 |
| auc | grislain | unified | 4 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.5925 | 0.6239 | -0.0315 | 0.950 |
| auc | grislain | unified | 4 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6417 | 0.6023 | 0.0394 | 1.065 |
| auc | grislain | unified | 4 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6031 | 0.6023 | 0.0008 | 1.001 |
| auc | grislain | unified | 4 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.6008 | 0.6023 | -0.0014 | 0.998 |
| auc | grislain | unified | 4 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6223 | 0.6215 | 0.0008 | 1.001 |
| auc | grislain | unified | 4 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6004 | 0.6215 | -0.0211 | 0.966 |
| auc | grislain | unified | 4 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.5700 | 0.6215 | -0.0515 | 0.917 |
| auc | grislain | unified | 8 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6494 | 0.6043 | 0.0451 | 1.075 |
| auc | grislain | unified | 8 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6271 | 0.6043 | 0.0228 | 1.038 |
| auc | grislain | unified | 8 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.6033 | 0.6043 | -0.0010 | 0.998 |
| auc | grislain | unified | 8 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6461 | 0.6003 | 0.0458 | 1.076 |
| auc | grislain | unified | 8 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6537 | 0.6003 | 0.0534 | 1.089 |
| auc | grislain | unified | 8 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.5869 | 0.6003 | -0.0134 | 0.978 |
| auc | grislain | unified | 8 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.6448 | 0.6100 | 0.0348 | 1.057 |
| auc | grislain | unified | 8 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6041 | 0.6100 | -0.0059 | 0.990 |
| auc | grislain | unified | 8 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.5753 | 0.6100 | -0.0347 | 0.943 |
| auc | grislain | unified | inf | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.8259 | 0.7130 | 0.1129 | 1.158 |
| auc | grislain | unified | inf | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.7648 | 0.7130 | 0.0518 | 1.073 |
| auc | grislain | unified | inf | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.7124 | 0.7130 | -0.0006 | 0.999 |
| auc | grislain | unified | inf | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.8251 | 0.6869 | 0.1382 | 1.201 |
| auc | grislain | unified | inf | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.7593 | 0.6869 | 0.0724 | 1.105 |
| auc | grislain | unified | inf | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.6839 | 0.6869 | -0.0030 | 0.996 |
| auc | grislain | unified | inf | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.8150 | 0.6697 | 0.1452 | 1.217 |
| auc | grislain | unified | inf | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.7328 | 0.6697 | 0.0631 | 1.094 |
| auc | grislain | unified | inf | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.6724 | 0.6697 | 0.0026 | 1.004 |
| auc | naive | unified | inf | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.9972 | 0.9838 | 0.0134 | 1.014 |
| auc | naive | unified | inf | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.9805 | 0.9838 | -0.0033 | 0.997 |
| auc | naive | unified | inf | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.9811 | 0.9838 | -0.0026 | 0.997 |
| auc | naive | unified | inf | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.9835 | 0.9708 | 0.0127 | 1.013 |
| auc | naive | unified | inf | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.9659 | 0.9708 | -0.0048 | 0.995 |
| auc | naive | unified | inf | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.9744 | 0.9708 | 0.0036 | 1.004 |
| auc | naive | unified | inf | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.9867 | 0.9589 | 0.0278 | 1.029 |
| auc | naive | unified | inf | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.9620 | 0.9589 | 0.0031 | 1.003 |
| auc | naive | unified | inf | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.9591 | 0.9589 | 0.0002 | 1.000 |
| tpr_at_fpr_01 | grislain | unified | 0.5 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0320 | 0.0440 | -0.0120 | 0.727 |
| tpr_at_fpr_01 | grislain | unified | 0.5 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0440 | 0.0440 | 0.0000 | 1.000 |
| tpr_at_fpr_01 | grislain | unified | 0.5 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0400 | 0.0440 | -0.0040 | 0.909 |
| tpr_at_fpr_01 | grislain | unified | 0.5 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0680 | 0.0440 | 0.0240 | 1.545 |
| tpr_at_fpr_01 | grislain | unified | 0.5 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0440 | 0.0440 | 0.0000 | 1.000 |
| tpr_at_fpr_01 | grislain | unified | 0.5 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0520 | 0.0440 | 0.0080 | 1.182 |
| tpr_at_fpr_01 | grislain | unified | 0.5 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0520 | 0.0240 | 0.0280 | 2.167 |
| tpr_at_fpr_01 | grislain | unified | 0.5 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0360 | 0.0240 | 0.0120 | 1.500 |
| tpr_at_fpr_01 | grislain | unified | 0.5 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0320 | 0.0240 | 0.0080 | 1.333 |
| tpr_at_fpr_01 | grislain | unified | 1 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0720 | 0.0400 | 0.0320 | 1.800 |
| tpr_at_fpr_01 | grislain | unified | 1 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0280 | 0.0400 | -0.0120 | 0.700 |
| tpr_at_fpr_01 | grislain | unified | 1 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0600 | 0.0400 | 0.0200 | 1.500 |
| tpr_at_fpr_01 | grislain | unified | 1 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0760 | 0.0400 | 0.0360 | 1.900 |
| tpr_at_fpr_01 | grislain | unified | 1 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0600 | 0.0400 | 0.0200 | 1.500 |
| tpr_at_fpr_01 | grislain | unified | 1 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0400 | 0.0400 | 0.0000 | 1.000 |
| tpr_at_fpr_01 | grislain | unified | 1 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0440 | 0.0360 | 0.0080 | 1.222 |
| tpr_at_fpr_01 | grislain | unified | 1 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0480 | 0.0360 | 0.0120 | 1.333 |
| tpr_at_fpr_01 | grislain | unified | 1 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0400 | 0.0360 | 0.0040 | 1.111 |
| tpr_at_fpr_01 | grislain | unified | 2 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0480 | 0.0440 | 0.0040 | 1.091 |
| tpr_at_fpr_01 | grislain | unified | 2 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0560 | 0.0440 | 0.0120 | 1.273 |
| tpr_at_fpr_01 | grislain | unified | 2 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0520 | 0.0440 | 0.0080 | 1.182 |
| tpr_at_fpr_01 | grislain | unified | 2 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0720 | 0.0480 | 0.0240 | 1.500 |
| tpr_at_fpr_01 | grislain | unified | 2 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0560 | 0.0480 | 0.0080 | 1.167 |
| tpr_at_fpr_01 | grislain | unified | 2 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0360 | 0.0480 | -0.0120 | 0.750 |
| tpr_at_fpr_01 | grislain | unified | 2 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0800 | 0.0360 | 0.0440 | 2.222 |
| tpr_at_fpr_01 | grislain | unified | 2 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0440 | 0.0360 | 0.0080 | 1.222 |
| tpr_at_fpr_01 | grislain | unified | 2 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0600 | 0.0360 | 0.0240 | 1.667 |
| tpr_at_fpr_01 | grislain | unified | 4 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0680 | 0.0640 | 0.0040 | 1.062 |
| tpr_at_fpr_01 | grislain | unified | 4 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0640 | 0.0640 | 0.0000 | 1.000 |
| tpr_at_fpr_01 | grislain | unified | 4 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0600 | 0.0640 | -0.0040 | 0.938 |
| tpr_at_fpr_01 | grislain | unified | 4 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0920 | 0.0440 | 0.0480 | 2.091 |
| tpr_at_fpr_01 | grislain | unified | 4 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0880 | 0.0440 | 0.0440 | 2.000 |
| tpr_at_fpr_01 | grislain | unified | 4 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0520 | 0.0440 | 0.0080 | 1.182 |
| tpr_at_fpr_01 | grislain | unified | 4 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1040 | 0.0440 | 0.0600 | 2.364 |
| tpr_at_fpr_01 | grislain | unified | 4 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0680 | 0.0440 | 0.0240 | 1.545 |
| tpr_at_fpr_01 | grislain | unified | 4 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0520 | 0.0440 | 0.0080 | 1.182 |
| tpr_at_fpr_01 | grislain | unified | 8 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1000 | 0.0920 | 0.0080 | 1.087 |
| tpr_at_fpr_01 | grislain | unified | 8 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0600 | 0.0920 | -0.0320 | 0.652 |
| tpr_at_fpr_01 | grislain | unified | 8 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0640 | 0.0920 | -0.0280 | 0.696 |
| tpr_at_fpr_01 | grislain | unified | 8 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1000 | 0.0480 | 0.0520 | 2.083 |
| tpr_at_fpr_01 | grislain | unified | 8 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0920 | 0.0480 | 0.0440 | 1.917 |
| tpr_at_fpr_01 | grislain | unified | 8 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0560 | 0.0480 | 0.0080 | 1.167 |
| tpr_at_fpr_01 | grislain | unified | 8 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0840 | 0.0520 | 0.0320 | 1.615 |
| tpr_at_fpr_01 | grislain | unified | 8 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0960 | 0.0520 | 0.0440 | 1.846 |
| tpr_at_fpr_01 | grislain | unified | 8 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0600 | 0.0520 | 0.0080 | 1.154 |
| tpr_at_fpr_01 | grislain | unified | inf | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.2960 | 0.0360 | 0.2600 | 8.222 |
| tpr_at_fpr_01 | grislain | unified | inf | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1520 | 0.0360 | 0.1160 | 4.222 |
| tpr_at_fpr_01 | grislain | unified | inf | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0360 | 0.0360 | 0.0000 | 1.000 |
| tpr_at_fpr_01 | grislain | unified | inf | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.2960 | 0.0800 | 0.2160 | 3.700 |
| tpr_at_fpr_01 | grislain | unified | inf | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1720 | 0.0800 | 0.0920 | 2.150 |
| tpr_at_fpr_01 | grislain | unified | inf | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0800 | 0.0800 | 0.0000 | 1.000 |
| tpr_at_fpr_01 | grislain | unified | inf | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1640 | 0.0880 | 0.0760 | 1.864 |
| tpr_at_fpr_01 | grislain | unified | inf | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1760 | 0.0880 | 0.0880 | 2.000 |
| tpr_at_fpr_01 | grislain | unified | inf | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0880 | 0.0880 | 0.0000 | 1.000 |
| tpr_at_fpr_01 | naive | unified | inf | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.9160 | 0.4480 | 0.4680 | 2.045 |
| tpr_at_fpr_01 | naive | unified | inf | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.5480 | 0.4480 | 0.1000 | 1.223 |
| tpr_at_fpr_01 | naive | unified | inf | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.4440 | 0.4480 | -0.0040 | 0.991 |
| tpr_at_fpr_01 | naive | unified | inf | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.7440 | 0.2800 | 0.4640 | 2.657 |
| tpr_at_fpr_01 | naive | unified | inf | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.3160 | 0.2800 | 0.0360 | 1.129 |
| tpr_at_fpr_01 | naive | unified | inf | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.3480 | 0.2800 | 0.0680 | 1.243 |
| tpr_at_fpr_01 | naive | unified | inf | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.7280 | 0.1200 | 0.6080 | 6.067 |
| tpr_at_fpr_01 | naive | unified | inf | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.2520 | 0.1200 | 0.1320 | 2.100 |
| tpr_at_fpr_01 | naive | unified | inf | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.1240 | 0.1200 | 0.0040 | 1.033 |
| tpr_at_fpr_05 | grislain | unified | 0.5 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0360 | 0.0680 | -0.0320 | 0.529 |
| tpr_at_fpr_05 | grislain | unified | 0.5 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0960 | 0.0680 | 0.0280 | 1.412 |
| tpr_at_fpr_05 | grislain | unified | 0.5 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0560 | 0.0680 | -0.0120 | 0.824 |
| tpr_at_fpr_05 | grislain | unified | 0.5 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1200 | 0.0600 | 0.0600 | 2.000 |
| tpr_at_fpr_05 | grislain | unified | 0.5 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1360 | 0.0600 | 0.0760 | 2.267 |
| tpr_at_fpr_05 | grislain | unified | 0.5 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0640 | 0.0600 | 0.0040 | 1.067 |
| tpr_at_fpr_05 | grislain | unified | 0.5 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0560 | 0.0680 | -0.0120 | 0.824 |
| tpr_at_fpr_05 | grislain | unified | 0.5 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1000 | 0.0680 | 0.0320 | 1.471 |
| tpr_at_fpr_05 | grislain | unified | 0.5 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0680 | 0.0680 | 0.0000 | 1.000 |
| tpr_at_fpr_05 | grislain | unified | 1 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1320 | 0.0640 | 0.0680 | 2.062 |
| tpr_at_fpr_05 | grislain | unified | 1 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0960 | 0.0640 | 0.0320 | 1.500 |
| tpr_at_fpr_05 | grislain | unified | 1 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0840 | 0.0640 | 0.0200 | 1.312 |
| tpr_at_fpr_05 | grislain | unified | 1 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0840 | 0.0640 | 0.0200 | 1.312 |
| tpr_at_fpr_05 | grislain | unified | 1 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0920 | 0.0640 | 0.0280 | 1.438 |
| tpr_at_fpr_05 | grislain | unified | 1 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0840 | 0.0640 | 0.0200 | 1.312 |
| tpr_at_fpr_05 | grislain | unified | 1 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1120 | 0.0800 | 0.0320 | 1.400 |
| tpr_at_fpr_05 | grislain | unified | 1 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0720 | 0.0800 | -0.0080 | 0.900 |
| tpr_at_fpr_05 | grislain | unified | 1 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0680 | 0.0800 | -0.0120 | 0.850 |
| tpr_at_fpr_05 | grislain | unified | 2 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.0880 | 0.0680 | 0.0200 | 1.294 |
| tpr_at_fpr_05 | grislain | unified | 2 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0920 | 0.0680 | 0.0240 | 1.353 |
| tpr_at_fpr_05 | grislain | unified | 2 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0880 | 0.0680 | 0.0200 | 1.294 |
| tpr_at_fpr_05 | grislain | unified | 2 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1080 | 0.1000 | 0.0080 | 1.080 |
| tpr_at_fpr_05 | grislain | unified | 2 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1280 | 0.1000 | 0.0280 | 1.280 |
| tpr_at_fpr_05 | grislain | unified | 2 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0880 | 0.1000 | -0.0120 | 0.880 |
| tpr_at_fpr_05 | grislain | unified | 2 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1000 | 0.0520 | 0.0480 | 1.923 |
| tpr_at_fpr_05 | grislain | unified | 2 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1160 | 0.0520 | 0.0640 | 2.231 |
| tpr_at_fpr_05 | grislain | unified | 2 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0840 | 0.0520 | 0.0320 | 1.615 |
| tpr_at_fpr_05 | grislain | unified | 4 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1000 | 0.0920 | 0.0080 | 1.087 |
| tpr_at_fpr_05 | grislain | unified | 4 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1080 | 0.0920 | 0.0160 | 1.174 |
| tpr_at_fpr_05 | grislain | unified | 4 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0880 | 0.0920 | -0.0040 | 0.957 |
| tpr_at_fpr_05 | grislain | unified | 4 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1240 | 0.0800 | 0.0440 | 1.550 |
| tpr_at_fpr_05 | grislain | unified | 4 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1240 | 0.0800 | 0.0440 | 1.550 |
| tpr_at_fpr_05 | grislain | unified | 4 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.1200 | 0.0800 | 0.0400 | 1.500 |
| tpr_at_fpr_05 | grislain | unified | 4 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1120 | 0.0920 | 0.0200 | 1.217 |
| tpr_at_fpr_05 | grislain | unified | 4 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1080 | 0.0920 | 0.0160 | 1.174 |
| tpr_at_fpr_05 | grislain | unified | 4 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.1000 | 0.0920 | 0.0080 | 1.087 |
| tpr_at_fpr_05 | grislain | unified | 8 | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1440 | 0.1280 | 0.0160 | 1.125 |
| tpr_at_fpr_05 | grislain | unified | 8 | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.0680 | 0.1280 | -0.0600 | 0.531 |
| tpr_at_fpr_05 | grislain | unified | 8 | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.1200 | 0.1280 | -0.0080 | 0.938 |
| tpr_at_fpr_05 | grislain | unified | 8 | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1720 | 0.0920 | 0.0800 | 1.870 |
| tpr_at_fpr_05 | grislain | unified | 8 | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1760 | 0.0920 | 0.0840 | 1.913 |
| tpr_at_fpr_05 | grislain | unified | 8 | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.1120 | 0.0920 | 0.0200 | 1.217 |
| tpr_at_fpr_05 | grislain | unified | 8 | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.1160 | 0.1360 | -0.0200 | 0.853 |
| tpr_at_fpr_05 | grislain | unified | 8 | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.1360 | 0.1360 | 0.0000 | 1.000 |
| tpr_at_fpr_05 | grislain | unified | 8 | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.0840 | 0.1360 | -0.0520 | 0.618 |
| tpr_at_fpr_05 | grislain | unified | inf | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.5400 | 0.2000 | 0.3400 | 2.700 |
| tpr_at_fpr_05 | grislain | unified | inf | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.3520 | 0.2000 | 0.1520 | 1.760 |
| tpr_at_fpr_05 | grislain | unified | inf | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.1960 | 0.2000 | -0.0040 | 0.980 |
| tpr_at_fpr_05 | grislain | unified | inf | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.4680 | 0.1640 | 0.3040 | 2.854 |
| tpr_at_fpr_05 | grislain | unified | inf | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.2680 | 0.1640 | 0.1040 | 1.634 |
| tpr_at_fpr_05 | grislain | unified | inf | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.1640 | 0.1640 | 0.0000 | 1.000 |
| tpr_at_fpr_05 | grislain | unified | inf | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.4880 | 0.1880 | 0.3000 | 2.596 |
| tpr_at_fpr_05 | grislain | unified | inf | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.2800 | 0.1880 | 0.0920 | 1.489 |
| tpr_at_fpr_05 | grislain | unified | inf | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.1880 | 0.1880 | 0.0000 | 1.000 |
| tpr_at_fpr_05 | naive | unified | inf | 42 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.9960 | 0.9360 | 0.0600 | 1.064 |
| tpr_at_fpr_05 | naive | unified | inf | 42 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.8560 | 0.9360 | -0.0800 | 0.915 |
| tpr_at_fpr_05 | naive | unified | inf | 42 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.9280 | 0.9360 | -0.0080 | 0.991 |
| tpr_at_fpr_05 | naive | unified | inf | 43 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.9240 | 0.8800 | 0.0440 | 1.050 |
| tpr_at_fpr_05 | naive | unified | inf | 43 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.8200 | 0.8800 | -0.0600 | 0.932 |
| tpr_at_fpr_05 | naive | unified | inf | 43 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.8720 | 0.8800 | -0.0080 | 0.991 |
| tpr_at_fpr_05 | naive | unified | inf | 44 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 0.9000 | 0.7320 | 0.1680 | 1.230 |
| tpr_at_fpr_05 | naive | unified | inf | 44 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 0.6760 | 0.7320 | -0.0560 | 0.923 |
| tpr_at_fpr_05 | naive | unified | inf | 44 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 0.7760 | 0.7320 | 0.0440 | 1.060 |

## Pooled across seeds

Mean, sample standard deviation and a 95% Student-t confidence interval over the seed axis. Seeds redraw the corpus split as well as the mechanism's noise, so this is the spread over independent replicates of the whole experiment. A `n_seeds` of 1 leaves the spread columns `nan` — one replicate has a value but no interval.

| metric | quantity | pipeline | mode | eps | variant | params | axis | strength | canary repeats | n | mean | std | sem | CI low | CI high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| auc | value | grislain | unified | 0.5 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.5900 | 0.0256 | 0.0148 | 0.5264 | 0.6537 |
| auc | delta | grislain | unified | 0.5 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0192 | 0.0157 | 0.0090 | -0.0197 | 0.0581 |
| auc | ratio | grislain | unified | 0.5 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.0342 | 0.0279 | 0.0161 | 0.9649 | 1.1034 |
| auc | value | grislain | unified | 0.5 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.6401 | 0.0233 | 0.0134 | 0.5823 | 0.6979 |
| auc | delta | grislain | unified | 0.5 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0693 | 0.0219 | 0.0126 | 0.0150 | 0.1235 |
| auc | ratio | grislain | unified | 0.5 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.1223 | 0.0422 | 0.0244 | 1.0174 | 1.2272 |
| auc | value | grislain | unified | 0.5 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.6347 | 0.0275 | 0.0159 | 0.5665 | 0.7029 |
| auc | delta | grislain | unified | 0.5 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0639 | 0.0118 | 0.0068 | 0.0346 | 0.0931 |
| auc | ratio | grislain | unified | 0.5 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.1123 | 0.0226 | 0.0130 | 1.0562 | 1.1683 |
| auc | value | grislain | unified | 1 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.5936 | 0.0186 | 0.0108 | 0.5473 | 0.6398 |
| auc | delta | grislain | unified | 1 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0024 | 0.0077 | 0.0044 | -0.0166 | 0.0215 |
| auc | ratio | grislain | unified | 1 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.0044 | 0.0134 | 0.0077 | 0.9712 | 1.0376 |
| auc | value | grislain | unified | 1 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.6573 | 0.0199 | 0.0115 | 0.6078 | 0.7067 |
| auc | delta | grislain | unified | 1 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0661 | 0.0160 | 0.0092 | 0.0265 | 0.1058 |
| auc | ratio | grislain | unified | 1 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.1125 | 0.0308 | 0.0178 | 1.0360 | 1.1890 |
| auc | value | grislain | unified | 1 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.6179 | 0.0311 | 0.0180 | 0.5406 | 0.6951 |
| auc | delta | grislain | unified | 1 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0268 | 0.0083 | 0.0048 | 0.0061 | 0.0474 |
| auc | ratio | grislain | unified | 1 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.0451 | 0.0126 | 0.0073 | 1.0138 | 1.0763 |
| auc | value | grislain | unified | 2 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.5986 | 0.0259 | 0.0149 | 0.5344 | 0.6629 |
| auc | delta | grislain | unified | 2 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | -0.0117 | 0.0204 | 0.0118 | -0.0624 | 0.0390 |
| auc | ratio | grislain | unified | 2 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.9806 | 0.0338 | 0.0195 | 0.8966 | 1.0647 |
| auc | value | grislain | unified | 2 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.6525 | 0.0235 | 0.0136 | 0.5941 | 0.7109 |
| auc | delta | grislain | unified | 2 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0421 | 0.0169 | 0.0097 | 0.0002 | 0.0840 |
| auc | ratio | grislain | unified | 2 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.0688 | 0.0267 | 0.0154 | 1.0024 | 1.1353 |
| auc | value | grislain | unified | 2 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.6314 | 0.0241 | 0.0139 | 0.5715 | 0.6912 |
| auc | delta | grislain | unified | 2 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0210 | 0.0246 | 0.0142 | -0.0401 | 0.0821 |
| auc | ratio | grislain | unified | 2 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.0345 | 0.0405 | 0.0234 | 0.9339 | 1.1350 |
| auc | value | grislain | unified | 4 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.5878 | 0.0160 | 0.0092 | 0.5481 | 0.6274 |
| auc | delta | grislain | unified | 4 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | -0.0281 | 0.0252 | 0.0145 | -0.0907 | 0.0345 |
| auc | ratio | grislain | unified | 4 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.9548 | 0.0405 | 0.0234 | 0.8542 | 1.0554 |
| auc | value | grislain | unified | 4 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.6455 | 0.0254 | 0.0147 | 0.5824 | 0.7087 |
| auc | delta | grislain | unified | 4 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0296 | 0.0254 | 0.0147 | -0.0335 | 0.0928 |
| auc | ratio | grislain | unified | 4 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.0483 | 0.0412 | 0.0238 | 0.9460 | 1.1505 |
| auc | value | grislain | unified | 4 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.6182 | 0.0285 | 0.0165 | 0.5473 | 0.6891 |
| auc | delta | grislain | unified | 4 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0023 | 0.0242 | 0.0140 | -0.0578 | 0.0623 |
| auc | ratio | grislain | unified | 4 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.0037 | 0.0388 | 0.0224 | 0.9073 | 1.1001 |
| auc | value | grislain | unified | 8 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.5885 | 0.0141 | 0.0081 | 0.5536 | 0.6235 |
| auc | delta | grislain | unified | 8 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | -0.0163 | 0.0171 | 0.0098 | -0.0587 | 0.0260 |
| auc | ratio | grislain | unified | 8 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.9731 | 0.0279 | 0.0161 | 0.9037 | 1.0425 |
| auc | value | grislain | unified | 8 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.6467 | 0.0024 | 0.0014 | 0.6409 | 0.6526 |
| auc | delta | grislain | unified | 8 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0419 | 0.0061 | 0.0035 | 0.0266 | 0.0571 |
| auc | ratio | grislain | unified | 8 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.0693 | 0.0106 | 0.0061 | 1.0429 | 1.0957 |
| auc | value | grislain | unified | 8 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.6283 | 0.0248 | 0.0143 | 0.5666 | 0.6900 |
| auc | delta | grislain | unified | 8 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0234 | 0.0297 | 0.0171 | -0.0503 | 0.0972 |
| auc | ratio | grislain | unified | 8 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.0390 | 0.0494 | 0.0285 | 0.9164 | 1.1616 |
| auc | value | grislain | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.6896 | 0.0206 | 0.0119 | 0.6383 | 0.7408 |
| auc | delta | grislain | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | -0.0003 | 0.0028 | 0.0016 | -0.0073 | 0.0067 |
| auc | ratio | grislain | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.9996 | 0.0041 | 0.0024 | 0.9893 | 1.0099 |
| auc | value | grislain | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.8220 | 0.0061 | 0.0035 | 0.8068 | 0.8372 |
| auc | delta | grislain | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.1321 | 0.0170 | 0.0098 | 0.0898 | 0.1744 |
| auc | ratio | grislain | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.1921 | 0.0303 | 0.0175 | 1.1168 | 1.2675 |
| auc | value | grislain | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.7523 | 0.0171 | 0.0099 | 0.7098 | 0.7948 |
| auc | delta | grislain | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0624 | 0.0103 | 0.0060 | 0.0368 | 0.0881 |
| auc | ratio | grislain | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.0907 | 0.0167 | 0.0096 | 1.0494 | 1.1321 |
| auc | value | naive | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.9715 | 0.0113 | 0.0065 | 0.9436 | 0.9995 |
| auc | delta | naive | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0004 | 0.0031 | 0.0018 | -0.0074 | 0.0082 |
| auc | ratio | naive | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.0004 | 0.0032 | 0.0019 | 0.9925 | 1.0084 |
| auc | value | naive | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.9891 | 0.0072 | 0.0041 | 0.9713 | 1.0069 |
| auc | delta | naive | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0180 | 0.0085 | 0.0049 | -0.0032 | 0.0391 |
| auc | ratio | naive | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.0186 | 0.0090 | 0.0052 | 0.9962 | 1.0410 |
| auc | value | naive | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.9695 | 0.0097 | 0.0056 | 0.9454 | 0.9936 |
| auc | delta | naive | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | -0.0017 | 0.0042 | 0.0024 | -0.0121 | 0.0088 |
| auc | ratio | naive | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.9983 | 0.0043 | 0.0025 | 0.9875 | 1.0091 |
| tpr_at_fpr_01 | value | grislain | unified | 0.5 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0413 | 0.0101 | 0.0058 | 0.0163 | 0.0663 |
| tpr_at_fpr_01 | delta | grislain | unified | 0.5 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0040 | 0.0069 | 0.0040 | -0.0132 | 0.0212 |
| tpr_at_fpr_01 | ratio | grislain | unified | 0.5 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.1414 | 0.2150 | 0.1241 | 0.6073 | 1.6755 |
| tpr_at_fpr_01 | value | grislain | unified | 0.5 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0507 | 0.0180 | 0.0104 | 0.0059 | 0.0955 |
| tpr_at_fpr_01 | delta | grislain | unified | 0.5 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0133 | 0.0220 | 0.0127 | -0.0414 | 0.0681 |
| tpr_at_fpr_01 | ratio | grislain | unified | 0.5 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.4798 | 0.7219 | 0.4168 | -0.3137 | 3.2733 |
| tpr_at_fpr_01 | value | grislain | unified | 0.5 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0413 | 0.0046 | 0.0027 | 0.0299 | 0.0528 |
| tpr_at_fpr_01 | delta | grislain | unified | 0.5 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0040 | 0.0069 | 0.0040 | -0.0132 | 0.0212 |
| tpr_at_fpr_01 | ratio | grislain | unified | 0.5 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.1667 | 0.2887 | 0.1667 | 0.4495 | 1.8838 |
| tpr_at_fpr_01 | value | grislain | unified | 1 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0467 | 0.0115 | 0.0067 | 0.0180 | 0.0754 |
| tpr_at_fpr_01 | delta | grislain | unified | 1 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0080 | 0.0106 | 0.0061 | -0.0183 | 0.0343 |
| tpr_at_fpr_01 | ratio | grislain | unified | 1 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.2037 | 0.2625 | 0.1516 | 0.5515 | 1.8560 |
| tpr_at_fpr_01 | value | grislain | unified | 1 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0640 | 0.0174 | 0.0101 | 0.0207 | 0.1073 |
| tpr_at_fpr_01 | delta | grislain | unified | 1 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0253 | 0.0151 | 0.0087 | -0.0123 | 0.0630 |
| tpr_at_fpr_01 | ratio | grislain | unified | 1 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.6407 | 0.3659 | 0.2112 | 0.7318 | 2.5497 |
| tpr_at_fpr_01 | value | grislain | unified | 1 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0453 | 0.0162 | 0.0093 | 0.0052 | 0.0855 |
| tpr_at_fpr_01 | delta | grislain | unified | 1 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0067 | 0.0167 | 0.0096 | -0.0347 | 0.0480 |
| tpr_at_fpr_01 | ratio | grislain | unified | 1 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.1778 | 0.4221 | 0.2437 | 0.1292 | 2.2264 |
| tpr_at_fpr_01 | value | grislain | unified | 2 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0493 | 0.0122 | 0.0071 | 0.0190 | 0.0797 |
| tpr_at_fpr_01 | delta | grislain | unified | 2 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0067 | 0.0180 | 0.0104 | -0.0381 | 0.0515 |
| tpr_at_fpr_01 | ratio | grislain | unified | 2 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.1995 | 0.4586 | 0.2648 | 0.0602 | 2.3388 |
| tpr_at_fpr_01 | value | grislain | unified | 2 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0667 | 0.0167 | 0.0096 | 0.0253 | 0.1080 |
| tpr_at_fpr_01 | delta | grislain | unified | 2 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0240 | 0.0200 | 0.0115 | -0.0257 | 0.0737 |
| tpr_at_fpr_01 | ratio | grislain | unified | 2 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.6044 | 0.5728 | 0.3307 | 0.1813 | 3.0275 |
| tpr_at_fpr_01 | value | grislain | unified | 2 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0520 | 0.0069 | 0.0040 | 0.0348 | 0.0692 |
| tpr_at_fpr_01 | delta | grislain | unified | 2 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0093 | 0.0023 | 0.0013 | 0.0036 | 0.0151 |
| tpr_at_fpr_01 | ratio | grislain | unified | 2 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.2205 | 0.0531 | 0.0306 | 1.0887 | 1.3523 |
| tpr_at_fpr_01 | value | grislain | unified | 4 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0547 | 0.0046 | 0.0027 | 0.0432 | 0.0661 |
| tpr_at_fpr_01 | delta | grislain | unified | 4 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0040 | 0.0069 | 0.0040 | -0.0132 | 0.0212 |
| tpr_at_fpr_01 | ratio | grislain | unified | 4 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.1004 | 0.1411 | 0.0814 | 0.7499 | 1.4508 |
| tpr_at_fpr_01 | value | grislain | unified | 4 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0880 | 0.0183 | 0.0106 | 0.0425 | 0.1335 |
| tpr_at_fpr_01 | delta | grislain | unified | 4 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0373 | 0.0295 | 0.0170 | -0.0359 | 0.1106 |
| tpr_at_fpr_01 | ratio | grislain | unified | 4 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.8390 | 0.6862 | 0.3962 | 0.1343 | 3.5437 |
| tpr_at_fpr_01 | value | grislain | unified | 4 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0733 | 0.0129 | 0.0074 | 0.0414 | 0.1053 |
| tpr_at_fpr_01 | delta | grislain | unified | 4 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0227 | 0.0220 | 0.0127 | -0.0321 | 0.0774 |
| tpr_at_fpr_01 | ratio | grislain | unified | 4 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.5152 | 0.5007 | 0.2891 | 0.2713 | 2.7590 |
| tpr_at_fpr_01 | value | grislain | unified | 8 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0600 | 0.0040 | 0.0023 | 0.0501 | 0.0699 |
| tpr_at_fpr_01 | delta | grislain | unified | 8 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | -0.0040 | 0.0208 | 0.0120 | -0.0556 | 0.0476 |
| tpr_at_fpr_01 | ratio | grislain | unified | 8 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.0054 | 0.2683 | 0.1549 | 0.3388 | 1.6720 |
| tpr_at_fpr_01 | value | grislain | unified | 8 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0947 | 0.0092 | 0.0053 | 0.0717 | 0.1176 |
| tpr_at_fpr_01 | delta | grislain | unified | 8 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0307 | 0.0220 | 0.0127 | -0.0241 | 0.0854 |
| tpr_at_fpr_01 | ratio | grislain | unified | 8 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.5952 | 0.4985 | 0.2878 | 0.3568 | 2.8337 |
| tpr_at_fpr_01 | value | grislain | unified | 8 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0827 | 0.0197 | 0.0114 | 0.0336 | 0.1317 |
| tpr_at_fpr_01 | delta | grislain | unified | 8 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0187 | 0.0439 | 0.0253 | -0.0903 | 0.1277 |
| tpr_at_fpr_01 | ratio | grislain | unified | 8 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.4717 | 0.7106 | 0.4103 | -0.2936 | 3.2370 |
| tpr_at_fpr_01 | value | grislain | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0680 | 0.0280 | 0.0162 | -0.0016 | 0.1376 |
| tpr_at_fpr_01 | delta | grislain | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| tpr_at_fpr_01 | ratio | grislain | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| tpr_at_fpr_01 | value | grislain | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.2520 | 0.0762 | 0.0440 | 0.0627 | 0.4413 |
| tpr_at_fpr_01 | delta | grislain | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.1840 | 0.0961 | 0.0555 | -0.0547 | 0.4227 |
| tpr_at_fpr_01 | ratio | grislain | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 4.5953 | 3.2725 | 1.8894 | -3.5346 | 12.7252 |
| tpr_at_fpr_01 | value | grislain | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.1667 | 0.0129 | 0.0074 | 0.1347 | 0.1986 |
| tpr_at_fpr_01 | delta | grislain | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0987 | 0.0151 | 0.0087 | 0.0610 | 0.1363 |
| tpr_at_fpr_01 | ratio | grislain | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 2.7907 | 1.2420 | 0.7170 | -0.2947 | 5.8762 |
| tpr_at_fpr_01 | value | naive | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.3053 | 0.1642 | 0.0948 | -0.1026 | 0.7133 |
| tpr_at_fpr_01 | delta | naive | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0227 | 0.0395 | 0.0228 | -0.0754 | 0.1207 |
| tpr_at_fpr_01 | ratio | naive | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.0891 | 0.1348 | 0.0778 | 0.7541 | 1.4241 |
| tpr_at_fpr_01 | value | naive | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.7960 | 0.1042 | 0.0602 | 0.5371 | 1.0549 |
| tpr_at_fpr_01 | delta | naive | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.5133 | 0.0820 | 0.0473 | 0.3096 | 0.7171 |
| tpr_at_fpr_01 | ratio | naive | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 3.5895 | 2.1671 | 1.2511 | -1.7942 | 8.9732 |
| tpr_at_fpr_01 | value | naive | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.3720 | 0.1557 | 0.0899 | -0.0149 | 0.7589 |
| tpr_at_fpr_01 | delta | naive | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0893 | 0.0489 | 0.0282 | -0.0321 | 0.2108 |
| tpr_at_fpr_01 | ratio | naive | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.4839 | 0.5356 | 0.3092 | 0.1532 | 2.8146 |
| tpr_at_fpr_05 | value | grislain | unified | 0.5 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0627 | 0.0061 | 0.0035 | 0.0475 | 0.0778 |
| tpr_at_fpr_05 | delta | grislain | unified | 0.5 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | -0.0027 | 0.0083 | 0.0048 | -0.0234 | 0.0180 |
| tpr_at_fpr_05 | ratio | grislain | unified | 0.5 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.9634 | 0.1256 | 0.0725 | 0.6513 | 1.2755 |
| tpr_at_fpr_05 | value | grislain | unified | 0.5 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0707 | 0.0439 | 0.0253 | -0.0383 | 0.1797 |
| tpr_at_fpr_05 | delta | grislain | unified | 0.5 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0053 | 0.0484 | 0.0279 | -0.1149 | 0.1255 |
| tpr_at_fpr_05 | ratio | grislain | unified | 0.5 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.1176 | 0.7782 | 0.4493 | -0.8156 | 3.0509 |
| tpr_at_fpr_05 | value | grislain | unified | 0.5 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.1107 | 0.0220 | 0.0127 | 0.0559 | 0.1654 |
| tpr_at_fpr_05 | delta | grislain | unified | 0.5 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0453 | 0.0266 | 0.0154 | -0.0208 | 0.1115 |
| tpr_at_fpr_05 | ratio | grislain | unified | 0.5 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.7163 | 0.4775 | 0.2757 | 0.5301 | 2.9026 |
| tpr_at_fpr_05 | value | grislain | unified | 1 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0787 | 0.0092 | 0.0053 | 0.0557 | 0.1016 |
| tpr_at_fpr_05 | delta | grislain | unified | 1 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0093 | 0.0185 | 0.0107 | -0.0366 | 0.0552 |
| tpr_at_fpr_05 | ratio | grislain | unified | 1 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.1583 | 0.2670 | 0.1542 | 0.4950 | 1.8217 |
| tpr_at_fpr_05 | value | grislain | unified | 1 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.1093 | 0.0241 | 0.0139 | 0.0494 | 0.1692 |
| tpr_at_fpr_05 | delta | grislain | unified | 1 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0400 | 0.0250 | 0.0144 | -0.0221 | 0.1021 |
| tpr_at_fpr_05 | ratio | grislain | unified | 1 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.5917 | 0.4101 | 0.2368 | 0.5729 | 2.6105 |
| tpr_at_fpr_05 | value | grislain | unified | 1 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0867 | 0.0129 | 0.0074 | 0.0547 | 0.1186 |
| tpr_at_fpr_05 | delta | grislain | unified | 1 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0173 | 0.0220 | 0.0127 | -0.0374 | 0.0721 |
| tpr_at_fpr_05 | ratio | grislain | unified | 1 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.2792 | 0.3299 | 0.1904 | 0.4597 | 2.0986 |
| tpr_at_fpr_05 | value | grislain | unified | 2 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0867 | 0.0023 | 0.0013 | 0.0809 | 0.0924 |
| tpr_at_fpr_05 | delta | grislain | unified | 2 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0133 | 0.0227 | 0.0131 | -0.0432 | 0.0698 |
| tpr_at_fpr_05 | ratio | grislain | unified | 2 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.2632 | 0.3687 | 0.2129 | 0.3473 | 2.1791 |
| tpr_at_fpr_05 | value | grislain | unified | 2 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0987 | 0.0101 | 0.0058 | 0.0737 | 0.1237 |
| tpr_at_fpr_05 | delta | grislain | unified | 2 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0253 | 0.0205 | 0.0119 | -0.0257 | 0.0763 |
| tpr_at_fpr_05 | ratio | grislain | unified | 2 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.4324 | 0.4382 | 0.2530 | 0.3437 | 2.5211 |
| tpr_at_fpr_05 | value | grislain | unified | 2 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.1120 | 0.0183 | 0.0106 | 0.0665 | 0.1575 |
| tpr_at_fpr_05 | delta | grislain | unified | 2 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0387 | 0.0220 | 0.0127 | -0.0161 | 0.0934 |
| tpr_at_fpr_05 | ratio | grislain | unified | 2 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.6212 | 0.5291 | 0.3055 | 0.3067 | 2.9358 |
| tpr_at_fpr_05 | value | grislain | unified | 4 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.1027 | 0.0162 | 0.0093 | 0.0625 | 0.1428 |
| tpr_at_fpr_05 | delta | grislain | unified | 4 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0147 | 0.0227 | 0.0131 | -0.0418 | 0.0712 |
| tpr_at_fpr_05 | ratio | grislain | unified | 4 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.1812 | 0.2837 | 0.1638 | 0.4763 | 1.8860 |
| tpr_at_fpr_05 | value | grislain | unified | 4 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.1120 | 0.0120 | 0.0069 | 0.0822 | 0.1418 |
| tpr_at_fpr_05 | delta | grislain | unified | 4 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0240 | 0.0183 | 0.0106 | -0.0215 | 0.0695 |
| tpr_at_fpr_05 | ratio | grislain | unified | 4 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.2848 | 0.2388 | 0.1379 | 0.6916 | 1.8780 |
| tpr_at_fpr_05 | value | grislain | unified | 4 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.1133 | 0.0092 | 0.0053 | 0.0904 | 0.1363 |
| tpr_at_fpr_05 | delta | grislain | unified | 4 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0253 | 0.0162 | 0.0093 | -0.0148 | 0.0655 |
| tpr_at_fpr_05 | ratio | grislain | unified | 4 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.2993 | 0.2171 | 0.1254 | 0.7598 | 1.8387 |
| tpr_at_fpr_05 | value | grislain | unified | 8 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.1053 | 0.0189 | 0.0109 | 0.0584 | 0.1523 |
| tpr_at_fpr_05 | delta | grislain | unified | 8 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | -0.0133 | 0.0363 | 0.0210 | -0.1035 | 0.0768 |
| tpr_at_fpr_05 | ratio | grislain | unified | 8 | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.9242 | 0.3001 | 0.1733 | 0.1786 | 1.6697 |
| tpr_at_fpr_05 | value | grislain | unified | 8 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.1440 | 0.0280 | 0.0162 | 0.0744 | 0.2136 |
| tpr_at_fpr_05 | delta | grislain | unified | 8 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0253 | 0.0506 | 0.0292 | -0.1005 | 0.1512 |
| tpr_at_fpr_05 | ratio | grislain | unified | 8 | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.2825 | 0.5263 | 0.3039 | -0.0250 | 2.5900 |
| tpr_at_fpr_05 | value | grislain | unified | 8 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.1267 | 0.0546 | 0.0315 | -0.0090 | 0.2623 |
| tpr_at_fpr_05 | delta | grislain | unified | 8 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.0080 | 0.0723 | 0.0418 | -0.1717 | 0.1877 |
| tpr_at_fpr_05 | ratio | grislain | unified | 8 | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.1481 | 0.7027 | 0.4057 | -0.5976 | 2.8938 |
| tpr_at_fpr_05 | value | grislain | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.1827 | 0.0167 | 0.0096 | 0.1413 | 0.2240 |
| tpr_at_fpr_05 | delta | grislain | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | -0.0013 | 0.0023 | 0.0013 | -0.0071 | 0.0044 |
| tpr_at_fpr_05 | ratio | grislain | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.9933 | 0.0115 | 0.0067 | 0.9646 | 1.0220 |
| tpr_at_fpr_05 | value | grislain | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.4987 | 0.0372 | 0.0215 | 0.4063 | 0.5910 |
| tpr_at_fpr_05 | delta | grislain | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.3147 | 0.0220 | 0.0127 | 0.2599 | 0.3694 |
| tpr_at_fpr_05 | ratio | grislain | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 2.7165 | 0.1297 | 0.0749 | 2.3941 | 3.0388 |
| tpr_at_fpr_05 | value | grislain | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.3000 | 0.0454 | 0.0262 | 0.1871 | 0.4129 |
| tpr_at_fpr_05 | delta | grislain | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.1160 | 0.0317 | 0.0183 | 0.0371 | 0.1949 |
| tpr_at_fpr_05 | ratio | grislain | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 1.6278 | 0.1354 | 0.0782 | 1.2914 | 1.9643 |
| tpr_at_fpr_05 | value | naive | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.8587 | 0.0769 | 0.0444 | 0.6677 | 1.0496 |
| tpr_at_fpr_05 | delta | naive | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 0.0093 | 0.0300 | 0.0173 | -0.0653 | 0.0839 |
| tpr_at_fpr_05 | ratio | naive | unified | inf | canary | n_canaries=10,n_repeats=1 | canary_n_repeats | 1.00 | 1 | 3 | 1.0142 | 0.0398 | 0.0230 | 0.9153 | 1.1130 |
| tpr_at_fpr_05 | value | naive | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.9400 | 0.0500 | 0.0288 | 0.8159 | 1.0641 |
| tpr_at_fpr_05 | delta | naive | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 0.0907 | 0.0674 | 0.0389 | -0.0769 | 0.2582 |
| tpr_at_fpr_05 | ratio | naive | unified | inf | full_duplicate | duplication_rate=0.3,n_copies=1 | duplication_rate | 0.30 | 0 | 3 | 1.1145 | 0.0998 | 0.0576 | 0.8666 | 1.3625 |
| tpr_at_fpr_05 | value | naive | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.7840 | 0.0952 | 0.0550 | 0.5474 | 1.0206 |
| tpr_at_fpr_05 | delta | naive | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | -0.0653 | 0.0129 | 0.0074 | -0.0973 | -0.0334 |
| tpr_at_fpr_05 | ratio | naive | unified | inf | partial_overlap | n_hosts=3,overlap_rate=0.3 | overlap_rate | 0.30 | 0 | 3 | 0.9233 | 0.0086 | 0.0050 | 0.9018 | 0.9448 |

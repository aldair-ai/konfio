# Ablation (CV)

All numbers are pooled out-of-fold predictions on the 85% CV portion (5 grouped folds); the test set is untouched. 95% CIs resample groups (normalized text). Diagnostic rows use a different split or a non-default C and are never used for selection.

**Baselines**

| Experiment | What changes | Macro F1 [95% CI] | Micro F1 | Hamming | Samples F1 | temp F1 | Mean PR-AUC |
|---|---|---|---|---|---|---|---|
| B0_all_zeros | Predict nothing | 0.000 [0.000, 0.000] | 0.000 | 0.1143 | 0.000 | 0.000 | 0.114 |
| B1_most_frequent | Always the most frequent label (inv) | 0.051 [0.050, 0.053] | 0.323 | 0.1451 | 0.323 | 0.000 | 0.113 |

**TF-IDF + logistic regression**

| Experiment | What changes | Macro F1 [95% CI] | Micro F1 | Hamming | Samples F1 | temp F1 | Mean PR-AUC |
|---|---|---|---|---|---|---|---|
| B2_tfidf_lr_br__thr0.5_norules | B2 scores, fixed 0.5, no rules | 0.664 [0.652, 0.676] | 0.726 | 0.0683 | 0.724 | 0.370 | 0.707 |
| B2_tfidf_lr_br__tuned_norules | + cross-fitted thresholds | 0.666 [0.654, 0.678] | 0.735 | 0.0639 | 0.726 | 0.374 | 0.707 |
| B2_tfidf_lr_br__tuned_atleastone | + at-least-one rule | 0.668 [0.656, 0.680] | 0.734 | 0.0658 | 0.747 | 0.378 | 0.707 |
| B2_tfidf_lr_br__tuned_noexclusive | + no-exclusive rule (no at-least-one) | 0.669 [0.657, 0.682] | 0.738 | 0.0630 | 0.727 | 0.380 | 0.707 |
| B2_tfidf_lr_br | B2: thresholds + both rules (reference) | 0.671 [0.659, 0.684] | 0.736 | 0.0650 | 0.748 | 0.384 | 0.707 |
| B2_margin | B2, at-least-one picks the largest margin | 0.673 [0.661, 0.686] | 0.738 | 0.0646 | 0.750 | 0.401 | 0.707 |
| B3_tfidf_lr_chain | Classifier chain instead of binary relevance | 0.665 [0.652, 0.678] | 0.735 | 0.0642 | 0.743 | 0.372 | 0.702 |
| B2_codebook | B2 + 10 codebook similarities | 0.666 [0.654, 0.679] | 0.731 | 0.0655 | 0.743 | 0.392 | 0.706 |

**Frozen sentence embeddings**

| Experiment | What changes | Macro F1 [95% CI] | Micro F1 | Hamming | Samples F1 | temp F1 | Mean PR-AUC |
|---|---|---|---|---|---|---|---|
| E1_e5 | e5 embeddings + LR, fixed 0.5 | 0.580 [0.568, 0.592] | 0.653 | 0.0888 | 0.651 | 0.236 | 0.607 |
| E2_e5 | + codebook similarities | 0.583 [0.570, 0.594] | 0.655 | 0.0881 | 0.653 | 0.238 | 0.607 |
| E3_e5 | + thresholds + rules | 0.601 [0.589, 0.613] | 0.672 | 0.0811 | 0.684 | 0.250 | 0.607 |
| E1_mpnet | mpnet embeddings + LR, fixed 0.5 | 0.574 [0.561, 0.586] | 0.640 | 0.0933 | 0.645 | 0.288 | 0.601 |
| E2_mpnet | + codebook similarities | 0.573 [0.560, 0.585] | 0.641 | 0.0931 | 0.645 | 0.294 | 0.602 |
| E3_mpnet | + thresholds + rules | 0.588 [0.574, 0.601] | 0.651 | 0.0871 | 0.664 | 0.291 | 0.602 |

**Fine-tuned e5 (two-stage)**

| Experiment | What changes | Macro F1 [95% CI] | Micro F1 | Hamming | Samples F1 | temp F1 | Mean PR-AUC |
|---|---|---|---|---|---|---|---|
| E4_ft_head | Fine-tuned head probabilities | 0.701 [0.688, 0.714] | 0.758 | 0.0575 | 0.772 | 0.410 | 0.753 |
| E5_ft_emb_lr | Fine-tuned embedding + LR | 0.637 [0.625, 0.649] | 0.693 | 0.0749 | 0.711 | 0.320 | 0.633 |
| E6_hybrid_lr | Embedding + B2 probs + codebook, LR | 0.642 [0.629, 0.653] | 0.693 | 0.0762 | 0.710 | 0.271 | 0.643 |
| E7_hybrid_lgbm | Same hybrid features, LightGBM | 0.702 [0.691, 0.715] | 0.759 | 0.0578 | 0.774 | 0.355 | 0.745 |

**Structure and blending**

| Experiment | What changes | Macro F1 [95% CI] | Micro F1 | Hamming | Samples F1 | temp F1 | Mean PR-AUC |
|---|---|---|---|---|---|---|---|
| E8_hierarchical | no vs business, then E7 on business rows | 0.693 [0.680, 0.706] | 0.753 | 0.0585 | 0.763 | 0.377 | 0.720 |
| E9_blend_B2_E7 | Blend B2 + E7, weight cross-fitted | 0.712 [0.700, 0.724] | 0.766 | 0.0562 | 0.778 | 0.399 | 0.747 |
| **E9b_blend_B2_E4** (selected) | Blend B2 + E4, weight cross-fitted | 0.715 [0.703, 0.727] | 0.771 | 0.0547 | 0.784 | 0.421 | 0.762 |

**Not comparable: diagnostics**

| Experiment | What changes | Macro F1 [95% CI] | Micro F1 | Hamming | Samples F1 | temp F1 | Mean PR-AUC |
|---|---|---|---|---|---|---|---|
| B2_random_split | B2 on a row-level split (duplicate leakage) | 0.679 [0.666, 0.691] | 0.741 | 0.0633 | 0.751 | 0.420 | 0.710 |
| E1_e5_C0.01 | E1_e5 with C = 0.01 | 0.579 [0.569, 0.589] | 0.650 | 0.1031 | 0.679 | 0.280 | 0.696 |
| E1_e5_C0.1 | E1_e5 with C = 0.1 | 0.605 [0.593, 0.615] | 0.671 | 0.0875 | 0.680 | 0.287 | 0.656 |

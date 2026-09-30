# Leakage analysis: Proposed hybrid (RF+XGB)

Balancing before splitting inflates results in two ways. This separates them.

| Setting                                                                   |   Precision |   Recall |    F1 |
|:--------------------------------------------------------------------------|------------:|---------:|------:|
| Paper protocol, as reported (test set ~50% APT, includes synthetic flows) |       94.81 |    99.43 | 97.06 |
| Paper protocol, re-scored at the real class ratio                         |       84.73 |    99.43 | 91.49 |
| Leakage-free protocol (real flows, real class ratio)                      |       81.42 |    97.77 | 88.85 |

- **Class-ratio effect:** re-scoring at the real ratio (5,694 APT vs 18,739 benign) moves F1 from 97.06% to 91.49% and precision from 94.81% to 84.73%: 5.6 of the 8.2 F1 points of difference.
- **Leakage effect:** the remaining 2.6 F1 points. 70% of the APT flows in the paper-protocol test set are synthetic; the model misses 0.34% of synthetic APT flows but 1.12% of real ones (leakage-free: 2.23%).

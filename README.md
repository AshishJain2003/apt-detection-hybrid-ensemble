# APT Detection with a Hybrid Ensemble (RF + XGBoost) on DAPT2020

Implementation of **Saini N., Bhat Kasaragod V., Prakasha K., Das A.K. (2023). _A hybrid ensemble
machine learning model for detecting APT attacks based on network behavior anomaly detection._
Concurrency and Computation: Practice and Experience 35(28):e7865.**
[doi:10.1002/cpe.7865](https://doi.org/10.1002/cpe.7865), re-run on an **APT-specific dataset**.

## 1. Why the dataset was changed

The paper claims to detect *Advanced Persistent Threats*, but evaluates on CSE-CIC-IDS2018,
CIC-IDS2017, NSL-KDD and UNSW-NB15. These are **generic intrusion-detection datasets**: their
attacks are mostly DoS/DDoS, brute force, botnets and web attacks, recorded as isolated events.
None of them is organised around the multi-stage, low-and-slow behaviour that defines an APT.

This project uses **DAPT2020** instead:

| | |
|---|---|
| Paper | Myneni S. et al., *DAPT 2020 – Constructing a Benchmark Dataset for Advanced Persistent Threats*, in *Deployable Machine Learning for Security Defense* (MLHat 2020), Springer CCIS ([doi:10.1007/978-3-030-59621-7_8](https://doi.org/10.1007/978-3-030-59621-7_8)) |
| Source | https://www.kaggle.com/datasets/sowmyamyneni/dapt2020 (published by the dataset's author) |
| What it is | 5 days of traffic from a public web server and a private network, with a red team running a full APT campaign |
| Labels | Every flow has an **Activity** (e.g., Network Scan, SQL Injection, Backdoor, Privilege Escalation) and an **APT Stage**: Benign, Reconnaissance, Establish Foothold, Lateral Movement, Data Exfiltration |
| Features | 76 CICFlowMeter flow features, **the same feature family as CIC-IDS2017/CSE-CIC-IDS2018**, so the paper's pipeline applies unchanged |
| Size used | 86,691 labelled flows (the 10 `*.pcap_Flow.csv` files, ~45 MB; the 9.7 GB rest is raw PCAPs and logs) |

Because DAPT2020 uses the same features as the paper's main datasets, results can be compared
directly, and its stage labels allow an extra experiment the paper could not run: identifying
*which APT stage* a flow belongs to.

Alternatives that were considered: SCVIC-APT-2021 (paywalled on IEEE DataPort), Unraveled (2023,
semi-synthetic, 835 MB) and CICAPT-IIoT (2024, Industrial IoT). DAPT2020 is the most widely used
APT network benchmark and the closest fit to the paper's setup.

### Data issues found and handled

* `enp0s3-pvt-thursday.pcap_Flow.csv` has **no header row**. A plain `pd.read_csv` treats its
  first flow as the header and loses 2,314 of the 2,451 Lateral Movement flows.
  `src/data_loader.py` detects this. With it fixed, the benign count is 63,712, which matches the
  count reported for DAPT2020 in later work ([DSRL-APT-2023, *ISeCure* 17(2), 2025](https://www.isecure-journal.com/article_214212_28d18c445b2b603021b857587fd98445.pdf)).
* The CSVs are parsed with `float_precision="round_trip"` (exact decimal-to-float conversion).
  pandas' default fast parser can differ in the last bit across versions and platforms, which
  changes which flows count as duplicates and therefore every split.
* Benign labels are spelled inconsistently (`BENIGN`, `Benign`, `Normal`); they are normalised.
* 12 constant columns and 5,248 duplicate flows are removed (paper Section 3.2).
* Possible label noise: on the Tuesday reconnaissance capture, 4,967 benign-labelled flows
  (4,432 spelled `BENIGN`, 535 `Benign`) involve the external attacker host. This is not proof of
  mislabelling: the attacker hosts also appear in 219 normal flows on Monday, when no attacks are
  labelled. See the error analysis below.

## 2. What was implemented (faithful to the paper)

Every step below comes from the paper (Sections 3.2–3.3, Algorithm 1, Table 7):

| Step | Paper | This implementation |
|---|---|---|
| Drop identifiers | Timestamp, Flow ID, Src IP, Dst IP, Src Port | same |
| Cleaning | remove NaN/inf, constant features, duplicates; dummy variables | same (Protocol one-hot encoded) |
| Target | Normal = 0, all attacks aggregated into one anomaly class = 1 | same (Benign = 0, any APT stage = 1) |
| Balancing | SMOTETomek | `imblearn` SMOTETomek with its default cleaning (Tomek links removed from both classes) |
| Split | 70:30 (their Fig. 11 test set of 15,583 rows = 30%) | 70:30, stratified |
| Feature selection | Pearson filter at 0.90, then mutual information top-30 | same (fitted on training data) |
| Hybrid model | `combo.SimpleClassifierAggregator([RandomForestClassifier(), XGBClassifier()], method='average')` | exact re-implementation (see below) |
| Hyperparameters | 100 trees, learning rate 1.0, max depth 4 | same; the depth limit is applied to XGBoost only (see Deviations) |
| Baselines | KNN, DT, SVM, RF, XGBoost, GB, Kernel SVM, AdaBoost, NB, LR, CNN, ANN, MLP | all 13 |
| Metrics | Accuracy, Precision, Recall, F1, TNR, FPR, FNR, AUC | all 8 |

**How the paper's hybrid actually combines its models.** Reading the `combo` library source shows
that `SimpleClassifierAggregator.predict()` averages the two models' *predicted labels* and
predicts 1 when the average is `>= 0.5`. With two binary models, **a flow is flagged as APT if
either Random Forest or XGBoost flags it** (an OR rule), which raises recall at the cost of more
false positives. `HybridRFXGB(vote="combo")` reproduces this exactly: running `combo`'s own
`SimpleClassifierAggregator` on our fitted models gives identical predictions and probabilities
on all 24,433 leakage-free test flows. A soft-voting variant
(average of probabilities, `vote="soft"`) is also reported, because the OR rule is undefined for
the multi-class stage experiment.

### Deviations and assumptions (documented, not hidden)

1. **Paper inconsistency:** the text says 70:30 but Algorithm 1 says 80:20. The paper's own
   confusion matrix (15,583 test rows out of 51,942) shows 70:30 was used, so 70:30 is used here.
2. **Baseline hyperparameters** are not given in the paper; library defaults are used. RF and
   XGBoost use the hybrid's settings so the hybrid is compared against its own components.
3. **Kernel SVM** (RBF) is trained on a stratified 20,000-row subsample; RBF SVM training grows
   quadratically with the number of rows.
4. **CNN/ANN/MLP** architectures are not described in the paper. Standard architectures are used
   (`src/deep_models.py`), trained for up to 200 epochs (as in the paper) with early stopping on a
   validation split. They train on the GPU when available (Apple Silicon MPS here, or CUDA). GPU
   kernels are not bit-for-bit deterministic, so CNN/ANN/MLP numbers can move slightly between runs;
   the classical models reproduce exactly.
5. **AUC** is computed from predicted probabilities. The paper's ROC curves have a single corner,
   meaning they were drawn from hard 0/1 predictions; that "AUC" equals balanced accuracy.
6. **Max depth 4 applies to XGBoost only.** Algorithm 1 lists "100 trees, learning rate 1.0, max
   depth 4" for the hybrid without saying which model each belongs to. Learning rate only exists for
   XGBoost, and Random Forest trees are normally grown fully, so depth 4 is applied to XGBoost only.
   This matters: also limiting Random Forest to depth 4 lowers the hybrid's leakage-free F1 from
   88.85% to 78.99%.
7. **Reproducibility.** Classical models are seeded (`SEED = 42`) and reproduce exactly; CSV parsing
   is exact (see Data issues); package versions are pinned in `requirements.txt`. CNN/ANN/MLP on a GPU
   can differ slightly between runs.

## 3. A methodological problem in the paper, and the fix

The paper applies SMOTETomek to the **whole dataset before splitting**. Section 3.2: "After
converting it into balanced dataset, the CSE-CIC-IDS2018 dataset has been divided into two parts:
training and testing." This has two consequences:

* the test set contains **synthetic SMOTE flows**, interpolated from real flows that may sit in the
  training set, so information leaks from training into testing;
* the test set is artificially 50/50 balanced, which inflates precision and F1 compared with real
  traffic, where benign flows dominate.

Both protocols are therefore run, with everything else identical. The difference between them
comes from the order of the two steps; Section 4.2 separates the two consequences above.

* **Paper protocol:** clean → SMOTETomek on all data → 70:30 split → feature selection → train/test.
* **Leakage-free protocol:** clean → 70:30 split → SMOTETomek on the **training set only** →
  feature selection → train/test on untouched real flows.

## 4. Results

Full tables for all 15 models are in [results/summary.md](results/summary.md); figures are in
[results/figures/](results/figures/). All numbers are on DAPT2020 (test set = 30% of flows).

### 4.1 Binary detection (Benign vs APT): the paper's experiment

| Proposed hybrid (RF+XGB) | Accuracy | Precision | Recall | F1 | FPR | AUC |
|---|---|---|---|---|---|---|
| Paper, CSE-CIC-IDS2018 (their Table 7) | 98.92 | 99.47 | 98.35 | 98.90 | 0.52 | 98.91* |
| **This work, DAPT2020, paper protocol** | 96.99 | 94.81 | 99.43 | 97.06 | 5.44 | 99.62 |
| **This work, DAPT2020, leakage-free** | 94.28 | 81.42 | 97.77 | 88.85 | 6.78 | 99.08 |

\* the paper's AUC is computed from hard predictions (see Deviations, item 5).

Best models under the leakage-free protocol (single run, seed 42):

| Model | Accuracy | Precision | Recall | F1 | FPR | AUC |
|---|---|---|---|---|---|---|
| Hybrid RF+XGB (soft vote) | 94.62 | 83.30 | 96.21 | 89.29 | 5.86 | 99.08 |
| Random forest | 94.58 | 83.78 | 95.17 | 89.11 | 5.60 | 98.91 |
| XGBoost | 94.43 | 83.14 | 95.43 | 88.86 | 5.88 | 99.01 |
| **Proposed hybrid (RF+XGB, OR rule)** | 94.28 | 81.42 | 97.77 | 88.85 | 6.78 | 99.08 |
| CNN | 93.15 | 77.76 | 98.91 | 87.07 | 8.60 | 98.88 |
| MLP | 93.11 | 77.67 | 98.86 | 86.99 | 8.63 | 98.87 |
| Naive Bayes (worst) | 75.15 | 48.38 | 99.12 | 65.02 | 32.14 | 88.05 |

### 4.2 Where the inflation comes from

Balancing before splitting changes two things at once: the test set becomes ~50% APT instead of
the real 23%, and synthetic flows enter it. `src/leakage_analysis.py` separates the two for
the hybrid ([details](results/binary_paper_protocol/leakage_analysis.md)). Recall and FPR do not
depend on the class ratio, so the paper-protocol predictions can be re-scored exactly at the real ratio.

| Setting | Precision | Recall | F1 |
|---|---|---|---|
| Paper protocol, as reported (~50% APT, synthetic flows included) | 94.81 | 99.43 | 97.06 |
| Same predictions, re-scored at the real class ratio | 84.73 | 99.43 | 91.49 |
| Leakage-free protocol | 81.42 | 97.77 | 88.85 |

* **Class ratio: 5.6 of the 8.2 F1 points.** A 50/50 test set has far fewer benign flows to raise
  false alarms on, so precision looks higher (94.8% vs 84.7% for the same predictions).
* **Leakage: the remaining 2.6 points.** 70% of the APT flows in the paper-protocol test set are
  synthetic; the model misses 0.34% of them but 1.12% of real APT flows
  (leakage-free protocol: 2.23%).

### 4.3 Does the hybrid beat its own components?

Leakage-free, mean ± std over 5 random seeds ([details](results/binary_leakage_free/seed_robustness.md)):

| Model | F1 | Recall | FPR | AUC |
|---|---|---|---|---|
| Proposed hybrid (OR rule, as published) | 88.82 ± 0.18 | 96.93 ± 0.38 | 6.48 ± 0.14 | 99.02 ± 0.03 |
| **Hybrid, soft vote** | **89.21 ± 0.24** | 95.09 ± 0.47 | 5.50 ± 0.16 | 99.02 ± 0.03 |
| Random forest | 89.00 ± 0.21 | 94.63 ± 0.22 | 5.47 ± 0.12 | 98.83 ± 0.02 |
| XGBoost | 89.01 ± 0.25 | 94.68 ± 0.51 | 5.49 ± 0.17 | 98.95 ± 0.06 |

The published OR hybrid does not beat its components: its F1 is at or slightly below theirs, with
the highest recall but also the highest false positive rate. The soft vote has the best mean F1;
it is ahead of XGBoost and the OR hybrid on 5/5 seeds and of Random Forest on
4/5, a small margin (about 0.2 points).

![Protocol comparison](results/figures/03_protocol_comparison.png)

### 4.4 APT-stage classification (extension, 5 classes, leakage-free)

| Model | Accuracy | Macro F1 | Macro F1 without exfiltration | FPR | APT detected | Recon | Foothold | Lateral | Exfiltration |
|---|---|---|---|---|---|---|---|---|---|
| Hybrid RF+XGB (soft vote) | 93.89 | 79.07 | 86.34 | 7.08 | 97.75 | 97.6 | 98.3 | 91.6 | 2/5 |
| Random forest | 93.78 | 78.88 | 86.10 | 7.02 | 97.23 | 96.4 | 98.2 | 92.1 | 2/5 |
| XGBoost | 93.84 | 76.99 | 86.23 | 7.08 | 97.49 | 97.5 | 98.2 | 90.4 | 2/5 |
| CNN | 92.65 | 73.39 | – | 8.77 | 98.72 | 98.1 | 97.8 | 92.2 | 2/5 |
| Gradient boosting | 92.29 | 69.58 | – | 9.48 | 99.17 | 99.2 | 97.9 | 93.6 | 3/5 |

("APT detected" = share of APT flows assigned to any APT stage; FPR = benign flows assigned to any
APT stage. Stage columns are recall in %.) Macro F1 should not be used to rank these models:
Data Exfiltration has only 5 test flows, so a single flow moves it by several points. Without that
class, Random Forest, XGBoost and the soft vote are within
0.2 points of each other.

![Per-stage recall](results/figures/06_stage_recall_heatmap.png)

### 4.5 Where the errors come from

From [error_analysis.md](results/binary_leakage_free/error_analysis.md) (hybrid, leakage-free):

* 50.6% of false alarms are benign-labelled flows to or from the external attacker hosts. This
  *may* be label noise, but it is not proof: the attacker hosts also appear in normal Monday traffic.
* 42.2% of false alarms fall on Monday, when no flow is labelled as an attack. These are genuine
  false alarms on normal traffic.
* By count, most missed APT flows are Account Discovery (57 of 127). Rare, low-volume steps (CSRF,
  command injection, SQL injection, privilege escalation) have the highest miss *rates* but few flows.

## 5. Key findings

1. **The paper's method works on real APT traffic, but not at the level claimed.** AUC is about
   99%, but the false positive rate is 6.78%, about 13× the 0.52% the paper reports.
   APT traffic is built to look like normal traffic.
2. **Balancing before splitting inflates the reported scores.** F1 falls from 97.06% to
   88.85% on a leakage-free test set: 5.6 points come from the unrealistic 50/50 test set and
   2.6 from synthetic flows leaking into the test set.
3. **The published hybrid does not beat its own components.** Its OR rule trades false alarms for
   recall. Averaging probabilities (soft vote) is slightly better: best mean F1 over 5 seeds.
4. **Deep models do not beat tree ensembles** on these tabular flow features (CNN F1
   87.07 vs Random Forest 89.11), consistent with the paper's own Table 7.
5. **APT stages can be identified, except exfiltration.** Reconnaissance 97.6%, foothold
   98.3%, lateral movement 91.6%; exfiltration 2 of 5, with only 15 flows in
   the whole dataset.
6. **The weak spots are identifiable.** Most misses are Account Discovery; the rare steps are missed most
   often; many false alarms fall on normal Monday traffic.
7. **Most informative feature:** Dst Port, also the top feature in the paper's CSE-CIC-IDS2018 ranking.

### Other problems found in the paper

These can be checked directly against the paper's tables:

* The CNN, ANN and MLP rows are **identical** in Table 7 (CSE-CIC-IDS2018), Table 9 (NSL-KDD) and
  Table 10 (UNSW-NB15), e.g. CNN 98.22 / 98.25 / 98.21 / 98.22 in all three.
* The CNN row is **internally inconsistent**: with recall 98.21% and TNR 99.51% on a (near-)balanced
  test set, precision must be about 99.5% and accuracy about 98.9%, not 98.25% and 98.22%.
* Table 7 gives XGBoost a **higher** false positive rate (0.56%) than the hybrid (0.52%). Under the
  OR rule, the hybrid flags everything XGBoost flags, so this is impossible if the standalone
  XGBoost is the same model as the one inside the hybrid.
* For UNSW-NB15 the text calls the hybrid's 97.11% accuracy "the highest among all other models",
  but Table 10 lists CNN (98.22%), MLP (97.98%) and ANN (97.73%) above it.
* Table 10's logistic regression has precision 93.84 and recall 99.66 but "F1 99.66"; F1 must lie
  between the two (it would be 96.66).
* The split is given as 70:30 in the text but 80:20 in Algorithm 1 (Deviations, item 1), and the
  reported AUCs come from hard predictions (item 5).

### Limitations

* One dataset. DAPT2020 is small (81,443 clean flows) and has very few exfiltration flows.
* The split is random over flows, as in the paper. Flows from the same attack session can land in
  both train and test, so these results are still optimistic compared with detecting a new,
  unseen campaign.
* Some DAPT2020 labels may be wrong (Section 4.5); this affects every model's false positive rate.

### Possible next steps

* Validate on a second APT dataset (e.g., Unraveled 2023, or SCVIC-APT-2021 if access is available).
* Time-based evaluation: train on earlier days, test on later days.
* Improve the hybrid: soft voting, tuned model weights and decision threshold.
* SHAP explanations of the hybrid (the paper's Figures 25–26).

### Verification

An independent re-check of this project (recomputing every metric from the saved predictions,
re-deriving the dataset facts, reading the `combo` source and rerunning the pipeline) prompted the
corrections in this version: exact CSV float parsing, SMOTETomek's default two-class cleaning, and
more careful interpretation of the protocol comparison, the label-noise evidence, the seed
comparison and the stage-level ranking.

## 6. Project layout

```
.
├── README.md                 this file
├── requirements.txt          pinned package versions
├── APT_Detection_MidEval.pptx  mid-term evaluation slides
├── data/                     DAPT2020 CSVs go in data/raw/DAPT2020/ (downloaded, not committed)
├── src/
│   ├── config.py             paths and the paper's hyperparameters
│   ├── download_data.py      fetches DAPT2020 CSVs from Kaggle
│   ├── data_loader.py        loads/fixes/labels DAPT2020
│   ├── preprocessing.py      cleaning, SMOTETomek, Pearson + MI feature selection, both protocols
│   ├── models.py             the 10 classical baselines + HybridRFXGB (combo-exact and soft vote)
│   ├── deep_models.py        CNN, ANN, MLP (PyTorch)
│   ├── metrics.py            Accuracy/Precision/Recall/F1/TNR/FPR/FNR/AUC + multi-class metrics
│   ├── run_experiments.py    runs the three experiments
│   ├── error_analysis.py     breaks down the hybrid's false positives and misses
│   ├── seed_robustness.py    repeats the leakage-free experiment over 5 seeds
│   ├── leakage_analysis.py   separates class-ratio and leakage effects in the paper protocol
│   └── report.py             figures + results/summary.md
└── results/
    ├── summary.md            all result tables
    ├── run_log.txt           console log of the last full run
    ├── figures/              PNG figures
    └── <experiment>/         metrics.csv, split_info.json, predictions.npz
                              (+ error_analysis.md, seed_robustness.md, leakage_analysis.md)
```

## 7. How to run

```bash
python3 -m venv --system-site-packages .venv      # or a plain venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/download_data.py              # ~45 MB into data/raw/DAPT2020/
.venv/bin/python src/run_experiments.py            # all 3 experiments (add --no-deep to skip CNN/ANN/MLP)
.venv/bin/python src/error_analysis.py
.venv/bin/python src/seed_robustness.py             # ~5 min
.venv/bin/python src/leakage_analysis.py
.venv/bin/python src/report.py                     # figures + results/summary.md
```

Classical models are seeded (`SEED = 42` in `src/config.py`) and CSV parsing is exact, so with the
pinned versions in `requirements.txt` reruns reproduce their numbers; CNN/ANN/MLP on a GPU can vary slightly.

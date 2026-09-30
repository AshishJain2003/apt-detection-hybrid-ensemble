"""Paths and the hyperparameters taken from Saini et al. (2023)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "DAPT2020"
RESULTS_DIR = ROOT / "results"

SEED = 42

# Section 3.2 of the paper. Timestamp, Flow ID, Src IP, Dst IP and Src Port are
# dropped: they identify a particular host/session rather than its behaviour.
ID_COLUMNS = ["Flow ID", "Src IP", "Dst IP", "Src Port", "Timestamp"]
LABEL_COLUMNS = ["Activity", "Stage", "Label", "source_file"]
CATEGORICAL_COLUMNS = ["Protocol"]  # one-hot encoded ("creating dummy variables")

TEST_SIZE = 0.30              # 70:30 split (the test set of 15,583 rows in Fig. 11 = 30%)
CORRELATION_THRESHOLD = 0.90  # Pearson filter
MI_TOP_K = 30                 # mutual-information top-30 features

# Algorithm 1: 100 trees, learning rate 1.0, max depth 4.
N_ESTIMATORS = 100
XGB_LEARNING_RATE = 1.0
XGB_MAX_DEPTH = 4

# CNN / ANN / MLP were trained for 200 epochs in the paper.
DL_MAX_EPOCHS = 200
DL_PATIENCE = 15
DL_BATCH_SIZE = 512

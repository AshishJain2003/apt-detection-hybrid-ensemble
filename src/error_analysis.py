"""Where do the hybrid's binary errors come from? (leakage-free experiment)

The leakage-free test set is the untouched 30% of real flows, so each test row
maps back to an original DAPT2020 flow. We recover those rows by re-running the
same stratified split on row indices, then break the errors down by capture file,
APT activity, and whether the flow involves an external attacker host.

Usage:  python src/error_analysis.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data_loader import load_dapt2020
from preprocessing import clean

ATTACKER_IPS = {"206.207.50.50", "184.98.36.245"}  # external red-team hosts in DAPT2020
MODEL = "Proposed hybrid (RF+XGB)"


def main():
    out_dir = config.RESULTS_DIR / "binary_leakage_free"
    preds = np.load(out_dir / "predictions.npz")
    X, labels = clean(load_dapt2020())
    y = labels["Label"].to_numpy()
    _, test_idx = train_test_split(np.arange(len(y)), test_size=config.TEST_SIZE,
                                   stratify=y, random_state=config.SEED)
    test = labels.iloc[test_idx].reset_index(drop=True)
    assert np.array_equal(test["Label"].to_numpy(), preds["y_test"]), "split mismatch"

    test["pred"] = preds[f"{MODEL}|pred"]
    test["capture"] = test["source_file"].str.replace("enp0s3-", "").str.replace(".pcap_Flow.csv", "")
    test["attacker_host"] = test["Src IP"].isin(ATTACKER_IPS) | test["Dst IP"].isin(ATTACKER_IPS)
    test["error"] = np.select([(test["Label"] == 0) & (test["pred"] == 1),
                               (test["Label"] == 1) & (test["pred"] == 0)], ["FP", "FN"], "")

    benign = test[test["Label"] == 0]
    fp_by = (benign.groupby(["capture", "attacker_host"])
             .agg(benign_flows=("error", "size"), false_positives=("error", lambda s: (s == "FP").sum())))
    fp_by["FPR"] = (fp_by["false_positives"] / fp_by["benign_flows"]).map("{:.2%}".format)
    fn_by = (test[test["Label"] == 1].groupby("Activity")
             .agg(apt_flows=("error", "size"), missed=("error", lambda s: (s == "FN").sum())))
    fn_by["miss rate"] = (fn_by["missed"] / fn_by["apt_flows"]).map("{:.1%}".format)

    n_fp = int((test["error"] == "FP").sum())
    fp_attacker = int(((test["error"] == "FP") & test["attacker_host"]).sum())
    fpr_all = n_fp / len(benign)
    clean_benign = benign[~benign["attacker_host"]]
    fpr_excl = (clean_benign["error"] == "FP").mean()

    lines = [f"# Error analysis: {MODEL}, leakage-free binary test set", "",
             f"- False positives: {n_fp:,} of {len(benign):,} benign test flows (FPR {fpr_all:.2%}).",
             f"- {fp_attacker:,} of those false positives ({fp_attacker / max(n_fp, 1):.1%}) are 'benign'-labelled "
             f"flows to/from an external attacker host ({', '.join(sorted(ATTACKER_IPS))}).",
             f"- FPR on benign flows that do not involve an attacker host: {fpr_excl:.2%}.", "",
             "## False positives by capture file", "", fp_by.to_markdown(), "",
             "## Missed APT flows by activity", "", fn_by.to_markdown(), ""]
    (out_dir / "error_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

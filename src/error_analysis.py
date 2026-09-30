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
ATTACK_FREE = {"monday", "monday-pvt"}             # captures in which every flow is labelled normal
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
    fp = benign["error"] == "FP"
    fp_by = (benign.groupby(["capture", "attacker_host"])
             .agg(benign_flows=("error", "size"), false_positives=("error", lambda s: (s == "FP").sum())))
    fp_by["FPR"] = (fp_by["false_positives"] / fp_by["benign_flows"]).map("{:.2%}".format)

    apt = test[test["Label"] == 1]
    fn_by = (apt.groupby("Activity")
             .agg(apt_flows=("error", "size"), missed=("error", lambda s: (s == "FN").sum()))
             .sort_values("missed", ascending=False))
    n_fn = int(fn_by["missed"].sum())
    fn_by["share of all misses"] = (fn_by["missed"] / n_fn).map("{:.1%}".format)
    fn_by["miss rate"] = (fn_by["missed"] / fn_by["apt_flows"]).map("{:.1%}".format)

    n_fp = int(fp.sum())
    fp_attacker = int((fp & benign["attacker_host"]).sum())
    fp_attack_free = int((fp & benign["capture"].isin(ATTACK_FREE)).sum())
    attacker_on_attack_free = int((test["capture"].isin(ATTACK_FREE) & test["attacker_host"]).sum())
    no_attacker = benign[~benign["attacker_host"]]
    fpr_excl = (no_attacker["error"] == "FP").mean()
    top = fn_by.index[0]

    lines = [f"# Error analysis: {MODEL}, leakage-free binary test set", "",
             "## False positives", "",
             f"- {n_fp:,} false positives out of {len(benign):,} benign test flows (FPR {n_fp / len(benign):.2%}).",
             f"- {fp_attacker:,} of them ({fp_attacker / n_fp:.1%}) are benign-labelled flows to or from an external "
             f"attacker host ({', '.join(sorted(ATTACKER_IPS))}). This may indicate label noise, but it is not proof: "
             f"the attacker hosts also appear in {attacker_on_attack_free:,} test flows of the attack-free Monday captures.",
             f"- {fp_attack_free:,} of them ({fp_attack_free / n_fp:.1%}) fall in the Monday captures, where no flow is labelled "
             "as an attack. These are genuine false alarms on normal traffic.",
             f"- FPR on benign flows that do not involve an attacker host: {fpr_excl:.2%}.", "",
             "### By capture file", "", fp_by.to_markdown(), "",
             "## Missed APT flows (false negatives)", "",
             f"- {n_fn:,} APT test flows are missed. By count, most are {top} "
             f"({int(fn_by.loc[top, 'missed']):,} of {n_fn:,}).",
             "- The rare, low-volume activities have the highest miss *rates* but contribute few misses in absolute terms.", "",
             fn_by.to_markdown(), ""]
    (out_dir / "error_analysis.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

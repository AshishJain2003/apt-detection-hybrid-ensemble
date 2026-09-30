"""Load the DAPT2020 flow CSVs into a single, correctly-labelled DataFrame.

DAPT2020 (Myneni et al., 2020) is an APT-specific benchmark: every flow is
labelled with the malicious *Activity* and the APT *Stage* it belongs to
(Benign, Reconnaissance, Establish Foothold, Lateral Movement, Data Exfiltration).
Its 76 CICFlowMeter features are the same family used by CIC-IDS2017 and
CSE-CIC-IDS2018, the main datasets of Saini et al. (2023).

Quirks handled here:
  * enp0s3-pvt-thursday.pcap_Flow.csv has NO header row. A naive
    pd.read_csv() treats its first flow as the header and mis-aligns the file,
    silently losing 2,314 of the 2,451 Lateral Movement flows.
  * Stage/Activity labels use inconsistent casing ("BENIGN", "Benign", "Normal").
"""
from pathlib import Path

import pandas as pd

from config import RAW_DIR

STAGE_ORDER = ["Benign", "Reconnaissance", "Establish Foothold",
               "Lateral Movement", "Data Exfiltration"]

_STAGE_MAP = {"benign": "Benign", "reconnaissance": "Reconnaissance",
              "establish foothold": "Establish Foothold",
              "lateral movement": "Lateral Movement",
              "data exfiltration": "Data Exfiltration"}
_ACTIVITY_BENIGN = {"benign", "normal"}


def _header(files):
    for f in files:
        with open(f) as fh:
            first = fh.readline()
        if first.startswith("Flow ID"):
            return [c.strip() for c in first.rstrip("\n").split(",")]
    raise RuntimeError("No DAPT2020 file with a header row was found.")


def load_dapt2020(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("*.pcap_Flow.csv"))
    if not files:
        raise FileNotFoundError(
            f"No DAPT2020 CSVs in {raw_dir}. Run: python src/download_data.py")
    columns = _header(files)

    frames = []
    for f in files:
        with open(f) as fh:
            has_header = fh.readline().startswith("Flow ID")
        df = pd.read_csv(f, header=0 if has_header else None,
                         names=None if has_header else columns, low_memory=False)
        df.columns = df.columns.str.strip()
        df["source_file"] = f.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    stage = df["Stage"].astype(str).str.strip().str.lower()
    unknown = set(stage) - set(_STAGE_MAP)
    if unknown:
        raise ValueError(f"Unexpected Stage labels: {unknown}")
    df["Stage"] = stage.map(_STAGE_MAP)

    activity = df["Activity"].astype(str).str.strip()
    df["Activity"] = activity.where(~activity.str.lower().isin(_ACTIVITY_BENIGN), "Benign")

    # Binary target used by the paper: normal = 0, any APT activity = 1.
    df["Label"] = (df["Stage"] != "Benign").astype(int)
    return df


if __name__ == "__main__":
    d = load_dapt2020()
    print(d.shape)
    print(d["Stage"].value_counts().reindex(STAGE_ORDER))
    print(d["Activity"].value_counts())

"""Download the DAPT2020 labelled flow CSVs (~45 MB) from the authors' Kaggle dataset.

Source: https://www.kaggle.com/datasets/sowmyamyneni/dapt2020  (Sowmya Myneni, ASU)
Only the csv/ folder is fetched; the full dataset is 9.7 GB because of raw PCAPs/logs.
Public files download anonymously; if Kaggle asks for credentials, put your
kaggle.json API token in ~/.kaggle/.

Usage:  python src/download_data.py
"""
import shutil

import kagglehub

from config import RAW_DIR

DATASET = "sowmyamyneni/dapt2020"
DAYS = ["monday-pvt", "monday", "public-thursday", "public-tuesday", "public-wednesday",
        "pvt-thursday", "pvt-tuesday", "pvt-wednesday", "tcpdump-friday", "tcpdump-pvt-friday"]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    files = {"csv/README.md": "CSV_README.md"}
    files.update({f"csv/enp0s3-{d}.pcap_Flow.csv": f"enp0s3-{d}.pcap_Flow.csv" for d in DAYS})
    for remote, local in files.items():
        dst = RAW_DIR / local
        if dst.exists():
            print("exists  ", dst.name)
            continue
        shutil.copy(kagglehub.dataset_download(DATASET, path=remote), dst)
        print("fetched ", dst.name)


if __name__ == "__main__":
    main()

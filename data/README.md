# Data

The DAPT2020 flow CSVs are not stored in this repository. Download them (~45 MB) with:

```bash
python src/download_data.py
```

This fetches the 10 `*.pcap_Flow.csv` files and the dataset's feature description from the
dataset author's Kaggle page (https://www.kaggle.com/datasets/sowmyamyneni/dapt2020) into
`data/raw/DAPT2020/`.

Dataset reference: S. Myneni et al., *DAPT 2020 – Constructing a Benchmark Dataset for Advanced
Persistent Threats*, Deployable Machine Learning for Security Defense (MLHat 2020), Springer CCIS,
2020. doi:10.1007/978-3-030-59621-7_8

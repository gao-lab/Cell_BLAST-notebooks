#!/usr/bin/env python

import os
import numpy as np
import pandas as pd

dataset_dict = {
    item: [
        file for file in os.listdir(item)
        if file.endswith(".pdf") and file != "peek.pdf"
    ] for item in os.listdir(".") if item not in (
        "__pycache__", ".ipynb_checkpoints"
    ) and os.path.isdir(item)
}

used_columns = (
    "dataset_name", "organism", "organ", "lifestage", "platform",
    "cell number", "publication", "remark"
)
single = pd.read_csv(
    "../../Datasets/ACA_datasets.csv",
    comment="#", skip_blank_lines=True
).loc[:, used_columns]
additional = pd.read_csv(
    "../../Datasets/additional_datasets.csv",
    comment="#", skip_blank_lines=True
).loc[:, used_columns]
single = pd.concat([single, additional], axis=0, ignore_index=True)
aligned = pd.read_csv(
    "../../Datasets/aligned_datasets.csv",
    comment="#", skip_blank_lines=True
).loc[:, used_columns]

for idx, row in aligned.iterrows():
    aligned.loc[idx, "cell number"] = single.loc[np.in1d(
        single["dataset_name"], row["remark"].split(", ")
    ), "cell number"].sum()

combined = pd.concat([single, aligned], axis=0, ignore_index=True)
combined = combined.loc[np.in1d(
    combined["dataset_name"], list(dataset_dict.keys())
), :]
combined["cell number"] = combined["cell number"].astype(np.int)

combined["self-projection coverage"] = np.nan
combined["self-projection accuracy"] = np.nan
for idx, row in combined.iterrows():
    spf_path = os.path.join(row["dataset_name"], "self_projection.txt")
    if not os.path.exists(spf_path):
        print("Missing: " + spf_path)
    else:
        with open(spf_path, "r") as spf:
            lines = spf.readlines()
            k1, v1 = lines[0].split()
            k2, v2 = lines[1].split()
            assert k1 == "coverage" and k2 == "accuracy"
            v1, v2 = float(v1.strip()), float(v2.strip())
            combined.loc[idx, "self-projection coverage"] = v1
            combined.loc[idx, "self-projection accuracy"] = v2

combined["visualization"] = np.vectorize(
    lambda x: ", ".join(dataset_dict[x])
)(combined["dataset_name"])

combined.to_csv("./datasets_meta.csv", index=False)
combined.to_json("./datasets_meta.json", orient="records", double_precision=3)


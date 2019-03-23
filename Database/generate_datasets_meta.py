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
single = pd.read_csv("../../Datasets/ACA_datasets.csv", comment="#", skip_blank_lines=True).loc[:, used_columns]
additional = pd.read_csv("../../Datasets/additional_datasets.csv", comment="#", skip_blank_lines=True).loc[:, used_columns]
aligned = pd.read_csv("../../Datasets/aligned_datasets.csv", comment="#", skip_blank_lines=True).loc[:, used_columns]

combined = pd.concat([single, additional, aligned], axis=0)
combined = combined.loc[np.in1d(combined["dataset_name"], list(dataset_dict.keys())), :]
combined["visualization"] = np.vectorize(lambda x: ", ".join(dataset_dict[x]))(combined["dataset_name"])

combined.to_csv("./datasets_meta.csv", index=False)
combined.to_json("./datasets_meta.json", orient="records")


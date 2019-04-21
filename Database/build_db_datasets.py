#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
import mysql.connector
from utils import nan_safe


def generate_datasets_meta():
    dataset_dict = {
        item: [
            file for file in os.listdir(item)
            if file.endswith(".pdf") and file != "peek.pdf"
        ] for item in os.listdir(".") if item not in (
            "__pycache__", ".ipynb_checkpoints"
        ) and os.path.isdir(item)
    }

    used_columns = (
        "dataset_name", "organism", "organ", "platform",
        "cell_number", "publication", "pmid", "remark"
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
        aligned.loc[idx, "cell_number"] = single.loc[np.in1d(
            single["dataset_name"], row["remark"].split(", ")
        ), "cell_number"].sum()

    combined = pd.concat([single, aligned], axis=0, ignore_index=True)
    combined["display"] = np.in1d(
        combined["dataset_name"], list(dataset_dict.keys()))
    # combined = combined.loc[np.in1d(
    #     combined["dataset_name"], list(dataset_dict.keys())
    # ), :]
    # combined["cell_number"] = combined["cell_number"].astype(np.int)

    combined["self-projection coverage"] = np.nan
    combined["self-projection accuracy"] = np.nan
    for idx, row in combined.iterrows():
        spf_path = os.path.join(row["dataset_name"], "self_projection.txt")
        if not os.path.exists(spf_path):
            if row["dataset_name"] in dataset_dict:
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

    combined["visualization"] = [
        (", ".join(dataset_dict[item]) if item in dataset_dict else np.nan)
        for item in combined["dataset_name"]
    ]

    # combined.to_csv("./datasets_meta.csv", index=False)
    # combined.to_json("./datasets_meta.json", orient="records", double_precision=3)
    return combined


def create_table(cnx, cursor):
    cursor.execute("DROP TABLE IF EXISTS `datasets`;")
    cursor.execute(
        "CREATE TABLE `datasets` ("
        "  `dataset_name` CHAR(50) NOT NULL UNIQUE,"
        "  `organism` char(50) NOT NULL,"
        "  `organ` char(100) NOT NULL,"
        "  `platform` char(50),"
        "  `cell_number` INT CHECK(`cell_number` > 0),"
        "  `publication` VARCHAR(300),"
        "  `pmid` CHAR(8),"
        "  `remark` VARCHAR(200),"
        "  `self-projection coverage` FLOAT CHECK(`self-projection coverage` BETWEEN 0 AND 1),"
        "  `self-projection accuracy` FLOAT CHECK(`self-projection accuracy` BETWEEN 0 AND 1),"
        "  `visualization` VARCHAR(200),"
        "  `display` BOOL NOT NULL,"
        "  PRIMARY KEY USING HASH(`dataset_name`)"
        ");"
    )


def insert_data(cnx, cursor, data):
    insert_sql = (
        "INSERT INTO `datasets` ("
        "  `dataset_name`, `organism`, `organ`, `platform`,"
        "  `cell_number`, `publication`, `pmid`, `remark`,"
        "  `self-projection coverage`, `self-projection accuracy`,"
        "  `visualization`, `display`"
        ") VALUES ("
        "  %s, %s, %s, %s,"
        "  %s, %s, %s, %s,"
        "  %s, %s, %s, %s"
        ");"
    )
    for idx, row in data.iterrows():
        cursor.execute(insert_sql, (
            nan_safe(row["dataset_name"]), nan_safe(row["organism"]),
            nan_safe(row["organ"]), nan_safe(row["platform"]),
            nan_safe(row["cell_number"], int), nan_safe(row["publication"]),
            nan_safe(row["pmid"], lambda x: str(int(x))), nan_safe(row["remark"]),
            nan_safe(row["self-projection coverage"], lambda x: float(np.round(x, 3))),
            nan_safe(row["self-projection accuracy"], lambda x: float(np.round(x, 3))),
            nan_safe(row["visualization"]), nan_safe(row["display"])
        ))


def main():
    cnx = mysql.connector.connect(
        user=input("Please enter username: "), password=input("Please enter password: "),
        host="127.0.0.1", database="aca"
    )
    cursor = cnx.cursor()
    create_table(cnx, cursor)
    insert_data(cnx, cursor, generate_datasets_meta())
    cnx.commit()
    cursor.close()
    cnx.close()


if __name__ == "__main__":
    main()

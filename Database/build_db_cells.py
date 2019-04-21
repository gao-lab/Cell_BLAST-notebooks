#!/usr/bin/env python

import numpy as np
import pandas as pd
import mysql.connector
import Cell_BLAST as cb
from utils import nan_safe


def get_datasets(cnx):
    cursor = cnx.cursor(buffered=True)
    query = "SELECT `dataset_name` FROM `datasets` WHERE `display` = 1;"
    cursor.execute(query)
    for item in cursor:
        yield item[0]
    cursor.close()


def get_data(name):
    obs = {
        key: val for key, val in cb.data.read_hybrid_path(
            "%s/ref.h5//obs" % name
        ).items() if not (
            key.startswith("latent_") or
            key in ("organism", "organ", "platform")
        )
    }
    obs["cid"] = cb.data.read_hybrid_path("%s/ref.h5//obs_names" % name)
    obs = pd.DataFrame(obs)
    if "dataset_name" not in obs.columns:
        obs["dataset_name"] = name
    return obs


def native_type(x):
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def create_table(cnx, name):
    data = get_data(name)
    cursor = cnx.cursor()
    drop_sql = "DROP TABLE IF EXISTS `%s`;" % name
    cursor.execute(drop_sql)
    create_sql = ["CREATE TABLE `%s` (" % name]
    for column in data.columns:
        if np.issubdtype(data[column].dtype.type, np.object_):
            if column in ("cid", "dataset_name"):
                dtype = "CHAR(50)"
            else:
                dtype = "CHAR(%d)" % np.vectorize(len)(data[column]).max()
        elif np.issubdtype(data[column].dtype.type, np.floating):
            dtype = "FLOAT"
        elif np.issubdtype(data[column].dtype.type, np.integer):
            dtype = "INT"
        else:
            raise Exception("Unexpected dtype!")
        if column == "cid":
            options = " NOT NULL UNIQUE"
        elif column == "dataset_name":
            options = " NOT NULL"
        else:
            options = ""
        create_sql.append("`%s` %s%s, " % (column, dtype, options))
    create_sql.append("PRIMARY KEY USING HASH(`cid`), ")
    create_sql.append("FOREIGN KEY(`dataset_name`) REFERENCES `datasets`(`dataset_name`));")
    create_sql = "".join(create_sql)
    cursor.execute(create_sql)

    columns = ", ".join(["`%s`" % item for item in data.columns])
    values = ", ".join(["%s"] * data.shape[1])
    insert_sql = "INSERT INTO `%s` (%s) VALUES (%s);" % (name, columns, values)
    for idx, row in data.iterrows():
        cursor.execute(insert_sql, tuple(
            nan_safe(row[column], native_type) for column in data.columns
        ))
    cursor.close()


def main():
    cnx = mysql.connector.connect(
        user=input("Please enter username: "), password=input("Please enter password: "),
        host="127.0.0.1", database="aca"
    )
    for dataset in get_datasets(cnx):
        print("Working on %s..." % dataset)
        create_table(cnx, dataset)
        cnx.commit()
    cnx.close()


if __name__ == "__main__":
    main()

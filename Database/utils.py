import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def mkdir(fn):
    def wrapped(*args):
        if not os.path.exists(args[1]):
            os.makedirs(args[1])
        return fn(*args)
    return wrapped


@mkdir
def peek(dataset, dataset_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    _ = sns.distplot(dataset.exprs.sum(axis=1), axlabel='nUMI', ax=ax1)
    _ = sns.distplot((dataset.exprs > 0).sum(axis=1), axlabel="nGene", ax=ax2)
    plt.tight_layout()
    fig.savefig("%s/peek.pdf" % dataset_name, bbox_inches="tight")


@mkdir
def self_projection(blast, dataset_name):
    hits = blast.query(blast.ref).reconcile_models().filter("pval", 0.05)
    for i in range(len(hits)):  # Remove self-hit (leave one out cv)
        mask = hits.hits[i] == i
        hits.hits[i] = hits.hits[i][~mask]
        hits.dist[i] = hits.dist[i][:, ~mask]
        hits.pval[i] = hits.pval[i][:, ~mask]
    pred = hits.annotate("cell_ontology_class").values.ravel()
    with open("%s/self_projection.txt" % dataset_name, "w") as f:
        covered = ~np.in1d(pred, ["ambiguous", "rejected"])
        print("Coverage = %.4f" % (covered.sum() / covered.size))
        f.write("coverage\t%f\n" % (covered.sum() / covered.size))
        correctness = pred[covered] == blast.ref.obs["cell_ontology_class"][covered]
        print("Accuracy = %.4f" % (correctness.sum() / correctness.size))
        f.write("accuracy\t%f\n" % (correctness.sum() / correctness.size))

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pytorch_lightning.metrics.functional as M
import seaborn as sns

ATTRIBUTES = [
    "external",
    "entrypoint",
    "native",
    "public",
    "static",
    "codesize",
    "api",
    "user",
]


def plot_curve(x, y, curve_type):
    """
    Plots ROC or PRC
    Inspired from https://plotly.com/python/roc-and-pr-curves/
    :param x: The x co-ordinates
    :param y: The y co-ordinates
    :param curve_type: one of 'roc' or 'prc'
    :return: Plotly figure
    """
    auc = M.classification.auc(x, y)
    x, y = x.numpy(), y.numpy()
    if curve_type == "roc":
        title = f"ROC, AUC = {auc}"
        labels = dict(x="FPR", y="TPR")
    elif curve_type == "prc":
        title = f"PRC, mAP = {auc}"
        labels = dict(x="Recall", y="Precision")
    else:
        raise ValueError(
            f"Invalid curve type - {curve_type}. Must be one of 'roc' or 'prc'."
        )
    fig = px.area(x=x, y=y, labels=labels, title=title)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    return fig


def plot_confusion_matrix(
    cf,
    group_names=None,
    categories="auto",
    count=True,
    percent=True,
    cbar=True,
    xyticks=True,
    xyplotlabels=True,
    sum_stats=True,
    fig_size=None,
    cmap="Blues",
    title=None,
):
    plt.clf()
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ["" for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = [
            "{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)
        ]
    else:
        group_percentages = blanks

    box_labels = [
        f"{v1}{v2}{v3}".strip()
        for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)
    ]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        # Accuracy is sum of diagonal divided by total observations
        accuracy = np.trace(cf) / float(np.sum(cf))

        # if it is a binary confusion matrix, show some more stats
        if len(cf) == 2:
            # Metrics for Binary Confusion Matrices
            precision = cf[1, 1] / sum(cf[:, 1])
            recall = cf[1, 1] / sum(cf[1, :])
            f1_score = 2 * precision * recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.4f}\nPrecision={:0.4f}\nRecall={:0.4f}\nF1 Score={:0.4f}".format(
                accuracy, precision, recall, f1_score
            )
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if fig_size is None:
        # Get default figure size if not set
        fig_size = plt.rcParams.get("figure.figsize")

    if not xyticks:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=fig_size)
    sns.heatmap(
        cf,
        annot=box_labels,
        fmt="",
        cmap=cmap,
        cbar=cbar,
        xticklabels=categories,
        yticklabels=categories,
    )

    if xyplotlabels:
        plt.ylabel("True label")
        plt.xlabel("Predicted label" + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig("CM.png")

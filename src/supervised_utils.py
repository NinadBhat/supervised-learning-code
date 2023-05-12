import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import StratifiedKFold
from constants import FONT_SIZE, TICK_FONT_SIZE


def get_data():
    df = pd.read_csv("../data/al_data.csv")
    df = df[df["class"] != "outlier"]
    df.drop(["Pb"], axis=1, inplace=True)

    df_ys = shuffle(
        df.dropna(subset=["Yield Strength (MPa)"]), random_state=30
    ).reset_index()

    df_ts = shuffle(
        df.dropna(subset=["Tensile Strength (MPa)"]), random_state=90
    ).reset_index()

    df_elong = shuffle(
        df.dropna(subset=["Elongation (%)"]), random_state=20
    ).reset_index()

    return [df_ys, df_ts, df_elong]


def get_target_classes(target):

    bins = np.linspace(target.min() * 0.99, target.max() * 1.1, 6)

    def find_class_of_traget(value):
        return int((value - bins[0]) / (bins[2] - bins[1]))

    return np.vectorize(find_class_of_traget)(target)


def get_straifying_split(feature, target, random_state=0):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    classes = get_target_classes(target)
    return skf.split(feature, classes)


def save_learning_curve(
    train_sizes,
    train_scores,
    test_scores,
    y_range,
    property_name,
    al_class,
    unit,
    supervised=False,
):
    if supervised:
        location = f"../images/{property_name.replace(' ', '_').lower()}/learning_curve/supervised/class_{al_class}.png"
    else:
        location = f"../images/{property_name.replace(' ', '_').lower()}/learning_curve/unsupervised/class_{al_class}.png"

    plt.figure(figsize=(5, 5))

    train_scores = [value for value in train_scores]
    test_scores = [value for value in test_scores]
    plt.plot(train_sizes, train_scores, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores, "o-", color="g", label="Cross-validation score")
    plt.legend(loc="upper right")
    plt.xticks(fontsize=TICK_FONT_SIZE, family="Arial")
    plt.yticks(fontsize=TICK_FONT_SIZE, family="Arial")

    plt.xlabel("Training examples", fontsize=FONT_SIZE, family="Arial")
    plt.ylabel(f"Mean absolute error ({unit})", fontsize=FONT_SIZE, family="Arial")
    plt.ylim(y_range)
    if al_class == "all instances":
        plt.title("All instances: " + property_name, fontsize=FONT_SIZE, family="Arial")
    else:
        plt.title(
            f"Class {al_class}: " + property_name, fontsize=FONT_SIZE, family="Arial"
        )
    plt.savefig(location, bbox_inches="tight", dpi=330)
    plt.close()


def save_feature_importance_profiles(
    feature, importance, property_name, x_lim, al_class
):
    location = f"../images/{property_name.replace(' ', '_').lower()}/feature_importance_profile/class_{al_class}.png"

    plt.figure(figsize=(4, 10))
    plt.rc("axes", labelsize=16)
    plt.rc("ytick", labelsize=20)
    a = sns.barplot(y=feature, x=importance, color="b")
    a.set_ylabel("Features", fontsize=FONT_SIZE, family="Arial")
    a.set_xlabel("Importance", fontsize=FONT_SIZE, family="Arial")
    plt.xlim(x_lim)

    plt.xticks(fontsize=TICK_FONT_SIZE, family="Arial")
    plt.yticks(fontsize=18, family="Arial")
    # f"Class {al_class}: Tensile strength"
    if al_class == "all instances":
        plt.title("All instances: " + property_name, fontsize=FONT_SIZE, family="Arial")
    else:
        plt.title(
            f"Class {al_class}: " + property_name, fontsize=FONT_SIZE, family="Arial"
        )
    plt.savefig(location, bbox_inches="tight", dpi=330)
    plt.close()


def save_rfecv(rfe_mean_test_scores, property_name, al_class, unit):
    if al_class == "all instances":
        title = "All instances: " + property_name
        location_35 = (
            f"../images/{property_name.replace(' ', '_').lower()}/rfe/35/all.png"
        )

    else:
        title = f"Class {al_class}: " + property_name
        location_35 = f"../images/{property_name.replace(' ', '_').lower()}/rfe/35/class_{al_class}.png"

    plt.figure(figsize=(5, 5))
    plt.xlim([0, 35])

    sns.set_theme()
    plt.xticks(fontsize=TICK_FONT_SIZE, family="Arial")
    plt.yticks(fontsize=TICK_FONT_SIZE, family="Arial")
    plt.xlabel("Number of features selected", fontsize=FONT_SIZE, family="Arial")
    plt.ylabel(
        f"Cross validation error ({unit})",
        fontsize=FONT_SIZE,
        family="Arial",
    )
    plt.plot(
        range(1, len(rfe_mean_test_scores) + 1, 2),
        -1 * rfe_mean_test_scores[::2],
        "o-",
    )
    min_rfe = min(-1 * rfe_mean_test_scores[::2])
    plt.title(title, fontsize=FONT_SIZE, family="Arial")
    plt.savefig(location_35, bbox_inches="tight", dpi=330)

    plt.close()

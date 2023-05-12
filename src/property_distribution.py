#!/usr/bin/env python


import pandas as pd
import numpy as np


# preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

import seaborn as sns

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from supervised_utils import (
    get_data,
    get_target_classes,
    save_feature_importance_profiles,
    save_learning_curve,
    save_rfecv,
)
from constants import (
    ALLOYING_ELEMENTS,
    PROCESSING_COLUMNS,
    TARGET_COLUMNS,
)



# Ploting Libraries

import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser(description="Force train models")
parser.add_argument("-f", action="store_true")
parser.add_argument("-i", type=int, help="Property index")
args = parser.parse_args()

font = {"family": "arial", "size": 18}

sns.set_theme()
plt.rc("font", **font)
import os

os.environ["PATH"] = (
    os.environ["PATH"] + ":/Users/ninadbhat/opt/anaconda3/envs/ap-venv/bin/:"
)


# # CONSTANTS

PROPERTIES = ["Yield strength", "Tensile strength", "Elongation"]
selected_property_index = args.i
PROPERTY_NAME = PROPERTIES[selected_property_index]
property_column = TARGET_COLUMNS[selected_property_index]
df_all = get_data()

df_interested = df_all[selected_property_index]

numeric_preprocessor = Pipeline(steps=[("normalize", preprocessing.MinMaxScaler())])

categorical_preprocessor = Pipeline(
    steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
)

preprocessor = ColumnTransformer(
    [
        ("categorical", categorical_preprocessor, PROCESSING_COLUMNS),
        ("numerical", numeric_preprocessor, ALLOYING_ELEMENTS),
    ]
)

preprocessor.fit(df_interested)


columns_names = np.concatenate(
    (
        preprocessor.transformers_[0][1]
        .named_steps["onehot"]
        .get_feature_names_out(["Processing"]),
        ALLOYING_ELEMENTS,
    )
)


def get_domain_class_from_processing(process):
    if process in ["No Processing"]:
        return "A"
    elif process in ["Strain Harderned (Hard)", "Strain hardened"]:
        return "B"
    else:
        return "C"


df_interested["domain_class"] = df_interested["Processing"].apply(
    get_domain_class_from_processing
)
# Preprocessing into data

feature_matrix = pd.DataFrame(
    preprocessor.transform(df_interested), columns=columns_names
)
target_martrix = df_interested[["domain_class", "class", property_column]]


X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    feature_matrix,
    target_martrix,
    test_size=0.2,
    random_state=30,
)
# count of class in traning set
train_count = y_train_all["domain_class"].value_counts()
# count of class in test set
test_count = y_test_all["domain_class"].value_counts()

print(test_count)
print(train_count)


y_train_all["Type"] = "Train"
y_test_all["Type"] = "Test"
y_all = pd.concat([y_train_all, y_test_all])
y_all = y_all.sort_values(by=["domain_class"])

# plot histogram of property distribution
plt.figure(figsize=(10, 10))
sns.set(font_scale=2.5)
sns.histplot(
    y_all,
    x="domain_class",
    hue="Type",
    multiple="stack",
    shrink=0.8,
)
plt.xlabel("Domain knowledge based partitions")
# save image
plt.savefig(
    f"../images/supervised_property_distribution_{PROPERTY_NAME}.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
)


# plot histogram of property distribution class
plt.figure(figsize=(10, 10))
# sort y_all by class
y_all = y_all.sort_values(by=["class"])
sns.histplot(
    y_all,
    x="class",
    hue="Type",
    multiple="stack",
    shrink=0.8,
)
plt.xlabel("Unsupervised data-driven class")
# save image
plt.savefig(
    f"../images/unsupervised_property_distribution_class_{PROPERTY_NAME}.png",
    bbox_inches="tight",
    pad_inches=0,
    dpi=300,
)

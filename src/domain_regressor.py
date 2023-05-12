#!/usr/bin/env python


import pandas as pd
import numpy as np


# preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

import seaborn as sns

# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from supervised_utils import (
    get_data,
    save_learning_curve,
)
from constants import (
    ALLOYING_ELEMENTS,
    PROCESSING_COLUMNS,
    TARGET_COLUMNS,
    N_JOBS,
    N_ITER,
    FONT_SIZE,
    RANGE_DICTIONARY,
)

# Regression Models Training
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# Model Evaluation
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Save Model
from joblib import dump, load


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


def get_bin_function(partitions):
    def get_property_bin(property):
        for i, partition in enumerate(partitions):
            if property < partition:
                return str(i + 1)
        return str(len(partitions) + 1)

    return get_property_bin


partitions = [(250, 500), (300, 550), (10, 25)]


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

distributions = dict(
    n_estimators=randint(100, 2000),
    max_depth=randint(5, 50),
    min_samples_split=randint(2, 12),
    min_samples_leaf=randint(1, 10),
    max_features=["sqrt", "log2", None],
)

rf_class = RandomForestClassifier(random_state=5)
rf = RandomForestRegressor(random_state=5)

class_cv = RandomizedSearchCV(
    estimator=rf_class,
    param_distributions=distributions,
    n_iter=N_ITER,
    n_jobs=N_JOBS,
    random_state=12,
)


def random_reg(random_state):
    return RandomizedSearchCV(
        estimator=rf,
        param_distributions=distributions,
        n_iter=N_ITER,
        n_jobs=N_JOBS,
        random_state=random_state,
    )


# Domain Knowledge class


retrain = ""
if not args.f:
    retrain = input("Retrain domain classifier [y]?")

if retrain == "y" or args.f:
    class_cv.fit(X_train_all, y_train_all["domain_class"])
    domain_classifier = class_cv.best_estimator_
    dump(
        domain_classifier,
        f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/domain_classifier.joblib",
    )

else:
    domain_classifier = load(
        f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/domain_classifier.joblib"
    )


print(
    confusion_matrix(
        domain_classifier.predict(X_test_all),
        y_test_all["domain_class"],
        normalize="true",
    )
)
# Plot heat map of confusion matrix
plt.figure(figsize=(10, 10))
confusion_materix_heatmap = sns.heatmap(
    confusion_matrix(
        domain_classifier.predict(X_test_all),
        y_test_all["domain_class"],
        normalize="true",
    ),
    annot=True,
    fmt=".2f",
    square=True,
    cmap="Blues",
    xticklabels=[f"Class {i}" for i in ["A", "B", "C"]],
    yticklabels=[f"Class {i}" for i in ["A", "B", "C"]],
)
plt.xlabel("Predicted Label", fontsize=FONT_SIZE * 1.5)
plt.ylabel("Actual Label", fontsize=FONT_SIZE * 1.5)


plt.xticks(fontsize=FONT_SIZE * 1.25)
plt.yticks(fontsize=FONT_SIZE * 1.25)

plt.savefig(
    f"../images/{PROPERTY_NAME.replace(' ', '_').lower()}/domain_confusion_matrix.png",
    bbox_inches="tight",
    dpi=300,
)
print(
    classification_report(
        domain_classifier.predict(X_test_all), y_test_all["domain_class"]
    )
)
rf_supervised_regressors = []
X_train_supervised_instances = []
y_train_supervised_instances = []

if not args.f:
    retrain = input("Retrain supervised regressors [y]?")
if retrain == "y" or args.f:
    for i in range(1, 4):
        X_train = X_train_all[y_train_all["domain_class"] == str(i)]
        X_train_supervised_instances.append(X_train)
        y_train = y_train_all[y_train_all["domain_class"] == str(i)][property_column]
        y_train_supervised_instances.append(y_train)
        reg_cv = random_reg(i + selected_property_index * 10)
        reg_cv.fit(X_train, y_train)
        rf_supervised_regressors.append(reg_cv.best_estimator_)

    dump(
        rf_supervised_regressors,
        f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/supervised_regressors.joblib",
    )
else:
    rf_supervised_regressors = load(
        f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/supervised_regressors.joblib"
    )


def predict_domian(domain_classifier, regressors):
    def predict_value(value):
        domain_class = int(domain_classifier.predict([value])[0])
        return regressors[domain_class - 1].predict([value])[0]

    return predict_value


predict_domian_value = predict_domian(domain_classifier, rf_supervised_regressors)

# supervised_predicted = X_test_all.apply(predict_domian_value, axis=1)

# print(mean_absolute_error(supervised_predicted, y_test_all[property_column]))


params = []
for i in range(3):

    params.append(list(rf_supervised_regressors[i].get_params().values()))
df_model_parameters = pd.DataFrame(
    params,
    columns=rf_supervised_regressors[0].get_params().keys(),
    index=[i for i in range(1, 4)],
)[
    [
        "max_depth",
        "max_features",
        "min_samples_leaf",
        "min_samples_split",
        "n_estimators",
    ]
]


df_model_parameters.to_csv(
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/model_params/supervised_model_parameter.csv"
)

if not args.f:
    retrain = input("Recalculate accuracy [y]?")
if retrain == "y" or args.f:
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    test_scores_r2 = []
    test_scores_mae = []
    test_scores_rmse = []
    cross_validation_score_r2 = []
    cross_validation_score_mae = []
    cross_validation_score_rmse = []
    for i in range(3):
        train_index = y_train_all["domain_class"] == str(i + 1)
        test_index = y_test_all["domain_class"] == str(i + 1)

        X_train = X_train_all[train_index]
        y_train = y_train_all[train_index][property_column]
        X_test = X_test_all[test_index]
        y_test = y_test_all[test_index][property_column]

        test_scores_r2.append(
            r2_score(
                rf_supervised_regressors[i].predict(X_test),
                y_test,
            )
        )
        test_scores_mae.append(
            mean_absolute_error(
                rf_supervised_regressors[i].predict(X_test),
                y_test,
            )
        )
        test_scores_rmse.append(
            mean_squared_error(
                rf_supervised_regressors[i].predict(X_test),
                y_test,
                squared=False,
            )
        )

        scores = cross_validate(
            rf_supervised_regressors[i],
            X_train,
            y_train,
            scoring=scoring,
            cv=5,
            n_jobs=N_JOBS,
        )
        cross_validation_score_r2.append(scores["test_r2"])
        cross_validation_score_mae.append(scores["test_mae"])
        cross_validation_score_rmse.append(scores["test_rmse"])
    dump(
        [
            cross_validation_score_mae,
            cross_validation_score_r2,
            cross_validation_score_rmse,
        ],
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/supervised_cross_validation_scores.joblib",
    )

    dump(
        [test_scores_mae, test_scores_r2, test_scores_rmse],
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/supervised_test_scores.joblib",
    )
else:
    (
        cross_validation_score_mae,
        cross_validation_score_r2,
        cross_validation_score_rmse,
    ) = load(
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/supervised_cross_validation_scores.joblib"
    )
    test_scores_mae, test_scores_r2, test_scores_rmse = load(
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/supervised_test_scores.joblib"
    )

test_scores_mae = [round(value, 2) for value in test_scores_mae]
test_scores_r2 = [round(value, 2) for value in test_scores_r2]

cross_validation_score_r2_mean = [
    str(round(value.mean(), 2)) + " + " + str(round(np.std(value), 2))
    for value in cross_validation_score_r2
]
cross_validation_score_mae_mean = [
    str(round(-1 * value.mean(), 2)) + " + " + str(round(np.std(value), 2))
    for value in cross_validation_score_mae
]


df_model_accuracy_scores = pd.DataFrame(
    data={
        "Test R2": test_scores_r2,
        "Test MAE": test_scores_mae,
        "5 fold Cross Valdiation R2": cross_validation_score_r2_mean,
        "5 fold Cross Valdiation MAE": cross_validation_score_mae_mean,
    }
)


df_model_accuracy_scores.to_csv(
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/supervised_model_accuracy_scores.csv"
)


# learning curves
train_sizes_classes = []
train_scores_classes = []
test_scores_classes = []

if not args.f:
    retrain = input("Recalculate learning curves [y]?")
if retrain == "y" or args.f:
    for i in range(1, 4):
        train_index = y_train_all["domain_class"] == str(i)
        test_index = y_test_all["domain_class"] == str(i)

        X_train = X_train_all[train_index]
        y_train = y_train_all[train_index][property_column]

        train_sizes, train_scores, test_scores = learning_curve(
            rf_supervised_regressors[i - 1],
            X_train,
            y_train,
            train_sizes=np.linspace(0.1, 1, 10),
            scoring="neg_mean_absolute_error",
            n_jobs=N_JOBS,
        )
        train_sizes_classes.append(train_sizes)
        train_scores_classes.append([-1 * value.mean() for value in train_scores])
        test_scores_classes.append([-1 * value.mean() for value in test_scores])
    dump(
        [train_sizes_classes, train_scores_classes, test_scores_classes],
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/supervised_learning_curves.joblib",
    )
else:
    train_sizes_classes, train_scores_classes, test_scores_classes = load(
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/supervised_learning_curves.joblib"
    )

for i in range(3):
    save_learning_curve(
        train_sizes_classes[i],
        train_scores_classes[i],
        test_scores_classes[i],
        RANGE_DICTIONARY[property_column]["learning_curve"],
        PROPERTY_NAME,
        ["A", "B", "C"][i],
        RANGE_DICTIONARY[property_column]["unit"],
        True,
    )

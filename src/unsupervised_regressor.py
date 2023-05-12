#!/usr/bin/env python


import pandas as pd
import numpy as np


# preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# Pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from supervised_utils import (
    get_data,
    save_feature_importance_profiles,
    save_learning_curve,
    save_rfecv,
)
from constants import (
    ALLOYING_ELEMENTS,
    PROCESSING_COLUMNS,
    TARGET_COLUMNS,
    N_JOBS,
    NAME_REPLACE,
    RANGE_DICTIONARY,
    N_ITER,
    FONT_SIZE,
)

# Regression Models Training
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint


# Model Evaluation
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import RFECV

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


def get_bin_function(partitions):
    def get_property_bin(property):
        for i, partition in enumerate(partitions):
            if property < partition:
                return str(i + 1)
        return str(len(partitions) + 1)

    return get_property_bin


partitions = [(250, 500), (300, 550), (10, 25)]


df_interested["domain_class"] = df_interested[property_column].apply(
    get_bin_function(partitions[selected_property_index])
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
    n_iter=100,
    n_jobs=N_JOBS,
    random_state=10,
)


def random_reg(random_state):
    return RandomizedSearchCV(
        estimator=rf,
        param_distributions=distributions,
        n_iter=N_ITER,
        n_jobs=N_JOBS,
        random_state=random_state,
    )


# Unsupervised classes


retrain = ""
if not args.f:
    retrain = input("Retrain unsupervised classifier [y]?")

if retrain == "y" or args.f:
    class_cv.fit(X_train_all, y_train_all["class"])
    unsupervised_classifier = class_cv.best_estimator_
    dump(
        unsupervised_classifier,
        f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_classifier.joblib",
    )
else:
    unsupervised_classifier = load(
        f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_classifier.joblib"
    )

print(
    confusion_matrix(
        unsupervised_classifier.predict(X_test_all),
        y_test_all["class"],
        normalize="true",
    )
)
decition_tree = DecisionTreeClassifier()
decition_tree.fit(X_train_all, y_train_all["class"])

# Plot confusion matrix
plt.figure(figsize=(10, 10))
confusion_materix_heatmap = sns.heatmap(
    confusion_matrix(
        unsupervised_classifier.predict(X_test_all),
        y_test_all["class"],
        normalize="true",
    ),
    annot=True,
    fmt=".2f",
    square=True,
    cmap="Blues",
    xticklabels=[f"Class {i}" for i in range(1, 9)],
    yticklabels=[f"Class {i}" for i in range(1, 9)],
)
plt.xlabel("Predicted Label", fontsize=FONT_SIZE * 1.5)
plt.ylabel("Actual Label", fontsize=FONT_SIZE * 1.5)

plt.xticks(rotation=45, fontsize=FONT_SIZE * 1.25)
plt.yticks(rotation=45, fontsize=FONT_SIZE * 1.25)
# dont show legend

plt.savefig(
    f"../images/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_confusion_matrix.png",
    bbox_inches="tight",
    dpi=500,
)
rf_unsupervised_regressors = []
X_train_unsupervised_instances = []
y_train_unsupervised_instances = []

if not args.f:
    retrain = input("Retrain Unsupervised regressors [y]?")
if retrain == "y" or args.f:
    for i in range(1, 9):
        X_train = X_train_all[y_train_all["class"] == str(i)]
        X_train_unsupervised_instances.append(X_train)
        y_train = y_train_all[y_train_all["class"] == str(i)][property_column]
        y_train_unsupervised_instances.append(y_train)

        reg_cv = random_reg(i + 20 * selected_property_index)
        reg_cv.fit(X_train, y_train)
        rf_unsupervised_regressors.append(reg_cv.best_estimator_)

    dump(
        rf_unsupervised_regressors,
        f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_regressors.joblib",
    )
else:
    rf_unsupervised_regressors = load(
        f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_regressors.joblib"
    )


def predict_using_regressors(classifier, regressors):
    def predict_value(value):
        predicted_class = int(classifier.predict([value])[0])
        return regressors[predicted_class - 1].predict([value])[0]

    return predict_value


predict_unsupervised_value = predict_using_regressors(
    unsupervised_classifier, rf_unsupervised_regressors
)

# unsupervised_predicted = X_test_all.apply(predict_unsupervised_value, axis=1)

# print(mean_absolute_error(unsupervised_predicted, y_test_all[property_column]))


params = []
for i in range(8):
    params.append(list(rf_unsupervised_regressors[i].get_params().values()))
df_model_parameters = pd.DataFrame(
    params,
    columns=rf_unsupervised_regressors[0].get_params().keys(),
    index=[i for i in range(1, 9)],
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
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/model_params/unsupervised_regressors_params.csv"
)


if not args.f:
    retrain = input("Recalculate accuracy [y]?")
if retrain == "y" or args.f:
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
    }
    test_scores_r2 = []
    test_scores_mae = []
    cross_validation_score_r2 = []
    cross_validation_score_mae = []
    for i in range(8):
        train_index = y_train_all["class"] == str(i + 1)
        test_index = y_test_all["class"] == str(i + 1)

        X_train = X_train_all[train_index]
        y_train = y_train_all[train_index][property_column]
        X_test = X_test_all[test_index]
        y_test = y_test_all[test_index][property_column]

        test_scores_r2.append(
            r2_score(
                rf_unsupervised_regressors[i].predict(X_test),
                y_test,
            )
        )
        test_scores_mae.append(
            mean_absolute_error(
                rf_unsupervised_regressors[i].predict(X_test),
                y_test,
            )
        )

        scores = cross_validate(
            rf_unsupervised_regressors[i],
            X_train,
            y_train,
            scoring=scoring,
            cv=5,
            n_jobs=N_JOBS,
        )
        cross_validation_score_r2.append(scores["test_r2"])
        cross_validation_score_mae.append(scores["test_mae"])
    dump(
        [
            cross_validation_score_mae,
            cross_validation_score_r2,
        ],
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_cross_validation_scores.joblib",
    )
    dump(
        [test_scores_mae, test_scores_r2],
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_test_scores.joblib",
    )
else:
    (cross_validation_score_mae, cross_validation_score_r2,) = load(
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_cross_validation_scores.joblib"
    )
    test_scores_mae, test_scores_r2, = load(
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_test_scores.joblib"
    )

## Do the above calculation for mean abosulte percentage error

if not args.f:
    retrain = input("Recalculate MAPE accuracy [y]?")
if retrain == "y" or args.f:
    scoring = {
        "mape": "neg_mean_absolute_percentage_error",
    }
    test_scores_mape = []
    cross_validation_score_mape = []
    for i in range(8):
        train_index = y_train_all["class"] == str(i + 1)
        test_index = y_test_all["class"] == str(i + 1)

        X_train = X_train_all[train_index]
        y_train = y_train_all[train_index][property_column]
        X_test = X_test_all[test_index]
        y_test = y_test_all[test_index][property_column]

        test_scores_mape.append(
            mean_absolute_percentage_error(
                rf_unsupervised_regressors[i].predict(X_test),
                y_test,
            )
        )

        scores = cross_validate(
            rf_unsupervised_regressors[i],
            X_train,
            y_train,
            scoring=scoring,
            cv=5,
            n_jobs=N_JOBS,
        )
        cross_validation_score_mape.append(-1 * scores["test_mape"])
    dump(
        cross_validation_score_mape,
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_cross_validation_scores_mape.joblib",
    )
    dump(
        test_scores_mape,
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_test_scores_mape.joblib",
    )
else:
    cross_validation_score_mape = load(
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_cross_validation_scores_mape.joblib"
    )
    test_scores_mape = load(
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_test_scores_mape.joblib"
    )


test_scores_mae = [round(value, 2) for value in test_scores_mae]
test_scores_r2 = [round(value, 2) for value in test_scores_r2]
test_scores_mape = [round(value, 2) for value in test_scores_mape]
cross_validation_score_r2_mean = [
    str(round(value.mean(), 2)) + " + " + str(round(np.std(value), 2))
    for value in cross_validation_score_r2
]
cross_validation_score_mae_mean = [
    str(round(-1 * value.mean(), 2)) + " + " + str(round(np.std(value), 2))
    for value in cross_validation_score_mae
]
cross_validation_score_mape_mean = [
    str(round(value.mean(), 2)) + " + " + str(round(np.std(value), 2))
    for value in cross_validation_score_mape
]

df_model_accuracy_scores = pd.DataFrame(
    data={
        "Test R2": test_scores_r2,
        "Test MAE": test_scores_mae,
        "Test MAPE": test_scores_mape,
        "5 fold Cross Valdiation R2": cross_validation_score_r2_mean,
        "5 fold Cross Valdiation MAE": cross_validation_score_mae_mean,
        "5 fold Cross Valdiation MAPE": cross_validation_score_mape_mean,
    }
)


df_model_accuracy_scores.to_csv(
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_regressors_accuracy_scores.csv"
)

# ### Learning Curves


train_sizes_classes = []
train_scores_classes = []
test_scores_classes = []

if not args.f:
    retrain = input("Recalculate learning curves [y]?")

if retrain == "y" or args.f:
    for i in range(8):
        train_index = y_train_all["class"] == str(i + 1)
        test_index = y_test_all["class"] == str(i + 1)
        X_train = X_train_all[train_index]
        y_train = y_train_all[train_index][property_column]
        train_sizes, train_scores, test_scores = learning_curve(
            rf_unsupervised_regressors[i],
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
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_learning_curves.joblib",
    )
else:
    train_sizes_classes, train_scores_classes, test_scores_classes = load(
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/unsupervised_learning_curves.joblib"
    )


# ### Save Learning Curves


for i in range(8):
    save_learning_curve(
        train_sizes_classes[i],
        train_scores_classes[i],
        test_scores_classes[i],
        RANGE_DICTIONARY[property_column]["learning_curve"],
        PROPERTY_NAME,
        i + 1,
        RANGE_DICTIONARY[property_column]["unit"],
    )


# ###  Feature importance profile


for i in range(8):
    train_index = y_train_all["class"] == str(i + 1)
    X_train = X_train_all[train_index]
    y_train = y_train_all[train_index][property_column]

    df_feature_importance = pd.DataFrame(
        data={
            "Features": X_train.columns,
            "Importance": rf_unsupervised_regressors[i].feature_importances_,
        }
    )

    df_feature_importance.sort_values(by=["Importance"], ascending=False, inplace=True)
    df_feature_importance["Features"].replace(
        NAME_REPLACE,
        inplace=True,
    )
    save_feature_importance_profiles(
        df_feature_importance["Features"][:10].values,
        df_feature_importance["Importance"][:10].values,
        PROPERTY_NAME,
        RANGE_DICTIONARY[property_column]["feature_importance"],
        i + 1,
    )

if not args.f:
    retrain = input("Recalculate recursive feature elimination [y]?")

if retrain == "y" or args.f:
    selectors_scores = []

    for i in range(8):
        train_index = y_train_all["class"] == str(i + 1)
        X_train = X_train_all[train_index]
        y_train = y_train_all[train_index]

        selector = RFECV(
            rf_unsupervised_regressors[i],
            step=1,
            scoring="neg_mean_absolute_error",
            min_features_to_select=1,
            n_jobs=N_JOBS,
        )
        selector = selector.fit(X_train, y_train)
        selectors_scores.append(selector.cv_results_["mean_test_score"])
    dump(
        selectors_scores,
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/selectors_scores.txt",
    )

selectors_scores = load(
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/selectors_scores.txt"
)

for i in range(8):
    save_rfecv(
        selectors_scores[i],
        PROPERTY_NAME,
        i + 1,
        RANGE_DICTIONARY[property_column]["unit"],
    )

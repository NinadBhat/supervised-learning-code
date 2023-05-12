#!/usr/bin/env python


import pandas as pd
import numpy as np


# preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix


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
    N_ITER,
    FONT_SIZE,
    NAME_REPLACE,
    RANGE_DICTIONARY,
    PRECISION,
)

# Regression Models Training
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.model_selection import learning_curve
from sklearn.feature_selection import RFECV

# Model Evaluation
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score, mean_absolute_error

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


# Entire data set model
y_train = y_train_all[property_column]
y_test = y_test_all[property_column]

retrain = ""
if not args.f:
    retrain = input("Retrain model [y]?")

if retrain == "y" or args.f:
    reg_cv = random_reg(12)
    reg_cv.fit(X_train_all, y_train)
    reg_best = reg_cv.best_estimator_
    dump(
        reg_cv.best_estimator_,
        f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/rf_all_instances_best_estimator.joblib",
    )
else:
    reg_best = load(
        open(
            f"../models/{PROPERTY_NAME.replace(' ', '_').lower()}/rf_all_instances_best_estimator.joblib",
            "rb",
        )
    )

# store model parameters
with open(
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/model_params/all_regressor_params.txt",
    "w",
) as f:
    f.write(str(reg_best.get_params()))

scoring = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}
print("R2 score :", round(r2_score(reg_best.predict(X_test_all), y_test), PRECISION))
print(
    "Mean absolute error :",
    round(
        mean_absolute_error(reg_best.predict(X_test_all), y_test),
        PRECISION,
    ),
)

scores = cross_validate(
    reg_best,
    X_train_all,
    y_train,
    scoring=scoring,
    cv=5,
    n_jobs=N_JOBS,
)
print(
    f"R2 cross validation score : {round(scores['test_r2'].mean(),2)} + {round(scores['test_r2'].std(),2)}"
)
print(
    f"MAE cross validation score : {round(-scores['test_mae'].mean(),2)} + {round(scores['test_mae'].std(),2)}"
)


plt.clf()
plt.figure(figsize=(5, 5), dpi=100)

x = reg_best.predict(X_test_all)
y = y_test_all[property_column]
plt.scatter(x, y, s=15, label="Test instances")
plt.plot(y, y, "-", color="r", label="Perfect fit")
plt.plot(
    np.unique(x),
    np.poly1d(np.polyfit(x, y, 1))(np.unique(x)),
    "-",
    color="g",
    label="Best fit line",
)
plt.xlabel(f"Predicted {PROPERTY_NAME.lower()}", fontsize=FONT_SIZE, family="Arial")
plt.ylabel(f"True {PROPERTY_NAME.lower()}", fontsize=FONT_SIZE, family="Arial")
plt.legend(loc="lower right")

plt.savefig(
    f"../images/{PROPERTY_NAME.replace(' ', '_').lower()}/learning_curve/true_vs_predicted.png",
    bbox_inches="tight",
    dpi=330,
)


# ## Learning Curve


if not args.f:
    retrain = input("Retrain learning curve [y]?")

if retrain == "y" or args.f:
    train_sizes, train_scores, test_scores = learning_curve(
        reg_best,
        X_train_all,
        y_train,
        train_sizes=np.linspace(0.1, 1, 10),
        scoring="neg_mean_absolute_error",
        n_jobs=N_JOBS,
    )
    dump(
        train_sizes,
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/train_sizes.txt",
    )
    dump(
        train_scores,
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/train_scores.txt",
    )
    dump(
        test_scores,
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/test_scores.txt",
    )
train_sizes = load(
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/train_sizes.txt"
)
train_scores = load(
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/train_scores.txt"
)
test_scores = load(
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/test_scores.txt"
)

plt.clf()
save_learning_curve(
    train_sizes,
    [-1 * value.mean() for value in train_scores],
    [-1 * value.mean() for value in test_scores],
    RANGE_DICTIONARY[property_column]["learning_curve"],
    PROPERTY_NAME,
    "all instances",
    RANGE_DICTIONARY[property_column]["unit"],
)

# # Feature Importance Profile


df_feature_importance = pd.DataFrame(
    data={
        "Features": X_train_all.columns,
        "Importance": reg_best.feature_importances_,
    }
)

df_feature_importance.sort_values(by=["Importance"], ascending=False, inplace=True)
df_feature_importance["Features"].replace(
    NAME_REPLACE,
    inplace=True,
)

save_feature_importance_profiles(
    df_feature_importance.iloc[:10]["Features"],
    df_feature_importance.iloc[:10]["Importance"],
    PROPERTY_NAME,
    RANGE_DICTIONARY[property_column]["feature_importance"],
    "all instances",
)

# ## Recursive feature elimination
#

if not args.f:
    retrain = input("Retrain RFE [y]?")

if retrain == "y" or args.f:
    selector = RFECV(
        reg_best,
        step=1,
        scoring="neg_mean_absolute_error",
        min_features_to_select=1,
        n_jobs=N_JOBS,
    )
    selector = selector.fit(X_train_all, y_train)
    dump(
        selector.cv_results_["mean_test_score"],
        f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/rfe_mean_test_scores.txt",
    )

rfe_mean_test_scores = load(
    f"../value_store/{PROPERTY_NAME.replace(' ', '_').lower()}/rfe_mean_test_scores.txt"
)

save_rfecv(
    rfe_mean_test_scores,
    PROPERTY_NAME,
    "all instances",
    RANGE_DICTIONARY[property_column]["unit"],
)

"""Hyperparameter tuning for the Adult Income classification_project project."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.consts import (
    N_JOBS,
    RANDOM_STATE,
)


def drop_fnlwgt(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    return X.drop(columns=["fnlwgt"], errors="ignore")


def keep_only_educational_num(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    return X.drop(columns=["education"], errors="ignore")


def drop_race(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    return X.drop(columns=["race"], errors="ignore")


def drop_native_country(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    return X.drop(columns=["native-country"], errors="ignore")


def add_capital_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X["has_capital_gain"] = (X["capital-gain"] > 0).astype(int)
    X["has_capital_loss"] = (X["capital-loss"] > 0).astype(int)
    X["log1p_capital_gain"] = np.log1p(X["capital-gain"])
    X["log1p_capital_loss"] = np.log1p(X["capital-loss"])
    X["capital_delta"] = X["capital-gain"] - X["capital-loss"]

    return X


def add_hours_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    X["is_part_time"] = (X["hours-per-week"] < 35).astype(int)
    X["is_overtime"] = (X["hours-per-week"] > 40).astype(int)
    X["hours_from_40"] = X["hours-per-week"] - 40

    return X


def apply_selected_feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Apply selected feature engineering from previous stage."""

    X = drop_fnlwgt(X)
    X = keep_only_educational_num(X)
    X = drop_race(X)
    X = drop_native_country(X)
    X = add_capital_features(X)
    X = add_hours_features(X)

    return X


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def get_feature_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features


def make_preprocessor(
    X_sample: pd.DataFrame,
) -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_columns(X_sample)

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_selected_pipeline(
    X_train: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    transformer = FunctionTransformer(
        apply_selected_feature_engineering,
        validate=False,
    )

    X_after_transform = transformer.transform(X_train)

    return Pipeline(
        steps=[
            ("feature_engineering", transformer),
            ("preprocessing", make_preprocessor(X_after_transform)),
            (
                "model",
                GradientBoostingClassifier(
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_scoring() -> dict:
    return {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(
            precision_score,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": make_scorer(
            recall_score,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": make_scorer(
            f1_score,
            average="macro",
            zero_division=0,
        ),
        "roc_auc": "roc_auc",
    }


def make_param_grid() -> dict:
    """Small grid for GradientBoostingClassifier.

    3 * 2 * 2 * 2 = 24 combinations.
    With 5-fold CV this gives 120 fits.
    """

    return {
        "model__learning_rate": [0.03, 0.05, 0.08],
        "model__n_estimators": [100, 200],
        "model__max_depth": [2, 3],
        "model__min_samples_leaf": [20, 50],
    }


def make_grid_search(
    X_train: pd.DataFrame,
    param_grid: dict | None = None,
    cv: StratifiedKFold | None = None,
    n_jobs: int = N_JOBS,
    verbose: int = 1,
) -> GridSearchCV:
    estimator = make_selected_pipeline(X_train)

    param_grid = param_grid or make_param_grid()

    cv = cv or StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=make_scoring(),
        refit="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )


def run_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict | None = None,
    cv: StratifiedKFold | None = None,
    n_jobs: int = N_JOBS,
    verbose: int = 1,
) -> GridSearchCV:
    search = make_grid_search(
        X_train=X_train,
        param_grid=param_grid,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    search.fit(X_train, y_train)

    return search


def get_tuning_results(search: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(search.cv_results_)

    selected_columns = [
        "rank_test_f1_macro",
        "mean_test_accuracy",
        "std_test_accuracy",
        "mean_test_precision_macro",
        "std_test_precision_macro",
        "mean_test_recall_macro",
        "std_test_recall_macro",
        "mean_test_f1_macro",
        "std_test_f1_macro",
        "mean_test_roc_auc",
        "std_test_roc_auc",
        "mean_train_f1_macro",
        "std_train_f1_macro",
        "mean_fit_time",
        "params",
    ]

    results = results[selected_columns].copy()

    return results.sort_values(
        "rank_test_f1_macro",
        ascending=True,
    )


def get_best_tuning_summary(search: GridSearchCV) -> dict:
    return {
        "best_f1_macro": search.best_score_,
        "best_params": search.best_params_,
        "best_estimator": search.best_estimator_,
    }


def plot_tuning_results(
    tuning_results: pd.DataFrame,
    metric: str = "mean_test_f1_macro",
    top_n: int = 10,
    figsize: tuple[int, int] = (10, 6),
):
    import matplotlib.pyplot as plt

    plot_data = tuning_results.head(top_n).sort_values(
        metric,
        ascending=True,
    )

    labels = [
        f"rank {rank}"
        for rank in plot_data["rank_test_f1_macro"]
    ]

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        labels,
        plot_data[metric],
        xerr=plot_data.get(metric.replace("mean", "std")),
    )

    ax.set_title(f"Top {top_n} tuning configurations by {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("configuration")

    for index, value in enumerate(plot_data[metric]):
        ax.text(
            value,
            index,
            f" {value:.4f}",
            va="center",
        )

    fig.tight_layout()

    return fig, ax


def plot_train_validation_gap(
    tuning_results: pd.DataFrame,
    top_n: int = 10,
    figsize: tuple[int, int] = (10, 6),
):
    import matplotlib.pyplot as plt

    plot_data = tuning_results.head(top_n).copy()
    x = np.arange(len(plot_data))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        x,
        plot_data["mean_train_f1_macro"],
        marker="o",
        label="train f1_macro",
    )

    ax.plot(
        x,
        plot_data["mean_test_f1_macro"],
        marker="o",
        label="cv f1_macro",
    )

    ax.set_title(f"Train vs CV f1_macro for top {top_n} configurations")
    ax.set_xlabel("configuration rank")
    ax.set_ylabel("f1_macro")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_data["rank_test_f1_macro"])
    ax.legend()

    fig.tight_layout()

    return fig, ax
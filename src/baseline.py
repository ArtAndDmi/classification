"""Baseline models for the Adult Income classification_project."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.consts import (
    RANDOM_STATE,
    TARGET,
    N_JOBS
)
from sklearn.inspection import permutation_importance



def make_cv(
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Create cross-validation splitter for classification_project."""

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def get_feature_columns(
        data: pd.DataFrame,
        target: str = TARGET,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature columns."""

    X = data.drop(columns=[target])

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with different sklearn versions."""

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


def make_preprocessor(
        numeric_features: list[str],
        categorical_features: list[str],
        scale_numeric: bool = False,
) -> ColumnTransformer:
    """Create preprocessing for baseline models.

    Baseline intentionally keeps '?' values as ordinary categories.
    Missing-value replacement is treated as a feature engineering hypothesis.
    """

    if scale_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

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


def make_baseline_models(
        random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Create baseline classifiers."""

    return {
        "dummy": DummyClassifier(
            strategy="most_frequent",
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=N_JOBS,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=random_state,
        ),
    }


def make_model_pipeline(
        model: object,
        numeric_features: list[str],
        categorical_features: list[str],
        scale_numeric: bool = False,
) -> Pipeline:
    """Build full preprocessing + model pipeline."""

    return Pipeline(
        steps=[
            (
                "preprocessing",
                make_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("model", model),
        ]
    )


def run_baseline_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold | None = None,
    n_jobs: int = N_JOBS,
) -> pd.DataFrame:
    """Run baseline cross-validation for classification_project models."""

    cv = cv or make_cv()

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    models = make_baseline_models()

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
    }

    results = []

    for model_name, model in models.items():
        scale_numeric = model_name == "logistic_regression"

        pipeline = make_model_pipeline(
            model=model,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scale_numeric=scale_numeric,
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
        )

        row = {
            "model": model_name,
        }

        for metric_name in scoring:
            test_scores = scores[f"test_{metric_name}"]

            row[f"mean_{metric_name}"] = np.mean(test_scores)
            row[f"std_{metric_name}"] = np.std(test_scores)

        results.append(row)

    return pd.DataFrame(results).sort_values(
        "mean_f1_macro",
        ascending=False,
    )


def plot_baseline_results(
        results: pd.DataFrame,
        metric: str = "mean_f1_macro",
        figsize: tuple[int, int] = (10, 5),
):
    """Plot baseline results by selected metric."""

    import matplotlib.pyplot as plt

    if metric not in results.columns:
        raise ValueError(f"Metric column '{metric}' not found in results.")

    plot_data = results.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        plot_data["model"],
        plot_data[metric],
    )

    ax.set_title(f"Baseline models by {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("model")

    for index, value in enumerate(plot_data[metric]):
        ax.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
        )

    fig.tight_layout()

    return fig, ax




def fit_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Pipeline:
    """Fit one baseline model on the full train data."""

    numeric_features = X_train.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    models = make_baseline_models()

    if model_name not in models:
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            f"Available models: {list(models.keys())}"
        )

    scale_numeric = model_name == "logistic_regression"

    pipeline = make_model_pipeline(
        model=models[model_name],
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def get_permutation_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: str = "f1_macro",
    n_repeats: int = 5,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
) -> pd.DataFrame:
    """Calculate permutation importance for original input features."""

    result = permutation_importance(
        pipeline,
        X,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )

    return importance.sort_values(
        "importance_mean",
        ascending=False,
    )


def plot_permutation_importance(
    importance: pd.DataFrame,
    top_n: int = 15,
    figsize: tuple[int, int] = (10, 6),
):
    """Plot top permutation importances."""

    import matplotlib.pyplot as plt

    plot_data = importance.head(top_n).sort_values(
        "importance_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        plot_data["feature"],
        plot_data["importance_mean"],
        xerr=plot_data["importance_std"],
    )

    ax.set_title("Permutation importance")
    ax.set_xlabel("Mean score decrease")
    ax.set_ylabel("Feature")

    for index, value in enumerate(plot_data["importance_mean"]):
        ax.text(
            value,
            index,
            f" {value:.4f}",
            va="center",
        )

    fig.tight_layout()

    return fig, ax

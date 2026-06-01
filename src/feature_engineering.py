"""Feature engineering experiments for the Adult Income classification project."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.consts import (
    N_JOBS,
    RANDOM_STATE,
)


@dataclass
class FeatureEngineeringExperiment:
    name: str
    description: str
    estimator: Pipeline


def make_cv(
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = RANDOM_STATE,
) -> StratifiedKFold:
    """Create cross-validation splitter for classification."""

    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


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


def make_scoring() -> dict:
    """Create scoring dictionary for classification experiments."""

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


def replace_question_marks_with_nan(X: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' markers with np.nan."""

    X = X.copy()
    return X.replace("?", np.nan)


def fill_question_marks_with_mode(X: pd.DataFrame) -> pd.DataFrame:
    """Replace '?' markers in categorical columns with column mode."""

    X = X.copy()

    for col in X.select_dtypes(include=["object", "category", "bool"]).columns:
        mode_values = X.loc[X[col] != "?", col].mode(dropna=True)

        if not mode_values.empty:
            X[col] = X[col].replace("?", mode_values.iloc[0])

    return X


def drop_fnlwgt(X: pd.DataFrame) -> pd.DataFrame:
    """Drop fnlwgt statistical weight column."""

    X = X.copy()
    return X.drop(columns=["fnlwgt"], errors="ignore")


def drop_race(X: pd.DataFrame) -> pd.DataFrame:
    """Drop race column."""

    X = X.copy()
    return X.drop(columns=["race"], errors="ignore")


def drop_native_country(X: pd.DataFrame) -> pd.DataFrame:
    """Drop native-country column."""

    X = X.copy()
    return X.drop(columns=["native-country"], errors="ignore")


def keep_only_education(X: pd.DataFrame) -> pd.DataFrame:
    """Keep education and drop educational-num."""

    X = X.copy()
    return X.drop(columns=["educational-num"], errors="ignore")


def keep_only_educational_num(X: pd.DataFrame) -> pd.DataFrame:
    """Keep educational-num and drop education."""

    X = X.copy()
    return X.drop(columns=["education"], errors="ignore")


def add_capital_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add capital-related features."""

    X = X.copy()

    X["has_capital_gain"] = (X["capital-gain"] > 0).astype(int)
    X["has_capital_loss"] = (X["capital-loss"] > 0).astype(int)
    X["log1p_capital_gain"] = np.log1p(X["capital-gain"])
    X["log1p_capital_loss"] = np.log1p(X["capital-loss"])
    X["capital_delta"] = X["capital-gain"] - X["capital-loss"]

    return X


def add_hours_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add working-hours features."""

    X = X.copy()

    X["is_part_time"] = (X["hours-per-week"] < 35).astype(int)
    X["is_overtime"] = (X["hours-per-week"] > 40).astype(int)
    X["hours_from_40"] = X["hours-per-week"] - 40

    return X


def add_native_country_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add country-related features."""

    X = X.copy()

    X["is_us_native_country"] = (X["native-country"] == "United-States").astype(int)

    return X


def make_transformer(function: Callable[[pd.DataFrame], pd.DataFrame]) -> FunctionTransformer:
    """Create pandas-friendly FunctionTransformer."""

    return FunctionTransformer(
        func=function,
        validate=False,
    )


def combine_transformers(
    *functions: Callable[[pd.DataFrame], pd.DataFrame],
) -> FunctionTransformer:
    """Combine several dataframe transformations into one transformer."""

    def transform(X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()

        for function in functions:
            X_transformed = function(X_transformed)

        return X_transformed

    return make_transformer(transform)


def get_feature_columns(
    X: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return numeric and categorical feature columns."""

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numeric_features, categorical_features


def make_preprocessor(
    X_sample: pd.DataFrame,
) -> ColumnTransformer:
    """Create preprocessing after feature engineering."""

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


def make_gb_pipeline(
    X_sample: pd.DataFrame,
    transformer: FunctionTransformer | None = None,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build GradientBoostingClassifier pipeline for one experiment."""

    if transformer is not None:
        X_after_transform = transformer.transform(X_sample)
    else:
        X_after_transform = X_sample.copy()

    steps = []

    if transformer is not None:
        steps.append(("feature_engineering", transformer))

    steps.extend(
        [
            ("preprocessing", make_preprocessor(X_after_transform)),
            (
                "model",
                GradientBoostingClassifier(random_state=random_state),
            ),
        ]
    )

    return Pipeline(steps=steps)


def make_feature_engineering_experiments(
    X_train: pd.DataFrame,
) -> list[FeatureEngineeringExperiment]:
    """Create feature engineering experiments for selected baseline model."""

    return [
        FeatureEngineeringExperiment(
            name="baseline_gradient_boosting",
            description="Baseline GradientBoostingClassifier without feature engineering.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
            ),
        ),
        FeatureEngineeringExperiment(
            name="question_marks_to_mode",
            description="Replace '?' markers with the most frequent category in each categorical column.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=make_transformer(fill_question_marks_with_mode),
            ),
        ),
        FeatureEngineeringExperiment(
            name="drop_fnlwgt",
            description="Drop fnlwgt statistical weight column.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=make_transformer(drop_fnlwgt),
            ),
        ),
        FeatureEngineeringExperiment(
            name="only_education",
            description="Keep education and drop educational-num.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=make_transformer(keep_only_education),
            ),
        ),
        FeatureEngineeringExperiment(
            name="only_educational_num",
            description="Keep educational-num and drop education.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=make_transformer(keep_only_educational_num),
            ),
        ),
        FeatureEngineeringExperiment(
            name="drop_race",
            description="Drop race column.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=make_transformer(drop_race),
            ),
        ),
        FeatureEngineeringExperiment(
            name="capital_features",
            description="Add capital indicators, log capital features and capital_delta.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=make_transformer(add_capital_features),
            ),
        ),
        FeatureEngineeringExperiment(
            name="hours_features",
            description="Add working-hours features: is_part_time, is_overtime, hours_from_40.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=make_transformer(add_hours_features),
            ),
        ),
        FeatureEngineeringExperiment(
            name="drop_native_country",
            description="Drop native-country column.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=make_transformer(drop_native_country),
            ),
        ),
        FeatureEngineeringExperiment(
            name="is_us_native_country",
            description="Add binary feature is_us_native_country.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=make_transformer(add_native_country_features),
            ),
        ),
        FeatureEngineeringExperiment(
            name="combined_reasonable_features",
            description="Drop fnlwgt, keep educational-num, add capital features, add hours features, add is_us_native_country.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=combine_transformers(
                    drop_fnlwgt,
                    keep_only_educational_num,
                    add_capital_features,
                    add_hours_features,
                    add_native_country_features,
                ),
            ),
        ),
        FeatureEngineeringExperiment(
            name="combined_without_race_and_native_country",
            description="Combined features plus drop race and native-country.",
            estimator=make_gb_pipeline(
                X_sample=X_train,
                transformer=combine_transformers(
                    drop_fnlwgt,
                    keep_only_educational_num,
                    drop_race,
                    drop_native_country,
                    add_capital_features,
                    add_hours_features,
                ),
            ),
        ),
    ]


def run_feature_engineering_experiments(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    experiments: list[FeatureEngineeringExperiment],
    cv: StratifiedKFold | None = None,
    n_jobs: int = N_JOBS,
) -> pd.DataFrame:
    """Run feature engineering experiments with cross-validation."""

    cv = cv or make_cv()
    scoring = make_scoring()

    results = []

    for experiment in experiments:
        scores = cross_validate(
            experiment.estimator,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            return_train_score=False,
        )

        row = {
            "experiment": experiment.name,
            "description": experiment.description,
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


def plot_feature_engineering_results(
    results: pd.DataFrame,
    metric: str = "mean_f1_macro",
    figsize: tuple[int, int] = (10, 6),
):
    """Plot feature engineering experiment results."""

    import matplotlib.pyplot as plt

    if metric not in results.columns:
        raise ValueError(f"Metric column '{metric}' not found in results.")

    plot_data = results.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=figsize)

    ax.barh(
        plot_data["experiment"],
        plot_data[metric],
    )

    ax.set_title(f"Feature engineering experiments by {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("experiment")

    for index, value in enumerate(plot_data[metric]):
        ax.text(
            value,
            index,
            f" {value:.3f}",
            va="center",
        )

    fig.tight_layout()

    return fig, ax
"""Final model utilities for the Adult Income classification project."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from src.consts import (
    RANDOM_STATE,
)


def drop_fnlwgt(X: pd.DataFrame) -> pd.DataFrame:
    """Drop fnlwgt statistical weight column."""

    X = X.copy()
    return X.drop(columns=["fnlwgt"], errors="ignore")


def keep_only_educational_num(X: pd.DataFrame) -> pd.DataFrame:
    """Keep educational-num and drop education."""

    X = X.copy()
    return X.drop(columns=["education"], errors="ignore")


def drop_race(X: pd.DataFrame) -> pd.DataFrame:
    """Drop race column."""

    X = X.copy()
    return X.drop(columns=["race"], errors="ignore")


def drop_native_country(X: pd.DataFrame) -> pd.DataFrame:
    """Drop native-country column."""

    X = X.copy()
    return X.drop(columns=["native-country"], errors="ignore")


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


def apply_final_feature_engineering(X: pd.DataFrame) -> pd.DataFrame:
    """Apply selected feature engineering from the classification project."""

    X = drop_fnlwgt(X)
    X = keep_only_educational_num(X)
    X = drop_race(X)
    X = drop_native_country(X)
    X = add_capital_features(X)
    X = add_hours_features(X)

    return X


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
    """Create preprocessing for final model."""

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


def make_final_model(
    X_train: pd.DataFrame,
) -> Pipeline:
    """Create final tuned classification pipeline."""

    feature_transformer = FunctionTransformer(
        apply_final_feature_engineering,
        validate=False,
    )

    X_after_feature_engineering = feature_transformer.transform(X_train)

    return Pipeline(
        steps=[
            ("feature_engineering", feature_transformer),
            ("preprocessing", make_preprocessor(X_after_feature_engineering)),
            (
                "model",
                GradientBoostingClassifier(
                    learning_rate=0.08,
                    n_estimators=200,
                    max_depth=3,
                    min_samples_leaf=20,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit final model on the full train data."""

    model = make_final_model(X_train)
    model.fit(X_train, y_train)

    return model


def predict_final_model(
    model: Pipeline,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Predict classes for test data."""

    return model.predict(X_test)


def predict_final_probabilities(
    model: Pipeline,
    X_test: pd.DataFrame,
    positive_class: str = ">50K",
) -> np.ndarray:
    """Predict probabilities for the positive class."""

    class_labels = list(model.classes_)

    if positive_class not in class_labels:
        raise ValueError(
            f"Positive class '{positive_class}' not found in model classes: {class_labels}"
        )

    positive_class_index = class_labels.index(positive_class)

    return model.predict_proba(X_test)[:, positive_class_index]


def evaluate_classification_model(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Calculate final classification metrics."""

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def make_classification_report(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Create text classification report."""

    return classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )


def make_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """Create confusion matrix."""

    return confusion_matrix(
        y_true,
        y_pred,
    )


def make_prediction_results(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    """Create dataframe with actual labels, predictions and probabilities."""

    results = X_test.copy()

    results["actual_income"] = y_test.values
    results["predicted_income"] = y_pred
    results["predicted_probability_positive"] = y_proba
    results["is_correct"] = results["actual_income"] == results["predicted_income"]

    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    figsize: tuple[int, int] = (6, 5),
):
    """Plot confusion matrix."""

    import matplotlib.pyplot as plt

    if labels is None:
        labels = ["<=50K", ">50K"]

    fig, ax = plt.subplots(figsize=figsize)

    image = ax.imshow(cm)
    fig.colorbar(image, ax=ax)

    ax.set_title("Confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
            )

    fig.tight_layout()

    return fig, ax


def plot_probability_distribution(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    positive_class: str = ">50K",
    figsize: tuple[int, int] = (10, 5),
):
    """Plot predicted probability distribution by true class."""

    import matplotlib.pyplot as plt

    y_true_series = pd.Series(y_true)

    fig, ax = plt.subplots(figsize=figsize)

    for label in sorted(y_true_series.unique()):
        mask = y_true_series == label

        ax.hist(
            y_proba[mask],
            bins=30,
            alpha=0.5,
            label=str(label),
        )

    ax.set_title("Predicted probability distribution")
    ax.set_xlabel(f"Predicted probability of {positive_class}")
    ax.set_ylabel("Count")
    ax.legend()

    fig.tight_layout()

    return fig, ax


def get_misclassified_examples(
    prediction_results: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return misclassified rows with highest positive-class probabilities."""

    return (
        prediction_results.loc[~prediction_results["is_correct"]]
        .sort_values("predicted_probability_positive", ascending=False)
        .head(top_n)
    )
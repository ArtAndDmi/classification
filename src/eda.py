import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)


def run_data_quality_report(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("DATA QUALITY REPORT")
    print("=" * 80)

    print(f"Shape: {df.shape}")
    print("\nDtypes:\n", df.dtypes)

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    print(pd.DataFrame({"missing": missing, "missing_%": missing_pct}).sort_values("missing_%", ascending=False))

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nConstant columns:")
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    print(const_cols if const_cols else "None")


def analyze_missing_values(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("MISSING VALUES ANALYSIS")
    print("=" * 80)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    report = pd.DataFrame({
        "missing": missing,
        "missing_%": missing_pct
    }).sort_values("missing_%", ascending=False)

    print(report[report["missing"] > 0])

    # визуализация топ пропусков
    top = report[report["missing"] > 0].head(20)

    if len(top) > 0:
        plt.figure(figsize=(10, 5))
        top["missing_%"].plot(kind="bar")
        plt.title("Top missing values (%)")
        plt.ylabel("% missing")
        plt.xticks(rotation=45)
        plt.show()
    else:
        print("No missing values found.")


def feature_summary_report(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("FEATURE SUMMARY REPORT")
    print("=" * 80)

    summary = []

    for col in df.columns:
        s = df[col]

        summary.append({
            "feature": col,
            "dtype": str(s.dtype),
            "missing_%": s.isna().mean() * 100,
            "nunique": s.nunique(dropna=False),
            "example_value": s.dropna().iloc[0] if s.notna().any() else None
        })

    summary_df = pd.DataFrame(summary)

    # разделим на numeric / categorical условно
    print("\nNUMERIC FEATURES:")
    print(summary_df[summary_df["dtype"].str.contains("int|float")])

    print("\nCATEGORICAL FEATURES:")
    print(summary_df[~summary_df["dtype"].str.contains("int|float")])


def analyze_numeric(series: pd.Series) -> None:
    series = series.copy()

    print("=" * 80)
    print(f"NUMERIC FEATURE: {series.name}")
    print("=" * 80)

    print(f"Count:          {series.count()}")
    print(f"Missing:        {series.isna().sum()}")
    print(f"Missing %:      {series.isna().mean() * 100:.2f}%")
    print(f"Unique:         {series.nunique()}")

    print("\nStatistics:")
    print(series.describe())

    print(f"\nMedian:         {series.median():.4f}")
    print(f"Skewness:       {series.skew():.4f}")
    print(f"Kurtosis:       {series.kurtosis():.4f}")

    plt.figure(figsize=(10, 4))
    plt.hist(series.dropna(), bins=30)
    plt.title(f"Distribution of {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 2))
    plt.boxplot(series.dropna(), vert=False)
    plt.title(f"Boxplot of {series.name}")
    plt.show()


def analyze_categorical(series: pd.Series, top_n: int = 20) -> None:
    """
    Анализ категориального признака.
    """

    series = series.copy()

    print("=" * 80)
    print(f"CATEGORICAL FEATURE: {series.name}")
    print("=" * 80)

    print(f"Count:          {series.count()}")
    print(f"Missing:        {series.isna().sum()}")
    print(f"Missing %:      {series.isna().mean() * 100:.2f}%")
    print(f"Unique:         {series.nunique()}")

    print("\nTop values:")
    print(series.value_counts(dropna=False).head(top_n))

    print("\nTop values (%):")
    print(
        (
                series.value_counts(dropna=False, normalize=True)
                * 100
        ).head(top_n)
    )

    value_counts = series.value_counts(dropna=False).head(top_n)

    plt.figure(figsize=(10, 5))
    value_counts.plot(kind="bar")
    plt.title(f"Top {top_n} categories: {series.name}")
    plt.ylabel("Count")
    plt.show()


def analyze_target_relationship(feature: pd.Series, target: pd.Series, top_n: int = 15) -> None:
    feature_name = feature.name
    target_name = target.name

    print("=" * 80)
    print(f"{feature_name} -> {target_name}")
    print("=" * 80)

    tmp = pd.concat(
        [feature, target],
        axis=1
    ).dropna()

    # NUMERIC FEATURE
    if pd.api.types.is_numeric_dtype(feature):

        corr = tmp.corr().iloc[0, 1]

        print(f"Correlation: {corr:.4f}")

        plt.figure(figsize=(8, 5))
        plt.scatter(
            tmp[feature_name],
            tmp[target_name],
            alpha=0.3
        )

        plt.xlabel(feature_name)
        plt.ylabel(target_name)
        plt.title(
            f"{feature_name} vs {target_name}"
        )

        plt.show()

    # CATEGORICAL FEATURE
    else:

        grouped = (
            tmp
            .groupby(feature_name)[target_name]
            .agg(["count", "mean"])
            .sort_values(
                "mean",
                ascending=False
            )
        )

        print(grouped.head(top_n))

        plt.figure(figsize=(10, 5))

        grouped["mean"].head(top_n).plot(
            kind="bar"
        )

        plt.ylabel(
            f"Mean {target_name}"
        )

        plt.title(
            f"Mean {target_name} by {feature_name}"
        )

        plt.xticks(rotation=45)

        plt.show()


def plot_classification_correlation_heatmap(
    df: pd.DataFrame,
    target: str | None = None,
    positive_class: str | None = None,
    max_categories: int = 20,
    figsize: tuple[int, int] = (12, 10),
) -> None:
    """
    Строит heatmap корреляций для задачи классификации.

    Что делает:
    - числовые признаки оставляет как есть;
    - категориальные признаки с небольшой кардинальностью кодирует через one-hot;
    - категориальные признаки с высокой кардинальностью пропускает;
    - бинарный target кодирует в 0/1;
    - если передан target, сортирует признаки по связи с target.
    """
    df_work = df.copy()

    if target is not None and target in df_work.columns:
        target_values = df_work[target].dropna().unique()

        if len(target_values) == 2:
            if positive_class is None:
                positive_class = sorted(target_values)[-1]

            df_work[target] = (df_work[target] == positive_class).astype(int)
        else:
            print(
                f"Target '{target}' has {len(target_values)} unique values. "
                "Target correlation sorting is most useful for binary classification_project."
            )

    categorical_cols = df_work.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    usable_categorical_cols = []
    skipped_categorical_cols = []

    for col in categorical_cols:
        if col == target:
            continue

        nunique = df_work[col].nunique(dropna=True)

        if nunique <= max_categories:
            usable_categorical_cols.append(col)
        else:
            skipped_categorical_cols.append(col)

    if skipped_categorical_cols:
        print("Skipped high-cardinality categorical columns:")
        print(skipped_categorical_cols)

    df_encoded = pd.get_dummies(
        df_work,
        columns=usable_categorical_cols,
        drop_first=True,
        dummy_na=False,
    )

    df_encoded = df_encoded.select_dtypes(include=["number", "bool"])

    corr = df_encoded.corr()

    if target is not None and target in corr.columns:
        sorted_cols = (
            corr[target]
            .abs()
            .sort_values(ascending=False)
            .index
        )

        corr = corr.loc[sorted_cols, sorted_cols]

    plt.figure(figsize=figsize)
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(corr.columns)),
        corr.columns,
        rotation=90,
    )

    plt.yticks(
        range(len(corr.index)),
        corr.index,
    )

    plt.title("Classification Correlation Heatmap")
    plt.tight_layout()
    plt.show()
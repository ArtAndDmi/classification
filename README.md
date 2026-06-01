# Adult Income Classification

## Описание проекта

Цель проекта — решить задачу бинарной классификации на датасете Adult Income и пройти полный классический ML pipeline в стиле реального Data Science проекта.

Задача состоит в предсказании уровня дохода человека:

```text
<=50K или >50K
```

по социально-демографическим и экономическим признакам: возрасту, образованию, типу занятости, профессии, семейному положению, рабочим часам, капиталу и другим характеристикам.

Проект был построен не только ради получения работающей модели, но и ради отработки правильного процесса:

- EDA;
- baseline;
- feature engineering;
- hyperparameter tuning;
- анализ feature importance;
- контроль качества модели на нескольких метриках;
- разделение логики между ноутбуками и `.py` файлами.

## Структура проекта

```text
project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   └── 05_final_model.ipynb
├── utils/
│   ├── consts.py
│   ├── eda.py
│   ├── baseline.py
│   ├── feature_engineering.py
│   └── hyperparamtuning.py
└── reports/
```

Основной архитектурный принцип проекта:

> Ноутбуки должны быть тонкими, а вся логика должна быть вынесена в `.py` файлы.

В ноутбуках остаются загрузка данных, вызовы функций, визуализация и выводы.

## Датасет

Использовался датасет Adult Income.

Основные характеристики:

- около 48.8k строк;
- 15 признаков;
- задача бинарной классификации;
- целевая переменная: `income`;
- классы: `<=50K` и `>50K`;
- пропуски встречаются только в некоторых категориальных признаках и закодированы знаком `?`.

## EDA

На этапе EDA были изучены распределения признаков, баланс классов, пропуски, выбросы, категориальные значения и связь признаков с целевой переменной.

Основные выводы:

- целевая переменная имеет умеренный дисбаланс: класс `<=50K` составляет около 76%, класс `>50K` около 24%;
- `age`, `educational-num`, `capital-gain`, `capital-loss` и `hours-per-week` связаны с целевой переменной;
- `capital-gain` и `capital-loss` имеют большое количество нулей и выраженные правые хвосты;
- `workclass`, `occupation` и `native-country` содержат пропуски, закодированные знаком `?`;
- `education` и `educational-num` фактически описывают одну и ту же сущность разными способами;
- `fnlwgt` является статистическим весом записи, а не прямой характеристикой человека;
- `native-country` сильно доминируется категорией `United-States`;
- `race` является чувствительным социальным признаком, его полезность и влияние стоит проверять отдельно.

По итогам EDA были сформированы гипотезы:

- обработать пропуски, закодированные как `?`;
- проверить замену `?` на моду;
- проверить удаление `fnlwgt`;
- проверить варианты использования `education` и `educational-num`;
- проверить удаление `race`;
- добавить признаки на основе `capital-gain` и `capital-loss`;
- добавить признаки на основе `hours-per-week`;
- проверить удаление `native-country`;
- проверить бинарный признак `is_us_native_country`.

## Baseline

На этапе baseline использовались исходные признаки без feature engineering.

Принцип baseline:

> Не применять гипотезы из EDA и не делать дополнительную обработку признаков.

Категориальные признаки кодировались через one-hot encoding.

Значения `?` на baseline не заменялись и рассматривались как обычные категории. Это позволяет не смешивать baseline с гипотезами по обработке пропусков.

Данные были разделены на train/test с сохранением пропорций классов:

```python
train_test_split(
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

Тестовая выборка была отложена до финальной оценки.

Для cross-validation использовался:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)
```

Stratified CV важен, потому что целевая переменная имеет дисбаланс классов.

Baseline-модели:

- `DummyClassifier`;
- `LogisticRegression`;
- `RandomForestClassifier`;
- `GradientBoostingClassifier`.

Использовались метрики:

- `accuracy`;
- `precision_macro`;
- `recall_macro`;
- `f1_macro`;
- `roc_auc`.

Основной метрикой для сравнения был выбран `f1_macro`, так как он лучше учитывает качество по обоим классам при дисбалансе.

Результаты baseline:

```text
GradientBoostingClassifier: f1_macro около 0.799
RandomForestClassifier:     f1_macro около 0.786
LogisticRegression:         f1_macro около 0.783
DummyClassifier:            f1_macro около 0.432
```

`DummyClassifier` получил достаточно высокий `accuracy` за счет предсказания доминирующего класса, но `f1_macro` и `roc_auc` показали, что такая модель не извлекает полезный сигнал из данных.

Лучшей baseline-моделью стал:

```text
GradientBoostingClassifier
```

## Feature Importance

Для лучшей baseline-модели был рассчитан permutation importance.

В качестве метрики использовался:

```text
f1_macro
```

Наиболее важными признаками оказались:

- `marital-status`;
- `capital-gain`;
- `educational-num`;
- `age`;
- `occupation`;
- `capital-loss`;
- `hours-per-week`.

Наиболее важным признаком оказался `marital-status`.

Важно:

> Permutation importance показывает вклад признака в качество модели, но не доказывает причинно-следственную связь.

Высокая важность `marital-status` не означает, что семейное положение само по себе напрямую определяет доход. Скорее всего, этот признак связан с возрастом, структурой домохозяйства, занятостью и другими социально-экономическими характеристиками.

Низкую важность показали:

- `race`;
- `native-country`;
- `fnlwgt`;
- `education`.

Низкая важность `education`, вероятно, связана с тем, что похожая информация уже содержится в `educational-num`.

## Feature Engineering

На этапе feature engineering проверялись гипотезы, сформированные после EDA.

Важно:

> Feature engineering проводился только на train-части данных. Test set не использовался.

В качестве основной модели использовался:

```text
GradientBoostingClassifier
```

Проверенные эксперименты:

- baseline `GradientBoostingClassifier`;
- замена `?` на моду;
- удаление `fnlwgt`;
- использование только `education`;
- использование только `educational-num`;
- удаление `race`;
- добавление capital features;
- добавление hours features;
- удаление `native-country`;
- добавление `is_us_native_country`;
- комбинированный набор разумных признаков;
- комбинированный набор с удалением `race` и `native-country`.

Capital features:

- `has_capital_gain`;
- `has_capital_loss`;
- `log1p_capital_gain`;
- `log1p_capital_loss`;
- `capital_delta`.

Hours features:

- `is_part_time`;
- `is_overtime`;
- `hours_from_40`.

Лучший результат показал эксперимент:

```text
combined_without_race_and_native_country
```

В этом варианте:

- удалялся `fnlwgt`;
- удалялся `education`, оставался `educational-num`;
- удалялся `race`;
- удалялся `native-country`;
- добавлялись признаки на основе `capital-gain` и `capital-loss`;
- добавлялись признаки на основе `hours-per-week`.

Сравнение:

```text
baseline GradientBoostingClassifier:        f1_macro около 0.7987
best feature engineering experiment:        f1_macro около 0.7995
```

Улучшение составило примерно:

```text
+0.0008 f1_macro
```

Вывод:

> Feature engineering не дал значимого улучшения относительно baseline. Лучший эксперимент формально улучшил качество, но улучшение находится внутри стандартного отклонения cross-validation.

При этом выбранный вариант признаков стал немного компактнее и чище, так как удалялись слабые или спорные признаки без потери качества.

## Hyperparameter Tuning

После выбора модели и схемы feature engineering был выполнен подбор гиперпараметров для `GradientBoostingClassifier`.

На этом этапе были зафиксированы:

- модель: `GradientBoostingClassifier`;
- preprocessing: `combined_without_race_and_native_country`;
- основная метрика: `f1_macro`;
- CV: stratified 5-fold cross-validation;
- test set не использовался.

Подбирались параметры:

- `learning_rate`;
- `n_estimators`;
- `max_depth`;
- `min_samples_leaf`.

Лучший результат после tuning:

```text
best CV f1_macro: около 0.8050
```

Лучшие параметры:

```python
{
    "learning_rate": 0.08,
    "n_estimators": 200,
    "max_depth": 3,
    "min_samples_leaf": 20,
}
```

Сравнение этапов:

```text
baseline GradientBoostingClassifier:        f1_macro около 0.7987
best feature engineering experiment:        f1_macro около 0.7995
tuned GradientBoostingClassifier:           f1_macro около 0.8050
```

Вывод:

> Основной прирост качества был получен за счет hyperparameter tuning, а не за счет feature engineering.

Полученное улучшение выглядит реалистичным: качество выросло умеренно, без подозрительно резкого скачка.

## Итоговый Результат

В проекте был пройден полный ML pipeline для задачи бинарной классификации:

1. Проведен EDA.
2. Сформированы гипотезы.
3. Построен честный baseline.
4. Выбрана лучшая baseline-модель.
5. Рассчитан permutation importance.
6. Проверены feature engineering гипотезы.
7. Выбрана схема признаков для tuning.
8. Выполнен hyperparameter tuning.
9. Получен итоговый tuned pipeline.

Итоговая выбранная модель:

```text
GradientBoostingClassifier
```

Итоговая схема feature engineering:

```text
combined_without_race_and_native_country
```

Итоговое качество на cross-validation:

```text
f1_macro около 0.8050
```

## Главные Выводы

- `GradientBoostingClassifier` оказался лучшей baseline-моделью.
- `accuracy` не является достаточной метрикой для этой задачи из-за дисбаланса классов.
- `f1_macro` лучше отражает качество модели по обоим классам.
- `DummyClassifier` показал, почему важно смотреть не только на accuracy.
- `marital-status`, `capital-gain`, `educational-num`, `age` и `occupation` оказались наиболее важными признаками.
- `education` можно удалить, если оставить `educational-num`.
- `race` и `native-country` можно удалить без потери качества.
- Feature engineering дал минимальное улучшение.
- Основной прирост качества был получен на этапе hyperparameter tuning.

## Что Можно Улучшить Дальше

В рамках учебного проекта задача считается решенной. Возможные направления развития:

- провести финальную оценку на test set;
- построить confusion matrix;
- построить classification report;
- отдельно проанализировать precision и recall для класса `>50K`;
- подобрать threshold классификации;
- сравнить ROC curve и PR curve;
- попробовать `HistGradientBoostingClassifier`;
- сравнить с LightGBM или XGBoost;
- добавить сохранение финального pipeline через `joblib`;
- провести отдельный fairness-анализ чувствительных признаков.

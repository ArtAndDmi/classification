## Выводы по hyperparameter tuning

На этапе hyperparameter tuning была зафиксирована лучшая выбранная схема feature engineering: `combined_without_race_and_native_country`.

Для модели `GradientBoostingClassifier` был выполнен подбор гиперпараметров с помощью `GridSearchCV` и stratified 5-fold cross-validation. В качестве основной метрики использовался `f1_macro`.

Лучший результат после подбора гиперпараметров составил `f1_macro = 0.8050`.

Для сравнения:

- baseline `GradientBoostingClassifier`: `f1_macro = 0.7987`;
- лучший feature engineering эксперимент: `f1_macro = 0.7995`;
- tuned `GradientBoostingClassifier`: `f1_macro = 0.8050`.

Таким образом, feature engineering дал только незначительное улучшение, а основной прирост качества был получен за счет подбора гиперпараметров.

Лучшие параметры модели:

- `learning_rate = 0.08`
- `n_estimators = 200`
- `max_depth = 3`
- `min_samples_leaf = 20`

Полученный результат выглядит реалистичным: качество улучшилось умеренно, без подозрительно резкого скачка. Для финального этапа выбираем tuned `GradientBoostingClassifier` с найденными гиперпараметрами.
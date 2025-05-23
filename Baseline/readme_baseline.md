## Сравнительная таблица моделей (Baseline 1–4)

| №  | Метод                                             | Описание                                                                                  | Доля выборки | Точность на тесте (%) |
|----|---------------------------------------------------|-------------------------------------------------------------------------------------------|--------------|------------------------|
| 1  | PCA + LogisticRegression                          | 64x64 → PCA (150) → LogisticRegression на 15% выборке                                     | 15%          | 26.0                   |
| 2  | PCA + SVM (RBF kernel)                            | 64x64 → PCA (150) → SVM (RBF) на 15% выборке                                              | 15%          | 48.0                   |
| 3  | PCA + RandomForestClassifier                      | 64x64 → PCA (150) → RandomForest на 15% выборке                                           | 15%          | 55.0                   |
| 4  | SIFT + HOG + RandomForest                         | 64x64 → SIFT (128) + HOG → объединение признаков → RandomForest                          | 100%         | 74.0                   |
| 5  | SIFT + HOG + Цветовые статистики + LightGBM       | 64x64 → SIFT + HOG + 48 цветовых статистик + корреляции → LightGBM с Optuna              | 100%         | 97.6                   |
| 6  | VegClassifier (CNN) + без аугментаций + SGD       | 128x128 → CNN (4 блока) + классификатор → SGD + BatchNorm + Dropout                      | 100%         | 98.6                   |
| 7  | Transfer Learning (VGG16)                         | Предобученная VGG16 + классификатор, fine-tuning слоёв                                   | 100%         | 99.5                   |

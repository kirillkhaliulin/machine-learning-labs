# Лабораторная работа № 3. Задание II. Wine

# Подключаем Path для работы с путями к файлам.
from pathlib import Path

# Подключаем pandas под именем pd для загрузки и обработки таблиц.
import pandas as pd

# Подключаем KFold для разбиения выборки на блоки
# и cross_val_score для оценки качества на кросс-валидации.
from sklearn.model_selection import KFold, cross_val_score

# Подключаем классификатор метода k-ближайших соседей.
from sklearn.neighbors import KNeighborsClassifier

# Подключаем scale для масштабирования признаков.
from sklearn.preprocessing import scale

# 1. Загрузите выборку wine.csv
# (https://github.com/tpo9hbi4/MachineDeepLearning)
# с помощью функции read_csv из пакета pandas.
folder = Path(__file__).resolve().parent
wine = pd.read_csv(folder / "wine.csv", header=None)

# 2. Разделите выборку на столбцы признаков и столбец ответов.
# Ответы записаны в 1-ом столбце, признаки —
# в столбцах со 2-ого до последнего.
X = wine.iloc[:, 1:].copy()
y = wine.iloc[:, 0].copy()

# 3. Используя кросс-валидацию по 10 блокам оцените качество.
# Создайте разбиения с перемешиванием выборки:
# shuffle=True и random_state=100.
cv = KFold(n_splits=10, shuffle=True, random_state=100)

# 4. Качество оцените с помощью метрики accuracy,
# используя cross_val_score.
def evaluate_k(X, y, k):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )
    return scores.mean()

# 5. Вычислите качество классификации для метода
# k-ближайших соседей при изменении k от 1 до 100.
# Какому k соответствует лучшее качество и чему оно равно?
quality = pd.Series({
    k: evaluate_k(X, y, k)
    for k in range(1, 101)
})

best_k = quality.idxmax()
best_quality = quality.loc[best_k]

print("Без масштабирования:")
print(f"Лучшее k: {best_k}")
print(f"Средняя accuracy: {best_quality:.6f}")

# 6. Используя sklearn.preprocessing.scale,
# масштабируйте признаки и снова вычислите
# лучшее k на кросс-валидации.
X_scaled = scale(X)

quality_scaled = pd.Series({
    k: evaluate_k(X_scaled, y, k)
    for k in range(1, 101)
})

best_k_scaled = quality_scaled.idxmax()
best_quality_scaled = quality_scaled.loc[best_k_scaled]

# 7. Какому k соответствует лучшее качество
# и чему оно равно для масштабированных признаков?
# Улучшило ли это работу алгоритма?
print("\nПосле масштабирования:")
print(f"Лучшее k: {best_k_scaled}")
print(f"Средняя accuracy: {best_quality_scaled:.6f}")

difference = best_quality_scaled - best_quality

print(f"\nИзменение accuracy: {difference:+.6f}")
print(
    "Масштабирование улучшило качество."
    if difference > 0
    else "Масштабирование не улучшило качество."
)
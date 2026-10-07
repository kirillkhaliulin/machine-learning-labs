# Подключаем NumPy для работы с числовыми массивами
# и создания 300 значений параметра p.
import numpy as np

# Подключаем функцию для загрузки выборки Boston из OpenML.
from sklearn.datasets import fetch_openml

# KFold разбивает выборку на блоки для кросс-валидации.
# cross_val_score обучает и проверяет модель на этих разбиениях.
from sklearn.model_selection import KFold, cross_val_score

# Подключаем модель метода k ближайших соседей
# для предсказания числового значения — стоимости жилья.
from sklearn.neighbors import KNeighborsRegressor

# Подключаем функцию для масштабирования признаков.
from sklearn.preprocessing import scale


# 1. Загрузка выборки Boston
# Загружаем данные. При первой загрузке потребуется интернет.
# as_frame=False позволяет получить данные в виде массивов.
boston = fetch_openml(name="boston", version=1, as_frame=False)

# X — характеристики жилья, используемые для предсказания.
# y — известные значения стоимости жилья.
# Преобразуем данные в массивы с числовым типом float.
X = np.asarray(boston.data, dtype=float)
y = np.asarray(boston.target, dtype=float)


# 2. Масштабирование признаков
# Из значений каждого признака вычитаем его среднее
# и делим результат на стандартное отклонение.
X_scaled = scale(X)


# 3. Создание разбиений для кросс-валидации
# Делим выборку на 10 блоков с предварительным перемешиванием.
# random_state=100 обеспечивает воспроизводимость разбиения.
cv = KFold(
    n_splits=10,
    shuffle=True,
    random_state=100
)


# 4. Перебор 300 значений p от 1 до 20
# Создаём 300 равномерно расположенных значений,
# включая начальное значение 1 и конечное значение 20.
p_values = np.linspace(1, 20, 300)

# В этом списке будем хранить средние оценки
# качества модели для каждого значения p.
mean_scores = []

for p in p_values:
    # Создаём модель с шестью ближайшими соседями.
    # weights="distance" задаёт больший вес более близким соседям.
    # p определяет параметр расстояния Минковского.
    model = KNeighborsRegressor(
        n_neighbors=6,
        weights="distance",
        metric="minkowski",
        p=float(p)
    )

    # Оцениваем модель на 10 разбиениях.
    # neg_mean_squared_error возвращает MSE со знаком минус:
    # чем ближе оценка к нулю, тем меньше ошибка.
    # error_score="raise" позволяет сразу увидеть ошибку,
    # если обучение или проверка модели завершатся неудачно.
    scores = cross_val_score(
        model,
        X_scaled,
        y,
        cv=cv,
        scoring="neg_mean_squared_error",
        error_score="raise"
    )

    # Сохраняем среднюю оценку по 10 проверочным блокам.
    mean_scores.append(scores.mean())


# 5. Выбор лучшего p
# Преобразуем список оценок в массив NumPy.
mean_scores = np.asarray(mean_scores)

# Находим положение максимальной средней оценки.
# Максимум отрицательной MSE соответствует минимуму ошибки.
best_index = np.argmax(mean_scores)

# Получаем лучшее p и соответствующую оценку.
best_p = p_values[best_index]
best_score = mean_scores[best_index]

# Меняем знак оценки, чтобы получить обычную положительную MSE.
best_mse = -best_score


# 6. Вывод результатов
print(f"Количество объектов: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Количество проверенных значений p: {len(p_values)}")

print("\nРезультаты подбора параметра:")
print(f"Лучшее p: {best_p:.6f}")
print(f"Средняя оценка neg_mean_squared_error: {best_score:.6f}")
print(f"Средняя MSE: {best_mse:.6f}")



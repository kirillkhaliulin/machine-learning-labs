# Лабораторная работа № 3. Задание I. Titanic

# Подключаем Path для работы с путями к файлам.
from pathlib import Path

# Подключаем pandas под именем pd для загрузки и обработки таблиц.
import pandas as pd

# Подключаем DecisionTreeClassifier для создания
# и последующего обучения модели дерева решений.
from sklearn.tree import DecisionTreeClassifier

# 1. Загрузите выборку titanic.scv
# (https://github.com/tpo9hbi4/MachineDeepLearning)
# с помощью функции read_csv из пакета pandas.
folder = Path(__file__).resolve().parent
titanic = pd.read_csv(folder / "Титаник.csv")

# 2. Отберите из выборки следующие признаки: Pclass, Fare, Age и Sex.
X = titanic[["Pclass", "Fare", "Age", "Sex"]].copy()

# 3. При присутствии строковых значений замените их на числовые.
X["Sex"] = X["Sex"].map({"male": 0, "female": 1})

# 4. Создайте столбец ответов — Survived.
y = titanic["Survived"].copy()

# 5. Пропущенные значения в выборке нужно удалить.
# Для обнаружения пропущенных значений имеется функция np.isnan.
# Здесь для проверки пропусков используем метод pandas notna.
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows].copy()
y = y.loc[valid_rows].copy()

# 6. Используя DecisionTreeClassifier постройте модель,
# где все параметры будут использоваться по умолчанию,
# кроме параметра random_state, его значение задайте равным 100.
model = DecisionTreeClassifier(random_state=100)
model.fit(X, y)

# 7. Найдите два самых важных признака.
# В ответе к задаче укажите их.
importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Два наиболее важных признака:")
print(importances.head(2))
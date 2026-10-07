# Подключаем Matplotlib для построения двух графиков:
# с реальными классами и с предсказанными классами.
import matplotlib.pyplot as plt

# Подключаем встроенную выборку с характеристиками ирисов.
from sklearn.datasets import load_iris

# Подключаем PCA для отображения четырёх признаков
# в виде двух координат на графике.
from sklearn.decomposition import PCA

# Подключаем функцию для разделения данных
# на обучающую и проверочную выборки.
from sklearn.model_selection import train_test_split

# Подключаем дерево решений для классификации ирисов.
from sklearn.tree import DecisionTreeClassifier

# Подключаем метрику доли правильных предсказаний.
from sklearn.metrics import accuracy_score


# 1. Загрузка выборки Iris
iris = load_iris()

# X — четыре характеристики каждого цветка.
X = iris.data

# y — известный класс каждого цветка.
y = iris.target

# Названия трёх классов ирисов.
class_names = iris.target_names


# Проверяем, что выборка загрузилась.
print(f"Количество объектов: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")
print("Названия классов:", ", ".join(class_names))

# 2. Разделение данных на обучающую и проверочную выборки
# 25% объектов используем для проверки, остальные для обучения.
# random_state=241 фиксирует разбиение, как в методичке.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=241
)


# 3. Создание и обучение дерева решений
# Фиксируем random_state для воспроизводимости результата.
model = DecisionTreeClassifier(random_state=241)

# Обучаем дерево на четырёх исходных признаках.
model.fit(X_train, y_train)


# 4. Предсказание классов проверочной выборки
# Эти объекты не использовались для обучения дерева.
y_pred = model.predict(X_test)


# 5. Оценка качества классификации
# Сравниваем предсказанные классы с известными ответами.
accuracy = accuracy_score(y_test, y_pred)

# Подсчитываем количество правильных ответов и ошибок.
correct_count = (y_test == y_pred).sum()
error_count = (y_test != y_pred).sum()

print(f"\nОбъектов для обучения: {X_train.shape[0]}")
print(f"Объектов для проверки: {X_test.shape[0]}")
print(f"Правильных предсказаний: {correct_count}")
print(f"Ошибочных предсказаний: {error_count}")
print(f"Accuracy: {accuracy:.6f}")
print(f"Доля правильных предсказаний: {accuracy:.2%}")

# 6. Получение двух координат для отображения данных
# Настраиваем PCA только на обучающей выборке.
pca = PCA(n_components=2)
pca.fit(X_train)

# Преобразуем проверочные объекты в двумерные координаты.
# Для обоих графиков используем одни и те же координаты.
X_test_2d = pca.transform(X_test)


# 7. Создание двух графиков
# sharex и sharey обеспечивают одинаковые шкалы осей.
fig, axes = plt.subplots(
    1,
    2,
    figsize=(13, 5),
    sharex=True,
    sharey=True
)

# Каждому классу назначаем постоянный цвет.
colors = ["blue", "green", "red"]

# На первом графике используем реальные классы,
# на втором графике используем предсказанные классы.
labels_for_plots = [y_test, y_pred]
titles = ["Реальные классы", "Предсказанные классы"]

for ax, labels, title in zip(axes, labels_for_plots, titles):
    for class_id, class_name in enumerate(class_names):
        # Выбираем объекты соответствующего класса.
        mask = labels == class_id

        ax.scatter(
            X_test_2d[mask, 0],
            X_test_2d[mask, 1],
            color=colors[class_id],
            label=class_name,
            s=60,
            edgecolors="black"
        )

    ax.set_title(title)
    ax.set_xlabel("Первая главная компонента")
    ax.set_ylabel("Вторая главная компонента")
    ax.legend(title="Класс ириса")
    ax.grid(alpha=0.3)


# 8. Выделение ошибочных предсказаний
# На правом графике обводим ошибочно классифицированные
# объекты дополнительными чёрными окружностями.
error_mask = y_test != y_pred

axes[1].scatter(
    X_test_2d[error_mask, 0],
    X_test_2d[error_mask, 1],
    s=180,
    facecolors="none",
    edgecolors="black",
    linewidths=1.5,
    label="Ошибка классификации"
)

axes[1].legend(title="Класс ириса")

fig.suptitle(
    f"Классификация ирисов деревом решений: "
    f"accuracy = {accuracy:.2%}"
)

fig.tight_layout()
plt.show()
# Лабораторная работа № 4. Задание II. Метод SVD

# Подключаем Path для работы с путями к файлам.
from pathlib import Path

# Подключаем NumPy для работы с числовыми массивами.
import numpy as np

# Подключаем pandas под именем pd для загрузки и обработки таблиц.
import pandas as pd

# Подключаем LabelEncoder для перенумерации пользователей и фильмов
# последовательными номерами с 0.
from sklearn.preprocessing import LabelEncoder

# Подключаем coo_matrix для создания разреженной матрицы рейтингов
# и svds для её сингулярного разложения.
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds

# Подключаем NearestNeighbors для поиска ближайших фильмов.
from sklearn.neighbors import NearestNeighbors

# Подключаем cosine_similarity для расчёта косинусной схожести.
from sklearn.metrics.pairwise import cosine_similarity

# Подключаем pdist для попарного расчёта расстояний
# и squareform для преобразования результата в квадратную матрицу.
from scipy.spatial.distance import pdist, squareform


# 1. Используя программу, реализующую метод SVD, приведённую
# в п. 4.3, найти и распечатать наиболее близкие фильмы
# к фильму «Терминатор 2».

# Загружаем таблицу рейтингов и таблицу характеристик фильмов.
folder = Path(__file__).resolve().parent
df_rates = pd.read_csv(folder / "user_ratedmovies.dat", sep="\t")
df_movies = pd.read_csv(folder / "movies.dat", sep="\t",
                        encoding="iso-8859-1")

# Перенумеровываем пользователей и фильмы.
enc_user = LabelEncoder().fit(df_rates.userID.values)
enc_mov = LabelEncoder().fit(df_rates.movieID.values)

# Оставляем только фильмы, которые оценивали пользователи.
idx = df_movies.loc[:, "id"].isin(df_rates.movieID)
df_movies = df_movies.loc[idx].copy()

df_rates["userID"] = enc_user.transform(df_rates["userID"].values)
df_rates["movieID"] = enc_mov.transform(df_rates["movieID"].values)
df_movies["id"] = enc_mov.transform(df_movies["id"].values)

# Строим матрицу рейтингов R: строки — пользователи,
# столбцы — фильмы, значения — оценки.
R = coo_matrix((df_rates.rating.values,
                (df_rates.userID.values, df_rates.movieID.values)))

# Выполняем сингулярное разложение с 6 компонентами.
u, s, vt = svds(R, k=6)

# Ищем по 10 ближайших фильмов в пространстве компонент SVD.
# Первым в списке находится сам фильм.
v = vt.T
nn = NearestNeighbors(n_neighbors=10)
nn.fit(v)
_, ind = nn.kneighbors(v, n_neighbors=10)

movie_titles = df_movies.sort_values("id").loc[:, "title"].values
cols = ["movie"] + [f"nn_{i}" for i in range(1, 10)]
df_ind_nn = pd.DataFrame(data=movie_titles[ind], columns=cols)

# Выводим фильмы, ближайшие к «Терминатору 2».
idx = df_ind_nn.movie.str.contains("Terminator 2")
print("Фильмы, ближайшие к «Терминатору 2»:")
print(df_ind_nn.loc[idx].T.to_string(header=False))


# 2. Найти ближайших пользователей по схожим интересам
# к пользователю № 1.

# А) Работаем с матрицей рейтингов R.
R_array = R.toarray()

# Б) По матрице R рассчитываем косинусную схожесть
# и выводим её размерность.
similarity = cosine_similarity(R)
print("\nРазмерность матрицы cosine_similarity:", similarity.shape)


# В) Скорректированная мера схожести.
# Находим фильмы, которые оценили оба пользователя.
# Если пересечение не пустое, считаем косинус между оценками
# на этих фильмах, берём его с минусом и добавляем единицу.
# Если общих фильмов нет, пользователи считаются несхожими.
def corrected_distance(u_ratings, v_ratings):
    common = (u_ratings > 0) & (v_ratings > 0)
    if not common.any():
        return 1.0
    a = u_ratings[common]
    b = v_ratings[common]
    cosine = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return 1 - cosine


# Г) Рассчитываем попарные расстояния методом pdist,
# передавая ему матрицу R.toarray() и написанную функцию.
distances = pdist(R_array, metric=corrected_distance)

# Д) Выводим размерность полученного вектора
# и преобразуем его в квадратную матрицу с помощью squareform.
print("Размерность вектора pdist:", distances.shape)
distance_matrix = squareform(distances)
print("Размерность матрицы squareform:", distance_matrix.shape)

# Ближайшие пользователи к пользователю № 1
# (первая строка матрицы, исходный userID определяем обратно).
user_index = 0
user_id = enc_user.inverse_transform([user_index])[0]

# Значения, отличающиеся от нуля только из-за погрешности
# вычислений, округляем до 0.
distances_to_user = np.clip(distance_matrix[user_index], 0, None).round(6)
common_counts = ((R_array > 0) & (R_array[user_index] > 0)).sum(axis=1)

# Сортируем пользователей по расстоянию, а при равном расстоянии —
# по количеству общих фильмов (чем больше, тем надёжнее схожесть).
result = pd.DataFrame({
    "userID": enc_user.classes_,
    "Расстояние": distances_to_user,
    "Общих фильмов": common_counts,
}).drop(index=user_index)
result = result.sort_values(["Расстояние", "Общих фильмов"],
                            ascending=[True, False]).head(10)

print(f"\nПользователи, ближайшие к пользователю № 1 "
      f"(userID = {user_id}):")
print(result.to_string(index=False))

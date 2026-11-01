# Лабораторная работа № 4. Задание I. Apriori и FP-growth

# Подключаем Path для работы с путями к файлам.
from pathlib import Path

# Подключаем time для измерения времени работы алгоритмов.
import time

# Подключаем multiprocessing, чтобы запускать каждый расчёт
# в отдельном процессе и останавливать его по ограничению времени.
import multiprocessing as mp

# Подключаем NumPy для вычисления натурального логарифма времени.
import numpy as np

# Подключаем pandas под именем pd для работы с таблицами.
import pandas as pd

# Подключаем Matplotlib для построения графиков.
import matplotlib.pyplot as plt

# Подключаем TransactionEncoder для преобразования транзакций
# в таблицу, а также реализации алгоритмов Apriori и FP-growth.
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth


# Шесть выборок из репозитория практикума
# (https://github.com/tpo9hbi4/MachineDeepLearning).
DATASETS = [
    "c10d1к.csv",
    "c20d10k.csv",
    "С73D10к.csv",
    "T10I4D100К.csv",
    "T25I10D10К.csv",
    "T40I10D100K.csv",
]

# Значения минимальной поддержки из таблицы задания, в процентах.
SUPPORTS = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# Если алгоритм работает более 5 минут, расчёт останавливается,
# а меньшие значения поддержки не проверяются.
TIME_LIMIT = 300

folder = Path(__file__).resolve().parent


# Загрузка выборки: каждая строка — одна транзакция,
# предметы перечислены через запятую.
def load_transactions(file_name):
    text = (folder / file_name).read_text(encoding="utf-8")
    transactions = []
    for line in text.splitlines():
        items = [item for item in line.strip().split(",") if item]
        if items:
            transactions.append(items)
    return transactions


# Преобразование транзакций в таблицу True/False:
# строки — транзакции, столбцы — предметы.
def encode(transactions):
    encoder = TransactionEncoder()
    array = encoder.fit(transactions).transform(transactions)
    return pd.DataFrame(array, columns=encoder.columns_)


# Запуск одного алгоритма с измерением времени.
# Результат передаётся в основной процесс через очередь.
def run_algorithm(name, file_name, support, queue):
    data = encode(load_transactions(file_name))
    start = time.perf_counter()
    if name == "Apriori":
        itemsets = apriori(data, min_support=support / 100,
                           use_colnames=True, low_memory=True)
    else:
        itemsets = fpgrowth(data, min_support=support / 100,
                            use_colnames=True)
    queue.put((time.perf_counter() - start, len(itemsets)))


# Запуск расчёта с ограничением по времени.
def measure(name, file_name, support):
    queue = mp.Queue()
    process = mp.Process(target=run_algorithm,
                         args=(name, file_name, support, queue))
    process.start()
    process.join(TIME_LIMIT + 60)
    if process.is_alive():
        process.terminate()
        process.join()
        return None, None
    if queue.empty():
        return None, None
    seconds, count = queue.get()
    if seconds > TIME_LIMIT:
        return None, None
    return seconds, count


if __name__ == "__main__":
    all_tables = {}

    # 1. Выполнить алгоритм Apriori на разных выборках.
    # 2. Выполнить алгоритм FP-growth на разных выборках.
    # Поддержку уменьшаем от 90% до 1%; после превышения
    # 5 минут меньшие значения поддержки не проверяются.
    for file_name in DATASETS:
        table = pd.DataFrame(index=SUPPORTS)
        table.index.name = "Поддержка, %"

        for name in ["Apriori", "FP-growth"]:
            stopped = False
            for support in sorted(SUPPORTS, reverse=True):
                if stopped:
                    seconds, count = None, None
                else:
                    seconds, count = measure(name, file_name, support)
                    stopped = seconds is None
                table.loc[support, f"Время {name}, с"] = seconds
                table.loc[support, f"Наборов {name}"] = count

        # 3. Результаты по времени работы алгоритмов
        # заносим в таблицу.
        for name in ["Apriori", "FP-growth"]:
            column = f"Наборов {name}"
            table[column] = table[column].astype("Int64")
        all_tables[file_name] = table
        print(f"\nВыборка {Path(file_name).stem}")
        print(table.to_string(float_format=lambda v: f"{v:.4f}",
                              na_rep="> 300 с"))

    # Сохраняем все таблицы в один файл.
    with pd.ExcelWriter(folder / "lab4_1_results.xlsx") as writer:
        for file_name, table in all_tables.items():
            table.to_excel(writer, sheet_name=Path(file_name).stem)

    # 4. По шести таблицам строим шесть графиков зависимости:
    # по горизонтали — поддержка (%), по вертикали — Ln(t).
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (file_name, table) in zip(axes.flat, all_tables.items()):
        for name, marker in [("Apriori", "o"), ("FP-growth", "s")]:
            seconds = table[f"Время {name}, с"].astype(float)
            ax.plot(table.index, np.log(seconds), marker=marker,
                    label=name)
        ax.set_title(f"Выборка {Path(file_name).stem}")
        ax.set_xlabel("Поддержка, %")
        ax.set_ylabel("Ln(t)")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(folder / "lab4_1_graphs.png", dpi=150)
    plt.show()

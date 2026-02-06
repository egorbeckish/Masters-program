import numpy as np
import os


def MonteCarlo2d(ab: np.array, N: int, f, **kwargs):
    dots = {
        "x": np.random.uniform(*ab[0], N),
        "y": np.random.uniform(*ab[1], N)
    }

    return (abs(ab[0][0] - ab[0][1]) * abs(ab[1][0] - ab[1][1])) * np.mean(f(**dots))


if __name__ == "__main__":
    data = [
        {"ab": np.array([[0, 1], [0, 1]]), "N": 10 ** 4, "f": lambda x, y: x + y, "f_text": "x + y", "manual_result": 1},
    ]

    for parametrs in data:
        obj = MonteCarlo2d(**parametrs)
        print(
			f"Интервал: [{parametrs["ab"]}",
			f"N = {parametrs["N"]}",
			f"Метод решения: mm",
			f"Функция: {parametrs["f_text"]}",
			f"Оценка: {obj}",
			f"Ручной расчет: {parametrs["manual_result"]}",
			f"Погрешность: {abs(obj - parametrs["manual_result"])} (+-{obj / parametrs["manual_result"]} %)",
			"-" * 70,
			sep="\n",
			file=open(f"{os.path.dirname(__file__)}/montecarlo.txt", "a", encoding="utf-8")
		)

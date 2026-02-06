from monte_carlo import (
	MonteCarlo, 
	os, 
	np
)



if __name__ == "__main__":
	data = [
		{"a": 0, "b": 1, "N": 10 ** 2, "f": lambda x: x ** .5, "f_text": "x^(1/2)", "manual_result": 2/3},
		{"a": 0, "b": 1, "N": 10 ** 3, "f": lambda x: x ** .5, "f_text": "x^(1/2)", "manual_result": 2/3},
		{"a": 0, "b": 1, "N": 10 ** 4, "f": lambda x: x ** .5, "f_text": "x^(1/2)", "manual_result": 2/3},
		{"a": 0, "b": 1, "N": 10 ** 5, "f": lambda x: x ** .5, "f_text": "x^(1/2)", "manual_result": 2/3},
		{"a": 0, "b": np.pi / 2, "N": 10 ** 2, "f": lambda x: np.sin(x) ** 2, "f_text": "sin^2", "manual_result": np.pi/4},
		{"a": 0, "b": np.pi / 2, "N": 10 ** 3, "f": lambda x: np.sin(x) ** 2, "f_text": "sin^2", "manual_result": np.pi/4},
		{"a": 0, "b": np.pi / 2, "N": 10 ** 4, "f": lambda x: np.sin(x) ** 2, "f_text": "sin^2", "manual_result": np.pi/4},
		{"a": 0, "b": np.pi / 2, "N": 10 ** 5, "f": lambda x: np.sin(x) ** 2, "f_text": "sin^2", "manual_result": np.pi/4},
		{"a": -1, "b": 1, "N": 10 ** 2, "f": lambda x: x ** 3 + 1, "f_text": "x^3 + 1", "manual_result": 2},
		{"a": -1, "b": 1, "N": 10 ** 3, "f": lambda x: x ** 3 + 1, "f_text": "x^3 + 1", "manual_result": 2},
		{"a": -1, "b": 1, "N": 10 ** 4, "f": lambda x: x ** 3 + 1, "f_text": "x^3 + 1", "manual_result": 2},
		{"a": -1, "b": 1, "N": 10 ** 5, "f": lambda x: x ** 3 + 1, "f_text": "x^3 + 1", "manual_result": 2},
		{"a": 0, "b": 1, "N": 10 ** 2, "f": lambda x: 1 / (1 + x), "f_text": "1 / 1 + x", "manual_result": np.log(2)},
		{"a": 0, "b": 1, "N": 10 ** 3, "f": lambda x: 1 / (1 + x), "f_text": "1 / 1 + x", "manual_result": np.log(2)},
		{"a": 0, "b": 1, "N": 10 ** 4, "f": lambda x: 1 / (1 + x), "f_text": "1 / 1 + x", "manual_result": np.log(2)},
		{"a": 0, "b": 1, "N": 10 ** 5, "f": lambda x: 1 / (1 + x), "f_text": "1 / 1 + x", "manual_result": np.log(2)},
	]

	for parametrs in data:
		obj = MonteCarlo(**parametrs)
		result = obj.estimate
		print(
			f"Интервал: [{parametrs["a"]}, {parametrs["b"]}] -> [{obj.a}, {obj.b}]",
			f"N = {obj.N}",
			f"Метод решения: {obj.method}",
			f"Функция: {parametrs["f_text"]}",
			f"Оценка: {result}",
			f"Ручной расчет: {parametrs["manual_result"]}",
			f"Погрешность: {abs(result - parametrs["manual_result"])} (+-{result / parametrs["manual_result"]} %)",
			"-" * 70,
			sep="\n",
			file=open(f"{os.path.dirname(__file__)}/individual.txt", "a", encoding="utf-8")
		)

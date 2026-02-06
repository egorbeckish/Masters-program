import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Literal
import os


class MonteCarlo:
	def __init__(self, a: int | float, b: int | float, N: int, f, method: Literal["mm", "mam"]="mm", **kwargs):
		"""
		Docstring for __init__
		
		:param self: Description
		:param a: Description
		:type a: int | float
		:param b: Description
		:type b: int | float
		:param N: Description
		:type N: int
		:param f: Description
		:param method: "mm" - method middle, "mam" - method aimmis
		:type method: Literal["gm", "mm", "mam"]
		"""

		self.a = a
		self.b = b
		self.N = N
		self.f = f
		self.method = method
		self.dots = self.random
		self.f_x = self.f(self.dots)


	@property
	def random(self) -> np.ndarray:
		if self.a == float("-inf"):
			self.a = 0 if self.b != 0 else 1
		
		if self.b == float("inf"):
			self.b = 0 if self.a != 0 else 1

		return np.random.uniform(self.a, self.b, self.N)
    

	@property	
	def __mm(self) -> float:
		return abs(self.b - self.a) * np.mean(self.f_x)
	

	@property
	def __mam(self) -> float:
		M = self.dots.max()
		y = np.random.uniform(0, M, self.N)
		return (len(self.f_x[y <= self.f_x]) / self.N) * (abs(self.b - self.a) * M)

	
	@property
	def get_method(self) -> float:
		return {
			"mm": self.__mm,
			"mam": self.__mam,
		}[self.method]


	@property
	def estimate(self) -> float:
		return self.get_method	
	

	def draw_frequency(self, count: int):
		estimate_frequency = []
		for _ in range(count):
			obj = MonteCarlo(self.a, self.b, self.N, self.f)
			estimate_frequency += [obj.estimate]
		
		
		sns.displot(
			data=pd.DataFrame(estimate_frequency, columns=["Estimate"]),
			x="Estimate",
			kde=True,
			bins=30
		)

		# plt.hist(estimate_frequency, 30, ec="k")
		# plt.xlabel("Estimate")
		# plt.ylabel("Count")
		plt.title(f"{self.N = }")
		plt.show()


if __name__ == "__main__":
	data = [
		{"a": 0, "b": 1, "N": 10 ** 4, "f": lambda x: x ** 3, "f_text": "x^3", "manual_result": .25},
		{"a": 0, "b": np.pi, "N": 10 ** 4, "f": np.sin, "f_text": "sin", "manual_result": 2},
		{"a": 0, "b": 1, "N": 10 ** 4, "f": lambda x: 1 / (1 - x ** 2) ** .5, "f_text": "1 / (1 - x^2)^.5", "manual_result": np.pi / 2},
		{"a": 0, "b": 2, "N": 10 ** 4, "f": lambda x: np.exp(-x ** 2), "method": "mam", "f_text": "e^(-x^2)", "manual_result": .88208},
		{"a": 0, "b": 2, "N": 10 ** 4, "f": lambda x: np.exp(-x ** 2), "f_text": "e^(-x^2)", "manual_result": .88208},
		{"a": 1, "b": 4, "N": 10 ** 4, "f": lambda x: np.log(x), "f_text": "e^(-x^2)", "manual_result": 2.5451},
		{"a": -1, "b": 2, "N": 10 ** 4, "f": lambda x: np.where(x < 1, x ** 2, 2 * x), "f_text": "x^2 if x < 1; 2 * x if x >= 1", "manual_result": 11/3},
		{"a": 0, "b": 1, "N": 10 ** 2, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
		{"a": 0, "b": 1, "N": 10 ** 3, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
		{"a": 0, "b": 1, "N": 10 ** 4, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
		{"a": 0, "b": 1, "N": 10 ** 5, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
		{"a": 0, "b": 1, "N": 10 ** 6, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
		{"a": 1, "b": float("inf"), "N": 10 ** 4, "f": lambda x: x, "f_text": "1 / x^3", "manual_result": .5},
		{"a": 0, "b": 1, "N": 10 ** 4, "f": lambda x: x ** .5 - x ** 2, "f_text": "x^.5 - x^2", "manual_result": 1/3},
		{"a": 0, "b": 2 * np.pi, "N": 10 ** 4, "f": lambda x: np.cos(x) ** 2, "f_text": "cos^2", "manual_result": np.pi},
		{"a": 2, "b": 5, "N": 10 ** 4, "f": lambda x: x / 3, "f_text": "x/3", "manual_result": 3.5},
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
			file=open(f"{os.path.dirname(__file__)}/montecarlo.txt", "a", encoding="utf-8")
		)

import numpy as np
from typing import Literal
import time
import os


def timer(func):
	def wrapper(*args, **kwargs):
		start = time.time()
		f = func(*args, **kwargs)
		finish = time.time()

		print(
			f"Время работа алгоритма: {finish - start}",
			"-" * 70,
			sep="\n",
			file=open(f"{os.path.dirname(__file__)}/results.txt", "a", encoding="utf-8")
		)
		return f
	return wrapper


class KaczmarzMethod:
	
	def __init__(self, A: np.ndarray, b: np.ndarray, lambda_: float=1.0, x0: np.ndarray=None, max_iter: int=10 ** 4, eps: float=1e-4, method: Literal["s", "r"]="s"):
		"""
		Docstring for __init__
        
		:param self: Description
		:param A: Description
		:type A: np.ndarray
		:param b: Description
		:type b: np.ndarray
		:param lambda_: Description
		:type lambda_: float
		:param x0: initial approximation
		:type x0: np.ndarray
		:param max_iter: number of iterations
		:type max_iter: int
		:param eps: accuracy
		:type eps: float
		:param method: s - standart method, r - randomized method
		:type method: Literal["s", "r"]
		"""

		self.A: np.ndarray = A
		self.b: np.ndarray = b
		self.lambda_ = lambda_
		self.n, self.m = A.shape
		self.x0: np.ndarray = x0 if x0 is not None else np.zeros(self.m)
		self.max_iter: int = max_iter
		self.eps: float = eps
		self.method: str = method


	def __get_index(self, iter, probs) -> int:
		if self.method == "s":
			return iter % self.n
		
		return np.random.choice(self.n, p=probs)


	@timer
	def solve(self) -> None:
		norms = np.sum(A ** 2, axis=1)
		probs = norms / sum(norms)
		param = 0
		
		for iter in range(self.max_iter):
			prev = self.x0.copy()

			if iter % self.n == 0:
				param += 1
				if np.linalg.norm(self.A @ self.x0 - self.b) < self.eps:
					break

			j = self.__get_index(iter, probs)
			projection = self.b[j] - np.dot(self.A[j], self.x0)
			self.x0 += (projection / norms[j]) * self.A[j] * self.lambda_			
		
		print(
			f"Матрица размера {self.n}х{self.m}",
			f"Выбран метод: {self.method}",
			f"Lambda: {self.lambda_}",
			f"Кол-во итераций: {iter + 1}",
			f"Ошибка: {np.linalg.norm(self.x0 - prev)}",
			f"Норма вектора невязки: {np.linalg.norm(self.A @ self.x0 - self.b)}",
			f"Параметр регулеризации: {param}\n",
			sep="\n",
			file=open(f"{os.path.dirname(__file__)}/results.txt", "a", encoding="utf-8")
		)

		#print(f"Вектора невязки:\n{self.A @ self.x0 - self.b}")


if __name__ == "__main__":
	np.random.seed(42)

	n, m = 100, 100
	A = np.random.randn(n, m)
	x = np.random.randn(m)
	b = A @ x

	for method in ["s", "r"]:
		obj = KaczmarzMethod(
			A,
			b,
			max_iter=10 ** 6,
			eps=1e-8,
			method=method
		)

		obj.solve()

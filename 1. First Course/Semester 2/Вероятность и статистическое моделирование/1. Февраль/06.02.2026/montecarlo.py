import numpy as np
from typing import Literal
import os


class MonteCarlo:

    def __init__(self, ab: tuple | list | np.ndarray, N: int, f, method: Literal["mm", "mam"]="mm", **kwargs):
        """
        Docstring for __init__
        
        :param self: Description
        :param ab: Description
        :type ab: tuple | list | np.ndarray
        :param N: Description
        :type N: int
        :param f: Description
        :param method: "mm" - method middle, "mam" - method aimmis
        :type method: Literal["mm", "mam"]
        """

        self.ab = ab
        self.N = N
        self.f = f
        self.method = method


    @property
    def __mm(self):
        if isinstance(self.ab, tuple):
            S = abs(self.ab[0] - self.ab[1])
        
        else:
            S = 1
            for ab in self.ab:
                S *= abs(ab[0] - ab[1])
        
        mean = np.mean(self.f(*self.random))
        
        return S * mean


    @property
    def __mam(self):
        random = self.random
        f: np.ndarray = self.f(*random)
        M = f.max()
        y = np.random.uniform(0, M, self.N)
        M_N = len(f[y <= f])
        
        if isinstance(self.ab, tuple):
            S = abs(self.ab[0] - self.ab[1]) * M
        
        else:
            S = M
            for i in range(len(self.ab)):
                S *= abs(self.ab[i][0] - self.ab[i][1])
            
        aim = M_N / self.N

        return aim * S

    
    @property
    def get_method(self):
        return {
            "mm": self.__mm,
            "mam": self.__mam
        }[self.method]
    

    @property
    def random(self) -> np.ndarray | list[np.ndarray]:
        if isinstance(self.ab, tuple):
            a, b = self.ab
            return np.random.uniform(a, b, self.N).reshape(1, -1)
        
        return (np.random.uniform(ab[0], ab[1], self.N) for ab in self.ab)
    

    @property
    def estimate(self) -> float:
        return self.get_method



if __name__ == "__main__":
    data = [
	    {"ab": (0, 1), "N": 10 ** 4, "f": lambda x: x ** 3, "f_text": "x^3", "manual_result": .25},
	    {"ab": (0, np.pi), "N": 10 ** 4, "f": np.sin, "f_text": "sin", "manual_result": 2},
	    {"ab": (0, 1), "N": 10 ** 4, "f": lambda x: 1 / (1 - x ** 2) ** .5, "f_text": "1 / (1 - x^2)^.5", "manual_result": np.pi / 2},
	    {"ab": [[0, 1], [0, 1]], "N": 10 ** 4, "f": lambda x, y: x + y, "f_text": "x + y", "manual_result": 1},
        {"ab": (0, 2), "N": 10 ** 4, "f": lambda x: np.exp(-x ** 2), "method": "mam", "f_text": "e^(-x^2)", "manual_result": .88208},
	    {"ab": (0, 2), "N": 10 ** 4, "f": lambda x: np.exp(-x ** 2), "f_text": "e^(-x^2)", "manual_result": .88208},
	    {"ab": (1, 4), "N": 10 ** 4, "f": lambda x: np.log(x), "f_text": "e^(-x^2)", "manual_result": 2.5451},
	    {"ab": (-1, 2), "N": 10 ** 4, "f": lambda x: np.where(x < 1, x ** 2, 2 * x), "f_text": "x^2 if x < 1; 2 * x if x >= 1", "manual_result": 11/3},
        {"ab": [[0, 1], [0, 1], [0, 1]], "N": 10 ** 4, "f": lambda x, y, z: x * y * z, "f_text": "xyz", "manual_result": .125},
	    {"ab": (0, 1), "N": 10 ** 2, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
	    {"ab": (0, 1), "N": 10 ** 3, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
	    {"ab": (0, 1), "N": 10 ** 4, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
	    {"ab": (0, 1), "N": 10 ** 5, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
	    {"ab": (0, 1), "N": 10 ** 6, "f": lambda x: np.exp(x), "f_text": "e^x", "manual_result": 1.7182},
	    {"ab": (0, 1), "N": 10 ** 4, "f": lambda x: x, "f_text": "1 / x^3", "manual_result": .5},
	    {"ab": (0, 1), "N": 10 ** 4, "f": lambda x: x ** .5 - x ** 2, "f_text": "x^.5 - x^2", "manual_result": 1/3},
	    {"ab": (0, 2 * np.pi), "N": 10 ** 4, "f": lambda x: np.cos(x) ** 2, "f_text": "cos^2", "manual_result": np.pi},
	    {"ab": (2, 5), "N": 10 ** 4, "f": lambda x: x / 3, "f_text": "x/3", "manual_result": 3.5},
    ]

    os.makedirs("results", exist_ok=True)
    for parametrs in data:
        obj = MonteCarlo(**parametrs)
        result = obj.estimate
        print(
            f"Интервал: {parametrs["ab"]}",
            f"N = {obj.N}",
            f"Метод решения: {obj.method}",
            f"Функция: {parametrs["f_text"]}",
            f"Оценка: {result}",
            f"Ручной расчет: {parametrs["manual_result"]}",
            f"Погрешность: {abs(result - parametrs["manual_result"])} (+-{result / parametrs["manual_result"]} %)",
            "-" * 70,
            sep="\n",
            file=open(f"{os.path.dirname(__file__)}/results/montecarlo.txt", "a", encoding="utf-8")
        )

class PseudorandomNumberGenerator:
    def __init__(self, n: int | float):
        self.n = n
    

    def middle(self, n: int) -> int:
        n = str(n)
        lenght = len(n)
        n = "0" * (lenght % 2) + n
        lenght = len(n)
        return int(n[lenght // 2 - 2:lenght // 2 + 2])
    

    def random(self, count: int) -> list[int]:
        random_numbers = []
        n = self.n

        for _ in range(count):
            n **= 2
            n = self.middle(n)

            if n == 0:
                print(f"Недостатком данного метода является ограниченность множества ПСЧ из-за того, что последовательность зацикливается: {n = }")
                break
        
            random_numbers += [n]
        
        return random_numbers
    

if __name__ == "__main__":
    # a = PseudorandomNumberGenerator(28696953729)
    a = PseudorandomNumberGenerator(1234)
    print(a.random(100000))

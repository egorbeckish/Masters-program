import numpy as np
import math
# from collections import defaultdict, Counter

class NaiveBayesClassifier:
    def __init__(self, distribution='g'):
        self.distribution = distribution
        self.classes = None
        self.class_priors = {}  # Априорные вероятности классов
        self.class_stats = {}   # Статистики для каждого класса
        
    def fit(self, X, y):
        """
        Обучение классификатора
        X: список признаков (каждый признак - список чисел)
        y: список меток классов
        """
        self.classes = np.unique(y)
        
        # Вычисляем априорные вероятности классов
        total_samples = len(y)
        for c in self.classes:
            class_samples = X[y == c]
            self.class_priors[c] = len(class_samples) / total_samples
            
            # Вычисляем статистики в зависимости от распределения
            if self.distribution == 'g':
                self.class_stats[c] = {
                    'mean': np.mean(class_samples, axis=0),
                    'std': np.std(class_samples, axis=0) + 1e-9  # добавляем маленькое число чтобы избежать деления на 0
                }
            elif self.distribution == 'b':
                self.class_stats[c] = {
                    'p': np.mean(class_samples, axis=0)
                }
    
    def _gaussian_probability(self, x, mean, std):
        """Вычисление вероятности по Гауссовскому распределению"""
        exponent = math.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (math.sqrt(2 * math.pi) * std)) * exponent
    
    def _bernoulli_probability(self, x, p):
        """Вычисление вероятности по распределению Бернулли"""
        return p ** x * (1 - p) ** (1 - x)
    
    def _predict_sample(self, sample):
        """Предсказание для одного примера"""
        class_probabilities = {}
        
        for c in self.classes:
            # Начинаем с логарифма априорной вероятности (для избежания underflow)
            probability = math.log(self.class_priors[c])
            
            # Добавляем логарифмы условных вероятностей для каждого признака
            for feature_idx, feature_value in enumerate(sample):
                if self.distribution == 'g':
                    mean = self.class_stats[c]['mean'][feature_idx]
                    std = self.class_stats[c]['std'][feature_idx]
                    feature_prob = self._gaussian_probability(feature_value, mean, std)
                elif self.distribution == 'b':
                    p = self.class_stats[c]['p'][feature_idx]
                    feature_prob = self._bernoulli_probability(feature_value, p)
                probability += math.log(feature_prob + 1e-9)  # добавляем маленькое число для избежания log(0)
            
            class_probabilities[c] = probability
        
        # Возвращаем класс с максимальной вероятностью
        return max(class_probabilities, key=class_probabilities.get)
    
    def predict(self, X):
        """Предсказание для нескольких примеров"""
        return [self._predict_sample(sample) for sample in X]
    
    def score(self, X, y):
        """Оценка точности классификатора"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Пример использования
if __name__ == "__main__":
    # Создаем простой набор данных
    # Признаки: [длина, ширина]
    X_train = np.array([
        [1.0, 1.2],  # класс A
        [1.2, 1.1],  # класс A
        [1.1, 1.3],  # класс A
        [3.5, 3.2],  # класс B
        [3.3, 3.4],  # класс B
        [3.6, 3.1],  # класс B
        [5.1, 5.2],  # класс C
        [5.3, 5.0],  # класс C
        [5.0, 5.1],  # класс C
    ])
    
    y_train = np.array(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
    
    # Тестовые данные
    X_test = np.array([
        [1.15, 1.2],  # должен быть A
        [3.4, 3.3],   # должен быть B
        [5.2, 5.1],   # должен быть C
    ])
    
    y_test = np.array(['A', 'B', 'C'])
    
    # Создаем и обучаем классификатор
    nb_classifier = NaiveBayesClassifier('b')
    nb_classifier.fit(X_train, y_train)
    
    # Делаем предсказания
    predictions = nb_classifier.predict(X_test)
    
    print("Предсказания для тестовых данных:")
    for i, (sample, pred, true) in enumerate(zip(X_test, predictions, y_test)):
        print(f"Пример {i+1}: {sample} -> Предсказано: {pred}, Истинное: {true}")
    
    # Оценка точности
    accuracy = nb_classifier.score(X_test, y_test)
    print(f"\nТочность на тестовых данных: {accuracy:.2f}")
    
    # Вывод статистик классов
    print("\nСтатистики классов:")
    for c in nb_classifier.classes:
        print(f"Класс {c}:")
        print(f"  Априорная вероятность: {nb_classifier.class_priors[c]:.2f}")
        print(f"  Средние значения признаков: {nb_classifier.class_stats[c]['p']}")
        # print(f"  Средние значения признаков: {nb_classifier.class_stats[c]['mean']}")
        # print(f"  Стандартные отклонения: {nb_classifier.class_stats[c]['std']}")

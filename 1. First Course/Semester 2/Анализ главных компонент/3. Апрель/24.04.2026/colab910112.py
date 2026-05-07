# -*- coding: utf-8 -*-
"""
Единый ноутбук Colab: Практики 9–12
CA | MCA | MDS | Kernel PCA | Isomap

Включено:
- Интерактивная схема связей (Plotly)
- Примеры кода для каждого метода
- Визуализация результатов
- Онлайн-тест для самопроверки
"""

# %% Установка необходимых библиотек (если не установлены)
# !pip install prince plotly scikit-learn matplotlib pandas numpy -q

# %% Импорты
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import MDS, Isomap
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_swiss_roll
import prince
import warnings
warnings.filterwarnings('ignore')

# Настройка графиков
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

# %% ===============================
# 1. ИНТЕРАКТИВНАЯ СХЕМА СВЯЗЕЙ ПРАКТИК
# ===================================
print("\n=== 1. Интерактивная схема связей практик 9–12 ===\n")

nodes = [
    {"id": "Данные", "x": 0.5, "y": 0.9, "color": "#2c5a8c", "desc": "Исходные данные: числовые, категориальные, расстояния"},
    {"id": "CA", "x": 0.15, "y": 0.6, "color": "#e67e22", "desc": "Correspondence Analysis (контингентные таблицы)"},
    {"id": "MCA", "x": 0.5, "y": 0.6, "color": "#e67e22", "desc": "Multiple Correspondence Analysis (dummy-кодирование)"},
    {"id": "MDS", "x": 0.85, "y": 0.6, "color": "#e67e22", "desc": "Multidimensional Scaling (матрицы расстояний)"},
    {"id": "KernelPCA", "x": 0.3, "y": 0.3, "color": "#16a085", "desc": "Kernel PCA (нелинейное через ядро)"},
    {"id": "Isomap", "x": 0.7, "y": 0.3, "color": "#16a085", "desc": "Isomap (геодезические расстояния)"}
]

edges = [("Данные", "CA"), ("Данные", "MCA"), ("Данные", "MDS"),
         ("CA", "MCA"), ("MDS", "KernelPCA"), ("MDS", "Isomap"),
         ("KernelPCA", "Isomap")]

node_dict = {n["id"]: n for n in nodes}
edge_x, edge_y = [], []
for src, tgt in edges:
    x0, y0 = node_dict[src]["x"], node_dict[src]["y"]
    x1, y1 = node_dict[tgt]["x"], node_dict[tgt]["y"]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

fig = go.Figure()
fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                         line=dict(color='#9aa9b9', width=2, dash='dash'),
                         hoverinfo='none'))
fig.add_trace(go.Scatter(
    x=[n["x"] for n in nodes], y=[n["y"] for n in nodes],
    mode='markers+text',
    marker=dict(size=40, color=[n["color"] for n in nodes], line=dict(color='white', width=2)),
    text=[n["id"] for n in nodes], textposition="middle center",
    textfont=dict(color='white', size=11),
    hovertext=[f"<b>{n['id']}</b><br>{n['desc']}" for n in nodes],
    hoverinfo='text'
))
fig.update_layout(title="Схема связей практик 9–12 (zoom, pan, tooltips)",
                  xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
                  yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[0, 1]),
                  plot_bgcolor='#f8fafc', width=900, height=700, hovermode='closest')
fig.show()

# %% ===============================
# 2. ПРАКТИКА 9: CORRESPONDENCE ANALYSIS (CA)
# ===========================================
print("\n=== 2. Correspondence Analysis (CA) ===\n")

# Данные: города × типы преступлений
data_ca = pd.DataFrame({
    'Угон': [80, 20, 50],
    'Кража': [30, 70, 40],
    'Мошенничество': [20, 10, 60]
}, index=['Москва', 'Питер', 'Казань'])

print("Контингентная таблица:")
print(data_ca)

# CA через библиотеку prince
ca = prince.CA(n_components=2, random_state=42)
ca = ca.fit(data_ca)

row_coords = ca.row_coordinates(data_ca)
col_coords = ca.column_coordinates(data_ca)
print("\nКоординаты строк (города):\n", row_coords)
print("Координаты столбцов (преступления):\n", col_coords)

# Получаем собственные значения и общую инерцию
eigenvals = ca.eigenvalues_
total_inertia = eigenvals.sum()
print(f"Собственные значения: {eigenvals}")
print(f"Общая инерция: {total_inertia:.4f}")

# Визуализация (matplotlib)
plt.figure()
plt.scatter(row_coords[0], row_coords[1], c='blue', s=150, label='Города')
plt.scatter(col_coords[0], col_coords[1], c='red', marker='^', s=150, label='Преступления')
for city in row_coords.index:
    plt.annotate(city, (row_coords.loc[city,0], row_coords.loc[city,1]), fontsize=10)
for crime in col_coords.index:
    plt.annotate(crime, (col_coords.loc[crime,0], col_coords.loc[crime,1]), fontsize=10)
plt.title('CA Biplot: города ↔ преступления')
plt.xlabel(f'Dim1 ({eigenvals[0]/total_inertia*100:.1f}%)')
plt.ylabel(f'Dim2 ({eigenvals[1]/total_inertia*100:.1f}%)')
plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
plt.grid(alpha=0.3)
plt.legend()
plt.show()

# %% ===============================
# 3. ПРАКТИКА 10: MULTIPLE CORRESPONDENCE ANALYSIS (MCA)
# =======================================================
print("\n=== 3. Multiple Correspondence Analysis (MCA) ===\n")

# Данные опроса (категориальные)
df_mca = pd.DataFrame({
    'Пол': ['М','М','Ж','Ж','М','Ж','М','Ж'],
    'Возраст': ['молодой','средний','средний','пожилой','молодой','средний','пожилой','молодой'],
    'Доход': ['низкий','средний','высокий','средний','низкий','высокий','средний','низкий'],
    'Предпочтение': ['A','B','A','C','B','A','C','B']
})
print("Исходные данные (категориальные):")
print(df_mca.head())

# MCA через prince
mca = prince.MCA(n_components=2, one_hot=True, random_state=42)
mca = mca.fit(df_mca)

# Координаты индивидов и категорий
ind_coords = mca.transform(df_mca)
cat_coords = mca.column_coordinates(df_mca)

# Собственные значения (инерция)
eigenvals_mca = mca.eigenvalues_
total_inertia_mca = eigenvals_mca.sum()
print("\nИнерция (собственные значения):", eigenvals_mca[:2])
print("Доля объяснённой дисперсии:", eigenvals_mca[:2] / total_inertia_mca)

# Визуализация
plt.figure(figsize=(12,8))
plt.scatter(ind_coords[0], ind_coords[1], alpha=0.6, label='Индивиды')
plt.scatter(cat_coords[0], cat_coords[1], marker='s', color='red', s=100, label='Категории')
for idx in cat_coords.index:
    plt.annotate(idx, (cat_coords.loc[idx,0], cat_coords.loc[idx,1]), fontsize=9, ha='center')
plt.title('MCA: карта индивидов и категорий')
plt.xlabel(f'Dim1 ({eigenvals_mca[0]/total_inertia_mca*100:.1f}%)')
plt.ylabel(f'Dim2 ({eigenvals_mca[1]/total_inertia_mca*100:.1f}%)')
plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# %% ===============================
# 4. ПРАКТИКА 11: MULTIDIMENSIONAL SCALING (MDS)
# ================================================
print("\n=== 4. Multidimensional Scaling (MDS) ===\n")

# Создадим матрицу расстояний между 5 городами (пример)
cities = ['Москва', 'Питер', 'Казань', 'Новосибирск', 'Владивосток']
dist_matrix = np.array([
    [0, 700, 800, 3300, 9200],
    [700, 0, 1100, 3600, 9500],
    [800, 1100, 0, 2500, 8200],
    [3300, 3600, 2500, 0, 5800],
    [9200, 9500, 8200, 5800, 0]
])

# Классическое MDS
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
coords = mds.fit_transform(dist_matrix)

plt.figure()
plt.scatter(coords[:,0], coords[:,1], s=100)
for i, city in enumerate(cities):
    plt.annotate(city, (coords[i,0], coords[i,1]), fontsize=10)
plt.title('MDS: реконструкция координат городов (по расстояниям)')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(alpha=0.3)
plt.show()

# Неметрическое MDS (на тех же данных, но итеративно)
mds_nonmetric = MDS(n_components=2, metric=False, dissimilarity='precomputed', random_state=42)
coords_non = mds_nonmetric.fit_transform(dist_matrix)
print("Неметрическое MDS – стресс:", mds_nonmetric.stress_)

# %% ===============================
# 5. ПРАКТИКА 12: KERNEL PCA + ISOMAP (НЕЛИНЕЙНОЕ СЖАТИЕ)
# ========================================================
print("\n=== 5. Kernel PCA и Isomap на 'швейцарском руле' ===\n")

# Генерация данных "швейцарский руль"
X, color = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

# Kernel PCA (RBF)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1, random_state=42)
X_kpca = kpca.fit_transform(X)

# Isomap
iso = Isomap(n_components=2, n_neighbors=10)
X_iso = iso.fit_transform(X)

# Визуализация
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(X[:,0], X[:,1], c=color, cmap=plt.cm.Spectral, s=5)
axes[0].set_title("Исходные данные (3D -> 2D проекция)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

axes[1].scatter(X_kpca[:,0], X_kpca[:,1], c=color, cmap=plt.cm.Spectral, s=5)
axes[1].set_title("Kernel PCA (RBF)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")

axes[2].scatter(X_iso[:,0], X_iso[:,1], c=color, cmap=plt.cm.Spectral, s=5)
axes[2].set_title("Isomap (k=10)")
axes[2].set_xlabel("Component 1")
axes[2].set_ylabel("Component 2")

plt.tight_layout()
plt.show()

# Дополнительно: сравнение с линейным PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Линейный PCA объясняет {pca.explained_variance_ratio_.sum():.2%} дисперсии")

# %% ===============================
# 6. ОНЛАЙН-ТЕСТ ПО ПРАКТИКАМ 9–12
# =================================
print("\n=== 6. Тест для самопроверки ===\n")

def run_test():
    score = 0
    print("Ответьте на вопросы (введите номер варианта ответа):\n")

    questions = [
        ("1. Какое расстояние используется в CA для сравнения профилей строк?",
         ["1) Евклидово", "2) Манхэттенское", "3) Хи-квадрат (χ²)", "4) Косинусное"], 3),
        ("2. MCA – это частный случай CA, применённый к:",
         ["1) Матрице корреляций", "2) Индикаторной (dummy) матрице", "3) Ковариационной матрице", "4) Косинусной матрице"], 2),
        ("3. Классическое метрическое MDS на евклидовых расстояниях эквивалентно:",
         ["1) Факторному анализу", "2) Линейному PCA", "3) Кластеризации K‑means", "4) t‑SNE"], 2),
        ("4. Какой метод нелинейного сжатия строит граф k‑ближайших соседей и использует геодезические расстояния?",
         ["1) Kernel PCA", "2) Isomap", "3) t‑SNE", "4) LLE"], 2),
        ("5. Визуализация одновременно строк и столбцов в CA называется:",
         ["1) Scree-plot", "2) Biplot", "3) Дендрограмма", "4) Тепловая карта"], 2)
    ]

    for q, opts, correct in questions:
        print(q)
        for opt in opts:
            print(opt)
        ans = input("Ваш ответ (номер): ").strip()
        if ans == str(correct):
            score += 1
            print("✓ Верно\n")
        else:
            print(f"✗ Неверно. Правильный ответ: {correct}\n")

    print(f"Результат: {score} из {len(questions)}")
    if score == 5:
        print("Отлично! Вы усвоили материал практик 9–12.")
    elif score >= 3:
        print("Хорошо, но стоит повторить теорию.")
    else:
        print("Рекомендуем пересмотреть разделы с формулами и примерами.")

# Запуск теста (интерактивный – при выполнении в Colab)
run_test()

print("\n=== Все практики выполнены успешно ===")
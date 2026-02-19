import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


x = np.arange(-5, 5)
y = 2 * x + np.random.randn(10) * 3
X = np.column_stack((x, y))

m, n = X.shape

middle_X = []
for col in range(n):
    middle_X += [X[:, col].sum() / m]

middle_X = np.array(middle_X)
X_ = X - middle_X

cov = (X_.T @ X_) / m
N, V = np.linalg.eigh(cov)

# plt.scatter(cov[:, 0], cov[:, 1], s=50, ec="k", zorder=3, label="cov")

Z_manual = X_ @ V[:, N.argsort()[::-1]]
print(f"Manual:\n{Z_manual}\n")

pca = PCA(n_components=n)
Z_sklearn = pca.fit_transform(X)
print(f"Sklearn:\n{Z_sklearn}")


def scree_plot():
    plt.bar(range(1, len(N) + 1), N, alpha=0.7, color="steelblue", ec="k")
    plt.plot(range(1, len(N) + 1), N, "ro-", lw=2, ms=8)
    
    plt.xlabel("Component Numbers", fontsize=12)
    plt.ylabel("Eigenvalue", fontsize=12)
    plt.title("Scree-plot (Manual PCA)", fontsize=14)
    plt.xticks(range(1, n + 1))
    
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.show()


def biplot():
    plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    
    plt.scatter(Z_manual[:, 0], Z_manual[:, 1], ec="k", zorder=3)
    
    plt.title("Biplot (Manual PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")

    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.show()


def origin_data():
    plt.scatter(x, y, s=50, ec="k", zorder=3, label="origin")
    plt.scatter(X_[:, 0], X_[:, 1], s=50, ec="k", zorder=3, label="centered")

    plt.xlabel("X")
    plt.ylabel("Y")

    plt.plot(Z_manual[:, 0], Z_manual[:, 1], label="Z (manual)")
    plt.plot(Z_sklearn[:, 0], Z_sklearn[:, 1], label="Z (sklearn)")

    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.show()


origin_data()
scree_plot()
biplot()

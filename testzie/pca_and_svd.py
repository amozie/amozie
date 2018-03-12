from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import decomp_svd

x = np.linspace(-2, 2)
y = 2 - x

x += np.random.normal(0, 0.25, *x.shape)
y += np.random.normal(0, 0.25, *y.shape)

X = np.column_stack((x, y))

plt.scatter(x, y)

pca = PCA(1)
pca.fit(X)
X_pca = pca.transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1])


U, S, V = np.linalg.svd(X, False)

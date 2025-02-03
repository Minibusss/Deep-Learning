import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

np.random.seed(1)
x, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0, n_informative=1,
    n_clusters_per_class=1
)

# Visualisation des données
plt.figure(num=None, figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, edgecolors='k')
plt.xlabel('x0')
plt.ylabel('x1')
x.shape


# Génération d'un modele en utilisant la fonction cout 'log' pour Logistic Regression
model = SGDClassifier(max_iter=1000, eta0=0.001, loss='log_loss')
model.fit(x, y)
print('score:', model.score(x, y))

# Visualisation des données
h = .02
colors = "bry"
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
plt.axis('tight')

for i, color in zip(model.classes_, colors):
    idx = np.where(y == i)
    plt.scatter(x[idx, 0], x[idx, 1], c=color, cmap=plt.cm.Paired, edgecolor='black', s=20)

plt.show()
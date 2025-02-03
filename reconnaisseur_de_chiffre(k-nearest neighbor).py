import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier 

# importons une base de données de chiffre
digits = load_digits()

X = digits.data
y = digits.target


# Entraînement du modele
model = KNeighborsClassifier()
model.fit(X, y)
model.score(X, y) 


#Test du modele
test = digits['images'][1000].reshape(1, -1)
plt.imshow(digits['images'][1000], cmap = 'Greys_r')
print(model.predict(test))
plt.show()
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255.0  # normalize pixel values

# Split data
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train MLP
clf = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                    batch_size=256, max_iter=10, verbose=True, random_state=42)
clf.fit(Xtr, ytr)

# Evaluate
ypred = clf.predict(Xte)
print("Accuracy:", accuracy_score(yte, ypred))




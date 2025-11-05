import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LWLR:
    def __init__(self, tau=0.2): self.tau = tau
    def predict(self, X, Y, q):
        X_ = np.hstack([X, np.ones((len(X),1))])
        q_ = np.array([q,1])
        W = np.diag(np.exp(-np.sum((X_-q_)**2, axis=1)/(2*self.tau**2)))
        theta = np.linalg.pinv(X_.T @ W @ X_) @ (X_.T @ W @ Y)
        return q_ @ theta
    def fit_and_show(self, X, Y):
        X_test = np.linspace(-np.max(X), np.max(X), len(X))
        Y_pred = np.array([self.predict(X,Y,x)[0] for x in X_test])
        plt.scatter(X,Y,color='red')
        plt.scatter(X_test,Y_pred,color='green')
        plt.title(f"LWLR, tau={self.tau}"); plt.show()

# Load and normalize data
X = pd.read_csv("weightedX.csv").values
Y = pd.read_csv("weightedY.csv").values
X = (X - X.mean())/X.std()

LWLR(tau=0.2).fit_and_show(X,Y)
LWLR(tau=2).fit_and_show(X,Y)

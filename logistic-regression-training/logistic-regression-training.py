import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.asarray(X, dtype=float)

    y = np.asarray(y,dtype=float).reshape(-1)

    n, m = X.shape

    w = np.zeros(m)

    b = 0.0

    for i in range(steps):

# Predicción lineal

        z = X@w +b

# Probabilidades

        p = _sigmoid(z)

# Gradientes

        grad_w = (1 / n) * (X.T @ (p-y))

        grad_b = (1 / n) * np.sum(p-y)

# Actualización de parámetros
 
        b = b - lr * grad_b
        w = w - lr * grad_w
    return w, b
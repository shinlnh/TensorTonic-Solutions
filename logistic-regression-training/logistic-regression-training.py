import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w = np.zeros(X.shape[1])
    b = 0.0
    for i in range(steps):
        
        n = len(X)
        p = _sigmoid(X @ w + b)
        loss = 0
        for i in range(n):
            loss += -(1/n) * np.sum(y*np.log(p) + (1-y)*np.log(1-p))
        loss = -(1/n) * loss 
        dw = (1/n) * (X.T @ (p - y))
        db = (1/n) * np.sum(p - y)  

        w -= lr * dw
        b -= lr * db

    return w, b
    pass
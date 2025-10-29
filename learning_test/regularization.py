import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0.5, 1.5],
    [1.0, 2.0],
    [1.5, 0.5],
    [2.0, 1.0]
])

y = np.array([0, 0, 1, 1])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#X: (m, n) boyutlu özellik matrisi (m örnek, n özellik)

#y: (m,) boyutlu gerçek etiketler (0 veya 1)

#w: (n,) boyutlu ağırlık vektörü

#b: bias skaler

#lam: regularization katsayısı λ

#reg_type: "L2" veya "L1", hangi regularization kullanılacağını seçer

def compute_gradient(X, y, w, b, lam, reg_type):
    m = len(y)
    z = X.dot(w) + b
    y_predict = sigmoid(z)
    
    dw = 1/m * (X.T.dot(y_predict - y))
    db = 1/m * np.sum(y_predict - y)
    
    if reg_type == "L2":
        dw += (lam/m) * w
    elif reg_type == "L1":
        dw += (lam/m) * np.sign(w)
    
    return dw, db

def gradient_descent(X, y, w, b, lam, lr=0.1, iterations =50, reg_type="L2"):
    w_history = [w.copy()]
    b_history = [b]
    
    for i in range(iterations):
        dw, db = compute_gradient(X, y, w, b, lam, reg_type)
        w -= lr * dw
        b -= lr * db
        
        w_history.append(w.copy())
        b_history.append(b)
    
    return w, b, np.array(w_history), np.array(b_history)

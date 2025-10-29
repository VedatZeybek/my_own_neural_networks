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

#Diyelim ki sen bir bahçıvansın ve bir bahçe var:

#w = her bitkiye verdiğin su miktarı (ağırlıklar)

#b = genel gübre miktarı (bias)

#X = toprak özellikleri ve hava durumu (girdi verileri)

#y = bitkilerin olması gereken ideal boyları (hedef)

#lam = dikkat etmen gereken limit (regularization)

#lr = suyu/gübreyi ne kadar hızlı değiştireceğin (learning rate)

#iterations = kaç gün boyunca değişiklik yapacağın

def gradient_descent(X, y, w, b, lam, lr=0.1, iterations =50, reg_type="L2"):
    w_history = [w.copy()]
    b_history = [b]
    #Her gün ne yaptığını kaydediyoruz, böylece ilerlemeyi görebiliriz

    for i in range(iterations):
        dw, db = compute_gradient(X, y, w, b, lam, reg_type)
        w -= lr * dw
        b -= lr * db
        w_history.append(w.copy())
        b_history.append(b)
    
    return w, b, np.array(w_history), np.array(b_history)

w_init = np.array([0.5, 0.5])
b_init = 0.0
lam = 1.0
lr = 0.1
iterations = 30

w_l2, b_l2, w_hist_l2, b_hist_l2 = gradient_descent(
    X, y, w_init.copy(), b_init, lam, lr, iterations, reg_type="L2"
)

w_l1, b_l1, w_hist_l1, b_hist_l1 = gradient_descent(
    X, y, w_init.copy(), b_init, lam, lr, iterations, reg_type="L1"
)


#visualization

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(w_hist_l2[:,0], label="w1")
plt.plot(w_hist_l2[:,1], label="w2")
plt.title("L2 Regularization")
plt.xlabel("Iteration")
plt.ylabel("Weight value")
plt.legend()

plt.subplot(1,2,2)
plt.plot(w_hist_l1[:,0], label="w1")
plt.plot(w_hist_l1[:,1], label="w2")
plt.title("L1 Regularization")
plt.xlabel("Iteration")
plt.ylabel("Weight value")
plt.legend()

plt.tight_layout()
plt.savefig("regularization_weights.png")  # Kaydedilen dosya
import numpy as np

# --- Dense Layer ---
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # He initialization (ReLU için daha uygun)
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Basit bir backward (gradient hesaplama)
    def backward(self, dvalues): #loss fucnction sana dvalues gönderir.
        # Gradientleri hesapla
        self.dweights = np.dot(self.inputs.T, dvalues) #bu dvaluesi kullanarak dweights ve dbiases bulunur. ağırlıkları güncellemek için.
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True) #biasleri güncellemek için.
        # Gradientleri inputlara geçir
        self.dinputs = np.dot(dvalues, self.weights.T) #ve bu değer dinputs'a geçirlilir. ve önceki katmana gradyan aktarmak için.
        

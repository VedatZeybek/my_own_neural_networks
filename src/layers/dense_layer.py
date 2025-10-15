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
    def backward(self, dvalues):
        # Gradientleri hesapla
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradientleri inputlara geçir
        self.dinputs = np.dot(dvalues, self.weights.T)

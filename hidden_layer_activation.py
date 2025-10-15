import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(100, 3)

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons) #do not need to transpose
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLu:
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

class Activation_Softmax:
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		self.output = probabilities

layer1  = Layer_Dense(2, 5)
activation1 = Activation_ReLu()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

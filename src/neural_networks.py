import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from layers.dense_layer import Layer_Dense
from activations.activation_ReLu import Activation_ReLU
from activations.combined_softmax_ce import Activation_Softmax_Loss_CategoricalCrossentropy


nnfs.init()

# --- Veri ---
X, y = spiral_data(100, 3)

# --- Model ---
dense1 = Layer_Dense(2, 64) #backpropagation yapıyor.
activation1 = Activation_ReLU() # ReLu sana (non-linearity) verir. Yani karmaşık şekiller / eğrileri manipüle edebilirsin.
dense2 = Layer_Dense(64, 3) #ikinci nöral katman.
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy() # 

# --- Forward pass ---
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y)

# --- Sonuçlar ---
print("Loss:", loss)

# --- Accuracy ---
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print("Accuracy:", accuracy)

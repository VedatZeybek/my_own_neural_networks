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
dense1 = Layer_Dense(2, 128)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(128, 128)
activation2 = Activation_ReLU()
dense3 = Layer_Dense(128, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# --- Dropout parametreleri ---
dropout_rate = 0.1  # %10 dropout
dropout_mask1 = None
dropout_mask2 = None

# --- Hyperparametreler ---
learning_rate = 0.01
epochs = 1000
batch_size = 32

# Adam optimizer parametreleri
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-7

# Adam momentler
dense1_mw, dense1_mb, dense1_vw, dense1_vb = np.zeros_like(dense1.weights), np.zeros_like(dense1.biases), np.zeros_like(dense1.weights), np.zeros_like(dense1.biases)
dense2_mw, dense2_mb, dense2_vw, dense2_vb = np.zeros_like(dense2.weights), np.zeros_like(dense2.biases), np.zeros_like(dense2.weights), np.zeros_like(dense2.biases)
dense3_mw, dense3_mb, dense3_vw, dense3_vb = np.zeros_like(dense3.weights), np.zeros_like(dense3.biases), np.zeros_like(dense3.weights), np.zeros_like(dense3.biases)

# --- Training loop ---
for epoch in range(1, epochs + 1):
    # Mini-batch
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        xb = X_shuffled[start:end]
        yb = y_shuffled[start:end]

        # --- Forward pass ---
        dense1.forward(xb)
        activation1.forward(dense1.output)
        # Dropout
        dropout_mask1 = (np.random.rand(*activation1.output.shape) > dropout_rate) / (1 - dropout_rate)
        activation1.output *= dropout_mask1

        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dropout_mask2 = (np.random.rand(*activation2.output.shape) > dropout_rate) / (1 - dropout_rate)
        activation2.output *= dropout_mask2

        dense3.forward(activation2.output)
        loss = loss_activation.forward(dense3.output, yb)

        # --- Backward pass ---
        loss_activation.backward(loss_activation.output, yb)
        dense3.backward(loss_activation.dinputs)
        activation2.backward(dense3.dinputs)
        activation2.dinputs *= dropout_mask2  # Dropout backward
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        activation1.dinputs *= dropout_mask1  # Dropout backward
        dense1.backward(activation1.dinputs)

        # --- Adam güncellemesi ---
        for layer, mw, mb, vw, vb in [
            (dense1, dense1_mw, dense1_mb, dense1_vw, dense1_vb),
            (dense2, dense2_mw, dense2_mb, dense2_vw, dense2_vb),
            (dense3, dense3_mw, dense3_mb, dense3_vw, dense3_vb)
        ]:
            mw[:] = beta1 * mw + (1 - beta1) * layer.dweights
            mb[:] = beta1 * mb + (1 - beta1) * layer.dbiases
            vw[:] = beta2 * vw + (1 - beta2) * (layer.dweights ** 2)
            vb[:] = beta2 * vb + (1 - beta2) * (layer.dbiases ** 2)

            mw_corr = mw / (1 - beta1 ** epoch)
            mb_corr = mb / (1 - beta1 ** epoch)
            vw_corr = vw / (1 - beta2 ** epoch)
            vb_corr = vb / (1 - beta2 ** epoch)

            layer.weights -= learning_rate * mw_corr / (np.sqrt(vw_corr) + epsilon)
            layer.biases  -= learning_rate * mb_corr / (np.sqrt(vb_corr) + epsilon)

    # Epoch sonu Accuracy
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    loss = loss_activation.forward(dense3.output, y)
    predictions = np.argmax(loss_activation.output, axis=1)
    y_true = np.argmax(y, axis=1) if len(y.shape) == 2 else y
    accuracy = np.mean(predictions == y_true)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss={loss:.3f}, Accuracy={accuracy:.3f}")

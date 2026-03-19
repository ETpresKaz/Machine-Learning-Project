import matplotlib.pyplot as plt
from model import Model
from linear_layer import LinearLayer
from activation_layer import ActivationLayer
from activations import softmax, softmax_deriv, relu, relu_deriv
from losses import cross_entropy, cross_entropy_deriv
import numpy as np
import tensorflow as tf

# Reproducibility-ish
np.random.seed(42)
tf.random.set_seed(42)

mnist = tf.keras.datasets.mnist
to_categorical = tf.keras.utils.to_categorical

# Load and preprocess MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28*28).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28*28).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Split validation set from training data
X_val = X_train[-5000:]
y_val = y_train[-5000:]
X = X_train[:-5000]
y = y_train[:-5000]

# Build model
model = Model()
model.add(LinearLayer(784, 128))
model.add(ActivationLayer(relu, relu_deriv))
model.add(LinearLayer(128, 64))
model.add(ActivationLayer(relu, relu_deriv))
model.add(LinearLayer(64, 10))
model.add(ActivationLayer(softmax, softmax_deriv))  # softmax grad is handled in loss

model.summary()

BATCH_SIZE = 64
LR = 0.05
EPOCHS = 100
patience = 10

losses = []
best_loss = float('inf')
patience_counter = 0

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

for epoch in range(EPOCHS):
    perm = np.random.permutation(len(X))
    X_shuffled = X[perm]
    y_shuffled = y[perm]
    epoch_loss = 0.0

    for i in range(0, len(X), BATCH_SIZE):
        X_batch = X_shuffled[i:i + BATCH_SIZE]
        y_batch = y_shuffled[i:i + BATCH_SIZE]

        y_pred = model.forward(X_batch)
        loss = cross_entropy(y_batch, y_pred)
        epoch_loss += loss * len(X_batch)

        grad = cross_entropy_deriv(y_batch, y_pred)  # equals (y_pred - y_true)/batch
        model.backward(grad)
        model.update(LR)

    epoch_loss /= len(X)
    losses.append(epoch_loss)

    # Validation
    val_pred = model.forward(X_val)
    val_loss = cross_entropy(y_val, val_pred)
    val_acc = accuracy(y_val, val_pred)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        train_pred_sample = model.forward(X[:5000])
        train_acc = accuracy(y[:5000], train_pred_sample)
        print(f"Epoch {epoch+1:3d}: loss {epoch_loss:.4f} | val_loss {val_loss:.4f} | train_acc {train_acc*100:.2f}% | val_acc {val_acc*100:.2f}%")

    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        model.save("mnist_model.pkl")  # Save best model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best val loss: {best_loss:.4f}")
            break

# Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss Over Time")
plt.grid(True)
plt.show()

# Reload best model and evaluate
print("\nReloading best model from file...")
model.load("mnist_model.pkl")

print("Evaluating on train/val/test...")
train_pred = model.forward(X)
val_pred = model.forward(X_val)
test_pred = model.forward(X_test)

print(f"Train Accuracy: {accuracy(y, train_pred) * 100:.2f}%")
print(f"Val Accuracy:   {accuracy(y_val, val_pred) * 100:.2f}%")
print(f"Test Accuracy:  {accuracy(y_test, test_pred) * 100:.2f}%")
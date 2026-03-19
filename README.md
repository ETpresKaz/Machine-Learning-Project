# MNIST From-Scratch Neural Network (NumPy)

A from-scratch feedforward neural network trained on MNIST using only NumPy for the model and TensorFlow **only** for data loading. Includes a Tkinter GUI to draw digits and get live predictions.

## Features
- 3-layer FNN: 784 → 128 (ReLU) → 64 (ReLU) → 10 (softmax)
- Mini-batch SGD with early stopping
- Numerically stable softmax + cross-entropy
- Model checkpointing via pickle
- Tkinter prediction GUI with confidence bars
- Lightweight, readable code (teachable)

## Results (reference)
- Val accuracy: ~98.04%
- Test accuracy: ~97.51%
- Training time: ~01 min on CPU (M-series / Intel)

Reproduce these numbers with the commands below; may vary based on CPU.

## Project Structure
Machine Learning Project/

* main.py                # train/validate/test + checkpoint + plot
* predict_gui.py         # GUI to draw digits and predict
* model.py               # Layer stack, forward/backward, save/load
* linear_layer.py        # Fully-connected layer
* activation_layer.py    # Activation wrapper
* activations.py         # ReLU, sigmoid, softmax (+ derivs)
* losses.py              # MSE / Cross-entropy (+ grads)
* config.py              # Hyperparameters (optional use)
* mnist_model.pkl        # Saved best model (created after training)

## Requirements
- Python 3.10+
- NumPy
- TensorFlow (for MNIST dataset only)
- Matplotlib (for the training curve)
- Pillow + Tkinter (for the GUI)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy tensorflow matplotlib pillow
```

## Quickstart

1. Train
```bash
python main.py
```
This will:

* Download MNIST,
* Train with mini-batches,
* Early-stop on validation loss,
* Save the best weights to mnist_model.pkl
* plot the training loss curve

2. Evaluate

main.py automatically reloads the best checkpoint and prints Train/Val/Test accuracy.

3. Predict with the GUI
```bash
python predict_gui.py
```

Draw a digit (thicker strokes help). Click Predict to see the top-1 class and per-digit confidences.

## Configuration
Adjust architecture and hyperparameters in config.py (and optionally wire them into main.py):
```python
INPUT_SIZE = 784
HIDDEN_LAYERS = [128, 64]
OUTPUT_SIZE = 10
LEARNING_RATE = 0.05
```
## Implementation Notes
* We use softmax + cross-entropy, so the output-layer gradient simplifies to dZ = (P - Y) / batch_size.
* softmax_deriv is effectively passthrough here; the loss provides the correct gradient.
* Weights are N(0, 0.1); LR 0.05 is intentionally bold but stable for this init.

## Reproductability

We set NumPy and TF seeds, but exact parity across machines is not guaranteed due to differing BLAS kernels and TF internals.

## Troubleshooting
* GUI predicts gibberish: draw thicker; the preprocessor centers a 20×20 crop into 28×28.
* Training diverges: try LEARNING_RATE=0.01 or use smaller initial weights.
* Val plateaus early: increase patience or use larger batch size (e.g., 128).

## License

MIT

## Acknowledgments
* MNIST dataset (LeCun et al.)

* Goodfellow, Bengio, Courville - ***Deep Learning*** (for the backprop foundations)
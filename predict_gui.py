import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np

from model import Model
from linear_layer import LinearLayer
from activation_layer import ActivationLayer
from activations import relu, relu_deriv, softmax, softmax_deriv

# === Build the SAME architecture used in training and load weights ===
model = Model()
model.add(LinearLayer(784, 128))
model.add(ActivationLayer(relu, relu_deriv))
model.add(LinearLayer(128, 64))
model.add(ActivationLayer(relu, relu_deriv))
model.add(LinearLayer(64, 10))
model.add(ActivationLayer(softmax, softmax_deriv))
model.load("mnist_model.pkl")

# ---------- UI ----------
WIDTH, HEIGHT = 280, 280  # drawing area (10x MNIST scale)
PREVIEW_SCALE = 6         # upscale 28x28 -> 168x168 for preview

root = tk.Tk()
root.title("MNIST Digit Predictor")

# Layout: left = drawing canvas; right = preview + confidences
left_frame = tk.Frame(root)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

right_frame = tk.Frame(root)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Drawing canvas (white bg, black pen)
canvas = tk.Canvas(left_frame, width=WIDTH, height=HEIGHT, bg="white", highlightthickness=1, highlightbackground="#ccc")
canvas.pack()

# Backing PIL image to mirror the drawing
image = Image.new("L", (WIDTH, HEIGHT), color=255)  # L=grayscale, white background
draw = ImageDraw.Draw(image)

# Controls
controls = tk.Frame(left_frame)
controls.pack(fill=tk.X, pady=(8,0))

def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=255)
    result_label.config(text="Draw a digit and click Predict")
    # Clear preview + confidences
    set_preview(None)
    update_confidences([0.0]*10)

btn_clear = tk.Button(controls, text="Clear", command=clear_canvas)
btn_clear.pack(side=tk.LEFT)

btn_predict = tk.Button(controls, text="Predict", command=lambda: predict_digit())
btn_predict.pack(side=tk.RIGHT)

# Right side: MNIST-style preview (what the model sees)
preview_title = tk.Label(right_frame, text="MNIST-style 28×28 (preview)", font=("Helvetica", 12, "bold"))
preview_title.pack(anchor="w")

preview_label = tk.Label(right_frame, bd=1, relief=tk.SOLID, width=PREVIEW_SCALE*28, height=PREVIEW_SCALE*28)
preview_label.pack(pady=(2, 10))

# Confidence list for 0–9
conf_title = tk.Label(right_frame, text="Confidence by class", font=("Helvetica", 12, "bold"))
conf_title.pack(anchor="w")

conf_frame = tk.Frame(right_frame)
conf_frame.pack(anchor="w", pady=(2, 10))

conf_rows = []
for d in range(10):
    row = tk.Frame(conf_frame)
    row.pack(anchor="w")
    lbl_digit = tk.Label(row, width=2, text=str(d), font=("Menlo", 11))
    lbl_digit.pack(side=tk.LEFT)
    lbl_bar = tk.Canvas(row, width=140, height=12, bg="#f5f5f5", highlightthickness=0)
    lbl_bar.pack(side=tk.LEFT, padx=(6, 6))
    lbl_pct = tk.Label(row, width=6, text="0.0%", font=("Menlo", 11))
    lbl_pct.pack(side=tk.LEFT)
    conf_rows.append((lbl_bar, lbl_pct))

# Top-1 prediction
result_label = tk.Label(right_frame, text="Draw a digit and click Predict", font=("Helvetica", 14))
result_label.pack(anchor="w")

# ---------- Drawing handlers ----------
def paint(event):
    r = 8  # brush radius (thicker stroke helps downscale)
    x1, y1 = (event.x - r), (event.y - r)
    x2, y2 = (event.x + r), (event.y + r)
    canvas.create_oval(x1, y1, x2, y2, fill="black", outline="black")
    draw.ellipse([x1, y1, x2, y2], fill=0)  # draw on backing image (0=black)

canvas.bind("<B1-Motion>", paint)

# ---------- Helpers ----------
_preview_imgtk = None  # keep reference to avoid GC

def set_preview(img28: Image.Image | None):
    """Show a 28x28 grayscale image upscaled with nearest-neighbor. If None, clear."""
    global _preview_imgtk
    if img28 is None:
        preview_label.config(image="", text="")
        _preview_imgtk = None
        return
    up = img28.resize((28*PREVIEW_SCALE, 28*PREVIEW_SCALE), resample=Image.NEAREST)
    _preview_imgtk = ImageTk.PhotoImage(up.convert("L"))
    preview_label.config(image=_preview_imgtk)

def update_confidences(probs):
    """probs: length-10 iterable (values in [0,1]). Update bars and percentages."""
    max_w = 140
    for d in range(10):
        p = float(probs[d])
        bar, pct = conf_rows[d]
        bar.delete("all")
        w = int(max_w * p)
        if w > 0:
            bar.create_rectangle(0, 0, w, 12, fill="#4a90e2", width=0)
        pct.config(text=f"{p*100:4.1f}%")

def preprocess_to_mnist(img_drawn: Image.Image) -> np.ndarray:
    """
    Convert the high-res white canvas with black ink to a MNIST-like 28x28 array in [0,1],
    where the digit is light on dark background (MNIST style).
    Steps:
      1) Resize to 28x28 and invert -> white digit on black bg.
      2) If any ink present, crop to bounding box, resize to 20x20, and paste centered into 28x28.
    """
    # Step 1: downscale and invert
    img_resized = img_drawn.resize((28, 28)).convert("L")
    img_inverted = ImageOps.invert(img_resized)  # white strokes on black bg
    arr = np.asarray(img_inverted).astype("float32") / 255.0

    # If blank-ish, return as-is
    if arr.max() < 0.10:
        return arr

    # Simple bbox crop then center at 20x20 within 28x28 (classic heuristic)
    coords = np.argwhere(arr > 0.10)
    (x0, y0) = coords.min(axis=0)
    (x1, y1) = coords.max(axis=0) + 1
    cropped = arr[x0:x1, y0:y1]

    # Resize cropped content to 20x20 with antialiasing
    cropped_img = Image.fromarray((cropped * 255).astype(np.uint8)).resize((20, 20), Image.LANCZOS)

    # Paste into 28x28 black canvas centered
    canvas28 = Image.new("L", (28, 28), color=0)
    canvas28.paste(cropped_img, ((28 - 20) // 2, (28 - 20) // 2))

    # Convert back to [0,1] float32
    return np.asarray(canvas28).astype("float32") / 255.0

# ---------- Prediction ----------
def predict_digit():
    # 1) Preprocess to MNIST-like 28x28, show preview
    arr28 = preprocess_to_mnist(image)
    img28 = Image.fromarray((arr28 * 255).astype(np.uint8), mode="L")
    set_preview(img28)

    if arr28.max() < 0.10:
        result_label.config(text="No digit detected — try a thicker stroke.")
        update_confidences([0.0]*10)
        return

    # 2) Run model
    x = arr28.reshape(1, 784)
    probs = model.forward(x)[0]  # softmax output

    # 3) Update confidences and top-1
    update_confidences(probs)
    top = int(np.argmax(probs))
    result_label.config(text=f"Prediction: {top}  ({probs[top]*100:.1f}%)")

# Start clean
clear_canvas()

root.mainloop()
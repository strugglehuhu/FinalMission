# app.py
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
from model import SimpleLSTM

# --- Load scaler + model config ---
scaler = np.load("scaler.npz")
mean = scaler["mean"].item()
std = scaler["std"].item()
SEQ_LEN = int(scaler["seq_len"])
HIDDEN = int(scaler["hidden"])

# --- Load trained model ---
device = "cpu"
model = SimpleLSTM(input_size=1, hidden_size=HIDDEN)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# --- Flask app ---
app = Flask(__name__)

def preprocess(seq_str):
    """
    Convert '1,2,3,4,5' -> normalized torch tensor [1, seq_len, 1]
    """
    try:
        nums = [float(x) for x in seq_str.split(",")]
    except ValueError:
        raise ValueError("All inputs must be numbers separated by commas.")

    if len(nums) != SEQ_LEN:
        raise ValueError(f"Expected {SEQ_LEN} numbers, got {len(nums)}.")

    arr = np.array(nums, dtype=np.float32)
    arr = (arr - mean) / std  # normalize
    arr = arr[None, :, None]  # [1, seq_len, 1]
    return torch.from_numpy(arr).float()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "sequence" not in data:
        return jsonify({"error": "Missing 'sequence' field"}), 400

    try:
        x = preprocess(data["sequence"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    with torch.no_grad():
        yhat_norm = model(x).item()
    yhat = yhat_norm * std + mean

    return jsonify({"prediction": int(round(yhat))})

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

# Challenge 80 — Time-Series Forecaster

This project is my final AI challenge.  
It’s a simple web app that takes **5 numbers as input** and predicts the **6th number** using a small **LSTM** trained in PyTorch.

---

## ⚙️ How it works
- **Model:** LSTM forecaster (`train.py`) → saves `model.pth` + `scaler.npz`
- **API:** Flask app (`app.py`) → `/predict` endpoint
- **Front-end:** `templates/index.html` + `static/style.css`

---

## 🚀 Run locally
1. Clone repo:
   ```bash
   git clone https://github.com/your-username/challenge80_rnn.git
   cd challenge80_rnn

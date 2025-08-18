# train.py (stronger version)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from model import SimpleLSTM

# --- Reproducibility ---
np.random.seed(42)
torch.manual_seed(42)

# --- Config ---
SEQ_LEN   = 5          # input window length
N_SAMPLES = 20000       # more data -> better generalization
BATCH     = 256
HIDDEN    = 64
EPOCHS    = 40          # longer training
LR        = 1e-3
PATIENCE  = 6           # early stopping patience (epochs without val improvement)
DEVICE    = "cpu"

def make_dataset(n_samples=20000, seq_len=10):
    """
    Generate arithmetic-progressions: x_{t+1} = x_t + d
    Each sample: unique (start, step). No noise to keep the task clean.
    """
    X, y = [], []
    all_values = []

    # Randomize ranges a bit wider to avoid overfitting to tiny magnitudes
    for _ in range(n_samples):
        start = np.random.uniform(-200, 200)
        step  = np.random.uniform(-10, 10)
        seq   = start + step * np.arange(seq_len + 1, dtype=np.float32)  # length = seq_len+1
        X.append(seq[:seq_len])
        y.append(seq[-1])
        all_values.extend(seq.tolist())

    X = np.asarray(X, dtype=np.float32)   # [N, seq_len]
    y = np.asarray(y, dtype=np.float32)   # [N]

    # Standardize using global mean/std across all values
    mean = float(np.mean(all_values))
    std  = float(np.std(all_values) + 1e-8)

    Xn = (X - mean) / std
    yn = (y - mean) / std

    # Add channel dim -> [N, seq_len, 1], [N, 1]
    Xn = Xn[..., None]
    yn = yn[..., None]

    return Xn, yn, mean, std

def to_torch(x, y):
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()

def evaluate(model, loader, loss_fn):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb)
            total += loss_fn(pred, yb).item() * len(xb)
            n += len(xb)
    model.train()
    return total / max(n, 1)

def main():
    print("Preparing data…")
    X, Y, mean, std = make_dataset(N_SAMPLES, SEQ_LEN)

    # Train/val split
    idx = np.random.permutation(len(X))
    split = int(0.9 * len(X))
    tr_idx, va_idx = idx[:split], idx[split:]
    X_train, Y_train = X[tr_idx], Y[tr_idx]
    X_val,   Y_val   = X[va_idx], Y[va_idx]

    xtr, ytr = to_torch(X_train, Y_train)
    xva, yva = to_torch(X_val,   Y_val)

    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(TensorDataset(xva, yva), batch_size=BATCH)

    model = SimpleLSTM(input_size=1, hidden_size=HIDDEN, num_layers=1).to(DEVICE)
    loss_fn = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None
    patience_left = PATIENCE

    print("Training…")
    for epoch in range(1, EPOCHS + 1):
        # Train one epoch
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
            n_seen += len(xb)
        train_loss = running / n_seen

        # Validate
        val_loss = evaluate(model, val_loader, loss_fn)

        # Early stopping check
        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = PATIENCE
        else:
            patience_left -= 1

        print(f"Epoch {epoch:02d}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
              f"{'★' if improved else ''}")

        if patience_left == 0:
            print("Early stopping.")
            break

    # Load best weights before saving
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save weights and scaler/meta
    torch.save(model.state_dict(), "model.pth")
    np.savez("scaler.npz", mean=mean, std=std, seq_len=SEQ_LEN, hidden=HIDDEN)
    print(f"Saved: model.pth, scaler.npz")

    # Quick sanity checks
    def predict_next(seq):
        """seq: np.array of shape [SEQ_LEN] in raw scale"""
        x = ((seq - mean) / std)[None, :, None]
        x = torch.from_numpy(x).float()
        model.eval()
        with torch.no_grad():
            yhat_norm = model(x).item()
        return yhat_norm * std + mean

    print("\nSanity checks:")
    for start, step in [(0, 2), (5, -3), (-10, 7.5)]:
        seq = start + step * np.arange(SEQ_LEN, dtype=np.float32)
        true_next = start + step * SEQ_LEN
        yhat = predict_next(seq)
        print(f"  seq(last 3)={seq[-3:]}  true_next={true_next:.3f}  pred={yhat:.3f}")

if __name__ == "__main__":
    main()

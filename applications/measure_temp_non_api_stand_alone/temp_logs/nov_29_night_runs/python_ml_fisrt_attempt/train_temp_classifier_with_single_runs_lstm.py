import os
import glob
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import classification_report, confusion_matrix


# ==========================
# 1. Data loading utilities
# ==========================

def load_temp_file(path):
    """
    Load temp trace file of the form:
      <temp> C <timestamp> ns
    Returns t_s (seconds) and temp_C (numpy arrays).
    """
    data = np.loadtxt(path, usecols=(0, 2))

    if data.ndim == 1:
        data = data.reshape(1, -1)

    temp = data[:, 0]
    t_ns = data[:, 1]
    t_s = (t_ns - t_ns[0]) * 1e-9

    return t_s, temp


def build_sequence_fpd_lpd(fpd_path, lpd_path, seq_len=256):
    """
    Build a fixed-length sequence [seq_len, 2] from FPD and LPD traces.
    We resample each temperature trace to 'seq_len' points using interpolation
    over index (not physical time).
    """
    _, temp_fpd = load_temp_file(fpd_path)
    _, temp_lpd = load_temp_file(lpd_path)

    # Ensure both have at least 2 samples
    if len(temp_fpd) < 2 or len(temp_lpd) < 2:
        # pad with copies if too short
        temp_fpd = np.pad(temp_fpd, (0, max(0, 2-len(temp_fpd))), mode='edge')
        temp_lpd = np.pad(temp_lpd, (0, max(0, 2-len(temp_lpd))), mode='edge')

    def resample(temp, L):
        idx_orig = np.arange(len(temp))
        idx_new = np.linspace(0, len(temp) - 1, L)
        return np.interp(idx_new, idx_orig, temp)

    fpd_res = resample(temp_fpd, seq_len)
    lpd_res = resample(temp_lpd, seq_len)

    # Normalize each channel (zero mean, unit variance) per run
    def norm(x):
        m = x.mean()
        s = x.std()
        if s < 1e-6:
            s = 1.0
        return (x - m) / s

    fpd_norm = norm(fpd_res)
    lpd_norm = norm(lpd_res)

    seq = np.stack([fpd_norm, lpd_norm], axis=-1)  # [seq_len, 2]
    return seq.astype(np.float32)


# ===============================
# 2. Build train/test datasets
# ===============================

BASE_DIR = "."

label_map = {
    "Idle": 0,
    "Radar": 1,
    "Pulse_Doppler": 2,
    "SAR": 3,
    "WiFi": 4,
}
inv_label_map = {v: k for k, v in label_map.items()}

single_app_root = os.path.join(BASE_DIR, "single_app_runs")

app_dir_map = {
    "radar_correlator_run": "Radar",
    "pulse_doppler_run": "Pulse_Doppler",
    "sar_runs": "SAR",
    "wifi_runs": "WiFi",
}

TRAIN_RUNS = {"run_1", "run_2", "run_3", "run_4"}
TEST_RUNS = {"run_5"}

SEQ_LEN = 256

X_train, y_train = [], []
X_test, y_test = [], []
test_file_info = []


# ---- 2a. Idle: 2 train, 1 test ----

idle_train_dirs = [
    "idle_after_multi_apps_run_1",
    "idle_after_multi_apps_run_2",
]
idle_test_dirs = [
    "ilde_nov_29_night",
]

for idle_dir in idle_train_dirs:
    full_idle_dir = os.path.join(BASE_DIR, idle_dir)
    if not os.path.isdir(full_idle_dir):
        print(f"[WARN] Idle TRAIN directory not found: {full_idle_dir}")
        continue

    fpd_files = glob.glob(os.path.join(full_idle_dir, "*fpd_measured_temp*"))
    lpd_files = glob.glob(os.path.join(full_idle_dir, "*lpd_measured_temp*"))

    if len(fpd_files) != 1 or len(lpd_files) != 1:
        print(f"[WARN] Unexpected idle TRAIN files in {idle_dir}: fpd={fpd_files}, lpd={lpd_files}")
        continue

    seq = build_sequence_fpd_lpd(fpd_files[0], lpd_files[0], seq_len=SEQ_LEN)
    X_train.append(seq)
    y_train.append(label_map["Idle"])

for idle_dir in idle_test_dirs:
    full_idle_dir = os.path.join(BASE_DIR, idle_dir)
    if not os.path.isdir(full_idle_dir):
        print(f"[WARN] Idle TEST directory not found: {full_idle_dir}")
        continue

    fpd_files = glob.glob(os.path.join(full_idle_dir, "*fpd_measured_temp*"))
    lpd_files = glob.glob(os.path.join(full_idle_dir, "*lpd_measured_temp*"))

    if len(fpd_files) != 1 or len(lpd_files) != 1:
        print(f"[WARN] Unexpected idle TEST files in {idle_dir}: fpd={fpd_files}, lpd={lpd_files}")
        continue

    seq = build_sequence_fpd_lpd(fpd_files[0], lpd_files[0], seq_len=SEQ_LEN)
    X_test.append(seq)
    y_test.append(label_map["Idle"])
    test_file_info.append((fpd_files[0], lpd_files[0], "Idle"))


# ---- 2b. Single-app runs ----

for subdir, label_str in app_dir_map.items():
    app_dir = os.path.join(single_app_root, subdir)
    if not os.path.isdir(app_dir):
        print(f"[WARN] App directory not found: {app_dir}")
        continue

    run_dirs = sorted(d for d in os.listdir(app_dir)
                      if d.startswith("run_") and os.path.isdir(os.path.join(app_dir, d)))

    print(f"[INFO] Found runs {run_dirs} for {label_str} in {subdir}")
    label = label_map[label_str]

    for run in run_dirs:
        run_path = os.path.join(app_dir, run)

        fpd_files = sorted(glob.glob(os.path.join(run_path, "*fpd_measured_temp*")))
        lpd_files = sorted(glob.glob(os.path.join(run_path, "*lpd_measured_temp*")))

        if len(fpd_files) == 0 or len(lpd_files) == 0:
            print(f"[WARN] No temp files in {run_path} (fpd={fpd_files}, lpd={lpd_files})")
            continue

        for fpd_path, lpd_path in zip(fpd_files, lpd_files):
            seq = build_sequence_fpd_lpd(fpd_path, lpd_path, seq_len=SEQ_LEN)
            if run in TRAIN_RUNS:
                X_train.append(seq)
                y_train.append(label)
            elif run in TEST_RUNS:
                X_test.append(seq)
                y_test.append(label)
                test_file_info.append((fpd_path, lpd_path, label_str))
            else:
                print(f"[WARN] Ignoring {run_path} (not in TRAIN_RUNS or TEST_RUNS)")


X_train = np.stack(X_train)  # [N_train, seq_len, 2]
y_train = np.array(y_train)

X_test = np.stack(X_test)
y_test = np.array(y_test)

print(f"\n[INFO] Training dataset: X_train shape = {X_train.shape}, y_train size = {len(y_train)}")
print("[INFO] Training class counts:")
for lab, cnt in zip(*np.unique(y_train, return_counts=True)):
    print(f"   {inv_label_map[lab]:14s}: {cnt}")

print(f"\n[INFO] Test dataset: X_test shape = {X_test.shape}, y_test size = {len(y_test)}")
print("[INFO] Test class counts:")
for lab, cnt in zip(*np.unique(y_test, return_counts=True)):
    print(f"   {inv_label_map[lab]:14s}: {cnt}")


# ===============================
# 3. PyTorch Dataset & DataLoader
# ===============================

class TempSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)  # [N, T, 2]
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TempSeqDataset(X_train, y_train)
test_dataset = TempSeqDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# ===============================
# 4. LSTM Model Definition
# ===============================

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, (h_n, c_n) = self.lstm(x)
        # h_n: [num_layers, batch, hidden_size]
        h_last = h_n[-1]  # [batch, hidden_size]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n[INFO] Using device: {device}")

model = LSTMClassifier(input_size=2, hidden_size=64, num_layers=1, num_classes=len(label_map))
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ===============================
# 5. Train Loop
# ===============================

NUM_EPOCHS = 80

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(train_dataset)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}, train loss = {avg_loss:.4f}")


# ===============================
# 6. Evaluate on test set
# ===============================

model.eval()
all_preds = []
all_true = []

with torch.inference_mode():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(list(preds))
        all_true.extend(list(batch_y.numpy()))

all_preds = np.array(all_preds)
all_true = np.array(all_true)

print("\n=== LSTM test set evaluation (run_5 + idle) ===")
target_names = [inv_label_map[i] for i in range(len(label_map))]
print(classification_report(all_true, all_preds, target_names=target_names))
print("Confusion matrix (test):")
print(confusion_matrix(all_true, all_preds))
print("Label order:", target_names)

print("\n=== File-by-file predictions ===")
for (fp, lp, true_label), pred_idx in zip(test_file_info, all_preds):
    pred_label = inv_label_map[pred_idx]
    print(f"{os.path.basename(fp):60s}  true={true_label:14s}  pred={pred_label}")


# ===============================
# 7. Inference helper for a new run
# ===============================

def predict_run(fpd_path, lpd_path):
    seq = build_sequence_fpd_lpd(fpd_path, lpd_path, seq_len=SEQ_LEN)
    x = torch.from_numpy(seq).unsqueeze(0).to(device)  # [1, T, 2]
    model.eval()
    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
    return inv_label_map[pred_idx], {inv_label_map[i]: float(p) for i, p in enumerate(probs)}

# Example usage (uncomment and adjust paths):
# fp = "single_app_runs/radar_correlator_run/run_5/radar_5_fpd_measured_temp.txt"
# lp = "single_app_runs/radar_correlator_run/run_5/radar_5_lpd_measured_temp.txt"
# pred_label, prob_dict = predict_run(fp, lp)
# print("Predicted class:", pred_label)
# print("Class probabilities:", prob_dict)

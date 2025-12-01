import os
import glob
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==========================
# 1. Data loading utilities
# ==========================

def load_temp_file(path):
    """
    Load a temp trace file of the form:
      <temp> C <timestamp> ns
    Returns t_s (seconds) and temp_C (numpy arrays).
    """
    # Only read numeric columns: temp (0) and timestamp (2)
    data = np.loadtxt(path, usecols=(0, 2))

    # Handle single-line files
    if data.ndim == 1:
        data = data.reshape(1, -1)

    temp = data[:, 0]
    t_ns = data[:, 1]
    t_s = (t_ns - t_ns[0]) * 1e-9   # relative seconds

    return t_s, temp


def extract_features_from_trace(t, temp):
    """
    Extract richer time-series features from a single sensor trace.
    """
    N = len(temp)
    if N == 0:
        return np.zeros(16, dtype=float)

    mean = temp.mean()
    std = temp.std()
    tmax = temp.max()
    tmin = temp.min()
    rng = tmax - tmin

    dur = t[-1] - t[0] if N > 1 else 0.0
    slope = (temp[-1] - temp[0]) / (dur + 1e-9)

    median = np.median(temp)
    q25 = np.percentile(temp, 25)
    q75 = np.percentile(temp, 75)

    if N > 2:
        grad = np.gradient(temp)
        grad_abs_mean = np.mean(np.abs(grad))
        grad_max = np.max(grad)
        grad_min = np.min(grad)
        grad_std = np.std(grad)
    else:
        grad_abs_mean = grad_max = grad_min = grad_std = 0.0

    # FFT features
    x = temp - mean
    fft_mag = np.abs(np.fft.rfft(x))
    if len(fft_mag) >= 3:
        top3 = np.partition(fft_mag, -3)[-3:]
    else:
        top3 = np.pad(fft_mag, (0, max(0, 3 - len(fft_mag))), mode="constant")

    f1, f2, f3 = top3

    return np.array([
        mean, std, rng, tmax, tmin,
        dur, slope,
        median, q25, q75,
        grad_abs_mean, grad_max, grad_min, grad_std,
        f1, f2, f3
    ])


def extract_features_fpd_lpd(fpd_path, lpd_path):
    """
    Load FPD and LPD temp traces and concatenate their feature vectors.
    """
    t_fpd, temp_fpd = load_temp_file(fpd_path)
    t_lpd, temp_lpd = load_temp_file(lpd_path)

    feats_fpd = extract_features_from_trace(t_fpd, temp_fpd)
    feats_lpd = extract_features_from_trace(t_lpd, temp_lpd)

    return np.concatenate([feats_fpd, feats_lpd])


def split_indices(N, n_chunks):
    """
    Return list of (start, end) indices splitting range [0, N) into n_chunks.
    """
    if n_chunks <= 1 or N <= n_chunks:
        return [(0, N)]
    chunk_size = N // n_chunks
    idx = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < n_chunks - 1 else N
        idx.append((start, end))
    return idx


def extract_window_features_fpd_lpd(fpd_path, lpd_path, n_chunks):
    """
    For a given FPD/LPD file pair, split traces into n_chunks and
    return list of feature vectors (one per chunk).
    """
    t_fpd, temp_fpd = load_temp_file(fpd_path)
    t_lpd, temp_lpd = load_temp_file(lpd_path)

    N = min(len(temp_fpd), len(temp_lpd))
    if N == 0:
        return []

    idxs = split_indices(N, n_chunks)

    feats_list = []
    for (start, end) in idxs:
        t_f = t_fpd[start:end]
        x_f = temp_fpd[start:end]
        t_l = t_lpd[start:end]
        x_l = temp_lpd[start:end]

        feats_fpd = extract_features_from_trace(t_f, x_f)
        feats_lpd = extract_features_from_trace(t_l, x_l)
        feats_list.append(np.concatenate([feats_fpd, feats_lpd]))

    return feats_list


# ===============================
# 2. Build datasets
#    - train: idle + single_app_runs (full) + single_app_multi_runs/run_1 (windowed)
#    - test : single_app_multi_runs/run_2 (windowed)
# ===============================

BASE_DIR = "."

X_train_all = []
y_train_all = []

X_test_multi = []      # external multi-run test (window-level)
y_test_multi = []
test_file_ids = []     # (fpd_path, lpd_path, chunk_idx)

N_CHUNKS_MULTI = 10    # number of windows for multi-run traces


# ---- 2a. Idle runs (all used for training, full-trace) ----

idle_dirs = [
    "idle_after_multi_apps_run_1",
    "idle_after_multi_apps_run_2",
    "ilde_nov_29_night"
]

for idle_dir in idle_dirs:
    full_idle_dir = os.path.join(BASE_DIR, idle_dir)
    if not os.path.isdir(full_idle_dir):
        print(f"[WARN] Idle directory not found: {full_idle_dir}")
        continue

    fpd_files = glob.glob(os.path.join(full_idle_dir, "*fpd_measured_temp*"))
    lpd_files = glob.glob(os.path.join(full_idle_dir, "*lpd_measured_temp*"))

    if len(fpd_files) != 1 or len(lpd_files) != 1:
        print(f"[WARN] Unexpected idle files in {idle_dir}: fpd={fpd_files}, lpd={lpd_files}")
        continue

    feats = extract_features_fpd_lpd(fpd_files[0], lpd_files[0])
    X_train_all.append(feats)
    y_train_all.append("Idle")


# ---- 2b. Single-app runs (run_1...run_5) used for training, full-trace ----

single_app_root = os.path.join(BASE_DIR, "single_app_runs")

label_map = {
    "radar_correlator_run": "Radar",
    "pulse_doppler_run": "Pulse_Doppler",
    "sar_runs": "SAR",
    "wifi_runs": "WiFi",
}

for subdir, label in label_map.items():
    app_dir = os.path.join(single_app_root, subdir)
    if not os.path.isdir(app_dir):
        print(f"[WARN] App directory not found: {app_dir}")
        continue

    run_dirs = sorted(d for d in os.listdir(app_dir)
                      if d.startswith("run_") and os.path.isdir(os.path.join(app_dir, d)))

    print(f"[INFO] Found {len(run_dirs)} runs for {label} in {subdir}")

    for run in run_dirs:
        run_path = os.path.join(app_dir, run)

        fpd_files = sorted(glob.glob(os.path.join(run_path, "*fpd_measured_temp*")))
        lpd_files = sorted(glob.glob(os.path.join(run_path, "*lpd_measured_temp*")))

        if len(fpd_files) == 0 or len(lpd_files) == 0:
            print(f"[WARN] No temp files in {run_path} (fpd={fpd_files}, lpd={lpd_files})")
            continue

        for fpd_path, lpd_path in zip(fpd_files, lpd_files):
            feats = extract_features_fpd_lpd(fpd_path, lpd_path)
            X_train_all.append(feats)
            y_train_all.append(label)


# ---- 2c. single_app_multi_runs: run_1 → windowed training, run_2 → windowed external test ----

multi_root = os.path.join(BASE_DIR, "single_app_multi_runs")

multi_label_map = {
    "radar_multi_runs_10_p_30_sec_periodicity": "Radar",
    "pulse_doppler_multi_runs_10_apps_with_30_sec_periodicity": "Pulse_Doppler",
    "sar_multi_runs_10_apps_with_30_sec_periodicity": "SAR",
    "wifi_multi_runs_10_apps_with_30_sec_periodicity": "WiFi",
}

for subdir, label in multi_label_map.items():
    app_dir = os.path.join(multi_root, subdir)
    if not os.path.isdir(app_dir):
        print(f"[WARN] Multi-run app directory not found: {app_dir}")
        continue

    for run_name in ["run_1", "run_2"]:
        run_path = os.path.join(app_dir, run_name)
        if not os.path.isdir(run_path):
            print(f"[WARN] Missing {run_name} in {app_dir}")
            continue

        fpd_files = sorted(glob.glob(os.path.join(run_path, "*fpd_measured_temp*")))
        lpd_files = sorted(glob.glob(os.path.join(run_path, "*lpd_measured_temp*")))

        if len(fpd_files) == 0 or len(lpd_files) == 0:
            print(f"[WARN] No temp files in {run_path} (fpd={fpd_files}, lpd={lpd_files})")
            continue

        for fpd_path, lpd_path in zip(fpd_files, lpd_files):
            chunk_feats = extract_window_features_fpd_lpd(fpd_path, lpd_path, N_CHUNKS_MULTI)

            if run_name == "run_1":
                # use all chunks for training
                for feats in chunk_feats:
                    X_train_all.append(feats)
                    y_train_all.append(label)
            else:
                # run_2 → external test
                for i, feats in enumerate(chunk_feats):
                    X_test_multi.append(feats)
                    y_test_multi.append(label)
                    test_file_ids.append((fpd_path, lpd_path, i))


X_train_all = np.vstack(X_train_all)
y_train_all = np.array(y_train_all)

X_test_multi = np.vstack(X_test_multi) if len(X_test_multi) > 0 else None
y_test_multi = np.array(y_test_multi) if len(y_test_multi) > 0 else None

print(f"\n[INFO] Training dataset built: X_train_all shape = {X_train_all.shape}, y size = {len(y_train_all)}")
print("[INFO] Training class counts:")
unique, counts = np.unique(y_train_all, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"   {cls:14s}: {cnt}")

if X_test_multi is not None:
    print(f"\n[INFO] External multi-run test set (window-level): X_test_multi shape = {X_test_multi.shape}, size = {len(y_test_multi)}")
    print("[INFO] Test set class counts (window-level):")
    unique_t, counts_t = np.unique(y_test_multi, return_counts=True)
    for cls, cnt in zip(unique_t, counts_t):
        print(f"   {cls:14s}: {cnt}")
else:
    print("\n[WARN] No external multi-run test set built.")


# ===============================
# 3. Train / validation split and model
# ===============================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all,
    test_size=0.3,
    random_state=42,
    stratify=y_train_all
)

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42
)

model = make_pipeline(StandardScaler(), rf_clf)

model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)

print("\n=== Validation set (holdout from training data, sample-level) ===")
print(classification_report(y_val, y_val_pred))
print("Confusion matrix (val):")
print(confusion_matrix(y_val, y_val_pred, labels=["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"]))
print("Labels order:", ["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"])


# ===============================
# 4. Evaluate on external multi-run test set (window-level + file-level)
# ===============================

if X_test_multi is not None:
    y_test_pred = model.predict(X_test_multi)

    print("\n=== External test: single_app_multi_runs run_2 (window-level) ===")
    print(classification_report(y_test_multi, y_test_pred))
    print("Confusion matrix (external multi-runs, window-level):")
    print(confusion_matrix(y_test_multi, y_test_pred, labels=["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"]))
    print("Labels order:", ["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"])

    # File-level majority vote
    print("\n=== External test: file-level majority vote (run_2 per app) ===")
    from collections import defaultdict, Counter

    # group by file (fpd_path)
    file_to_preds = defaultdict(list)
    file_to_true = {}

    for (fp, lp, chunk_idx), true_label, pred_label in zip(test_file_ids, y_test_multi, y_test_pred):
        file_to_preds[fp].append(pred_label)
        file_to_true[fp] = true_label

    for fp in sorted(file_to_preds.keys()):
        preds = file_to_preds[fp]
        majority = Counter(preds).most_common(1)[0][0]
        true_label = file_to_true[fp]
        print(f"{os.path.basename(fp):60s}  true={true_label:14s}  majority_pred={majority}")
else:
    print("\n[WARN] Skipping external multi-run test; none was built.")


# ===============================
# 5. Inference helper on new files (full-run)
# ===============================

def predict_run(fpd_path, lpd_path):
    feats = extract_features_fpd_lpd(fpd_path, lpd_path).reshape(1, -1)
    pred = model.predict(feats)[0]
    proba = model.predict_proba(feats)[0]
    return pred, dict(zip(model.classes_, proba))

# Example usage:
# fp = "single_app_runs/radar_correlator_run/run_1/radar_1_fpd_measured_temp.txt"
# lp = "single_app_runs/radar_correlator_run/run_1/radar_1_lpd_measured_temp.txt"
# p, pr = predict_run(fp, lp)
# print("Prediction:", p)
# print("Probabilities:", pr)

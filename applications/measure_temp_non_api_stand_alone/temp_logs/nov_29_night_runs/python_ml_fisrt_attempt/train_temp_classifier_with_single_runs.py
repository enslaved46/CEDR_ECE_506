import os
import glob
import numpy as np

from sklearn.ensemble import RandomForestClassifier
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


# ===============================
# 2. Build datasets
#    - Train: idle(1,2) + single_app_runs run_1..run_4
#    - Test : idle(night) + single_app_runs run_5
# ===============================

BASE_DIR = "."

X_train = []
y_train = []
X_test = []
y_test = []
test_file_info = []   # keep (fpd_path, lpd_path, label) for pretty printing

# ---- 2a. Idle: use 2 for train, 1 for test ----

idle_train_dirs = [
    "idle_after_multi_apps_run_1",
    "idle_after_multi_apps_run_2"
]
idle_test_dirs = [
    "ilde_nov_29_night"
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

    feats = extract_features_fpd_lpd(fpd_files[0], lpd_files[0])
    X_train.append(feats)
    y_train.append("Idle")

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

    feats = extract_features_fpd_lpd(fpd_files[0], lpd_files[0])
    X_test.append(feats)
    y_test.append("Idle")
    test_file_info.append((fpd_files[0], lpd_files[0], "Idle"))


# ---- 2b. Single-app runs: train on run_1..run_4, test on run_5 ----

single_app_root = os.path.join(BASE_DIR, "single_app_runs")

label_map = {
    "radar_correlator_run": "Radar",
    "pulse_doppler_run": "Pulse_Doppler",
    "sar_runs": "SAR",
    "wifi_runs": "WiFi",
}

TRAIN_RUNS = {"run_1", "run_2", "run_3", "run_4"}
TEST_RUNS = {"run_5"}

for subdir, label in label_map.items():
    app_dir = os.path.join(single_app_root, subdir)
    if not os.path.isdir(app_dir):
        print(f"[WARN] App directory not found: {app_dir}")
        continue

    run_dirs = sorted(d for d in os.listdir(app_dir)
                      if d.startswith("run_") and os.path.isdir(os.path.join(app_dir, d)))

    print(f"[INFO] Found runs {run_dirs} for {label} in {subdir}")

    for run in run_dirs:
        run_path = os.path.join(app_dir, run)

        fpd_files = sorted(glob.glob(os.path.join(run_path, "*fpd_measured_temp*")))
        lpd_files = sorted(glob.glob(os.path.join(run_path, "*lpd_measured_temp*")))

        if len(fpd_files) == 0 or len(lpd_files) == 0:
            print(f"[WARN] No temp files in {run_path} (fpd={fpd_files}, lpd={lpd_files})")
            continue

        # each pair (fpd, lpd) = one sample
        for fpd_path, lpd_path in zip(fpd_files, lpd_files):
            feats = extract_features_fpd_lpd(fpd_path, lpd_path)
            if run in TRAIN_RUNS:
                X_train.append(feats)
                y_train.append(label)
            elif run in TEST_RUNS:
                X_test.append(feats)
                y_test.append(label)
                test_file_info.append((fpd_path, lpd_path, label))
            else:
                # if some other run_* appears, ignore
                print(f"[WARN] Ignoring {run_path} (not in TRAIN_RUNS or TEST_RUNS)")


X_train = np.vstack(X_train)
y_train = np.array(y_train)

X_test = np.vstack(X_test)
y_test = np.array(y_test)

print(f"\n[INFO] Training dataset: X_train shape = {X_train.shape}, y_train size = {len(y_train)}")
print("[INFO] Training class counts:")
unique_tr, counts_tr = np.unique(y_train, return_counts=True)
for cls, cnt in zip(unique_tr, counts_tr):
    print(f"   {cls:14s}: {cnt}")

print(f"\n[INFO] Test dataset: X_test shape = {X_test.shape}, y_test size = {len(y_test)}")
print("[INFO] Test class counts:")
unique_te, counts_te = np.unique(y_test, return_counts=True)
for cls, cnt in zip(unique_te, counts_te):
    print(f"   {cls:14s}: {cnt}")


# ===============================
# 3. Train model on all training data
# ===============================

rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    random_state=42
)

model = make_pipeline(StandardScaler(), rf_clf)

model.fit(X_train, y_train)

# ===============================
# 4. Evaluate on test set (run_5 + idle_night)
# ===============================

y_pred = model.predict(X_test)

print("\n=== Test set evaluation (single_app_runs run_5 + idle) ===")
print(classification_report(y_test, y_pred))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_pred, labels=["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"]))
print("Labels order:", ["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"])

print("\n=== File-by-file predictions ===")
for (fp, lp, true_label), pred_label in zip(test_file_info, y_pred):
    print(f"{os.path.basename(fp):60s}  true={true_label:14s}  pred={pred_label}")


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

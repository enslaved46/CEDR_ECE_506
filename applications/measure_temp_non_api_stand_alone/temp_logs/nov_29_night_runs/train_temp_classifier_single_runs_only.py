import os
import glob
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ==========================
# 1. Data loading utilities
# ==========================

def load_temp_file(path):
    """
    Load <temp> C <timestamp> ns
    Return (t_seconds, temp_array).
    """
    data = np.loadtxt(path, usecols=(0, 2))
    if data.ndim == 1:
        data = data.reshape(1, -1)

    temp = data[:, 0]
    t_ns = data[:, 1]
    t_s = (t_ns - t_ns[0]) * 1e-9
    return t_s, temp


def extract_features_from_trace(t, temp):
    """
    10 simple features from one trace.
    """
    mean = temp.mean()
    std = temp.std()
    tmax = temp.max()
    tmin = temp.min()
    rng = tmax - tmin

    dur = t[-1] - t[0] if len(t) > 1 else 0.0
    slope = (temp[-1] - temp[0]) / (dur + 1e-9)

    if len(temp) > 2:
        grad = np.gradient(temp)
        smooth = np.mean(np.abs(grad))
    else:
        smooth = 0.0

    x = temp - mean
    fft_mag = np.abs(np.fft.rfft(x))
    if len(fft_mag) >= 3:
        top3 = np.partition(fft_mag, -3)[-3:]
    else:
        top3 = np.pad(fft_mag, (0, max(0, 3 - len(fft_mag))), mode="constant")

    f1, f2, f3 = top3

    return np.array([
        mean, std, rng, tmax, tmin,
        slope, smooth,
        f1, f2, f3
    ])


def features_fpd(fpd_path):
    t, temp = load_temp_file(fpd_path)
    return extract_features_from_trace(t, temp)


def features_lpd(lpd_path):
    t, temp = load_temp_file(lpd_path)
    return extract_features_from_trace(t, temp)


def features_fpd_lpd(fpd_path, lpd_path):
    f_fpd = features_fpd(fpd_path)
    f_lpd = features_lpd(lpd_path)
    return np.concatenate([f_fpd, f_lpd])  # 20-dim


# ==========================
# 2. Build datasets (single runs only)
# ==========================

BASE_DIR = "."

TRAIN_RUNS = {"run_1", "run_2", "run_3", "run_4"}
TEST_RUNS  = {"run_5"}

label_map = {
    "radar_correlator_run": "Radar",
    "pulse_doppler_run":    "Pulse_Doppler",
    "sar_runs":             "SAR",
    "wifi_runs":            "WiFi",
}

single_app_root = os.path.join(BASE_DIR, "single_app_runs")

# We will build THREE parallel training sets and THREE test sets:
X_train_full, y_train_full = [], []
X_train_fpd,  y_train_fpd  = [], []
X_train_lpd,  y_train_lpd  = [], []

X_test_full,  y_test_full  = [], []
X_test_fpd,   y_test_fpd   = [], []
X_test_lpd,   y_test_lpd   = [], []

test_file_info = []  # (fpd_path, lpd_path, label)


# ---- 2a. Idle: train on two, test on night ----

idle_train_dirs = [
    "idle_after_multi_apps_run_1",
    "idle_after_multi_apps_run_2",
]
idle_test_dirs = [
    "ilde_nov_29_night",
]

for idle_dir in idle_train_dirs:
    full_dir = os.path.join(BASE_DIR, idle_dir)
    if not os.path.isdir(full_dir):
        print(f"[WARN] Idle TRAIN dir not found: {full_dir}")
        continue

    fpd_files = glob.glob(os.path.join(full_dir, "*fpd_measured_temp*"))
    lpd_files = glob.glob(os.path.join(full_dir, "*lpd_measured_temp*"))

    if len(fpd_files) != 1 or len(lpd_files) != 1:
        print(f"[WARN] Unexpected idle TRAIN files in {idle_dir}: fpd={fpd_files}, lpd={lpd_files}")
        continue

    fp, lp = fpd_files[0], lpd_files[0]

    f_fpd  = features_fpd(fp)
    f_lpd  = features_lpd(lp)
    f_full = np.concatenate([f_fpd, f_lpd])

    X_train_fpd.append(f_fpd)
    y_train_fpd.append("Idle")

    X_train_lpd.append(f_lpd)
    y_train_lpd.append("Idle")

    X_train_full.append(f_full)
    y_train_full.append("Idle")

for idle_dir in idle_test_dirs:
    full_dir = os.path.join(BASE_DIR, idle_dir)
    if not os.path.isdir(full_dir):
        print(f"[WARN] Idle TEST dir not found: {full_dir}")
        continue

    fpd_files = glob.glob(os.path.join(full_dir, "*fpd_measured_temp*"))
    lpd_files = glob.glob(os.path.join(full_dir, "*lpd_measured_temp*"))

    if len(fpd_files) != 1 or len(lpd_files) != 1:
        print(f"[WARN] Unexpected idle TEST files in {idle_dir}: fpd={fpd_files}, lpd={lpd_files}")
        continue

    fp, lp = fpd_files[0], lpd_files[0]

    f_fpd  = features_fpd(fp)
    f_lpd  = features_lpd(lp)
    f_full = np.concatenate([f_fpd, f_lpd])

    X_test_fpd.append(f_fpd)
    y_test_fpd.append("Idle")

    X_test_lpd.append(f_lpd)
    y_test_lpd.append("Idle")

    X_test_full.append(f_full)
    y_test_full.append("Idle")

    test_file_info.append((fp, lp, "Idle"))


# ---- 2b. Single-app runs ----

for subdir, label in label_map.items():
    app_dir = os.path.join(single_app_root, subdir)
    if not os.path.isdir(app_dir):
        print(f"[WARN] App directory not found: {app_dir}")
        continue

    run_dirs = sorted(
        d for d in os.listdir(app_dir)
        if d.startswith("run_") and os.path.isdir(os.path.join(app_dir, d))
    )

    print(f"[INFO] Found runs {run_dirs} for {label} in {subdir}")

    for run in run_dirs:
        run_path = os.path.join(app_dir, run)

        fpd_files = sorted(glob.glob(os.path.join(run_path, "*fpd_measured_temp*")))
        lpd_files = sorted(glob.glob(os.path.join(run_path, "*lpd_measured_temp*")))

        if len(fpd_files) == 0 or len(lpd_files) == 0:
            print(f"[WARN] No temp files in {run_path} (fpd={fpd_files}, lpd={lpd_files})")
            continue

        for fp, lp in zip(fpd_files, lpd_files):
            f_fpd  = features_fpd(fp)
            f_lpd  = features_lpd(lp)
            f_full = np.concatenate([f_fpd, f_lpd])

            if run in TRAIN_RUNS:
                X_train_fpd.append(f_fpd)
                y_train_fpd.append(label)

                X_train_lpd.append(f_lpd)
                y_train_lpd.append(label)

                X_train_full.append(f_full)
                y_train_full.append(label)

            elif run in TEST_RUNS:
                X_test_fpd.append(f_fpd)
                y_test_fpd.append(label)

                X_test_lpd.append(f_lpd)
                y_test_lpd.append(label)

                X_test_full.append(f_full)
                y_test_full.append(label)

                test_file_info.append((fp, lp, label))
            else:
                print(f"[WARN] Ignoring {run_path} (run not in TRAIN_RUNS or TEST_RUNS)")


# Convert to numpy arrays
X_train_fpd  = np.vstack(X_train_fpd)
y_train_fpd  = np.array(y_train_fpd)

X_train_lpd  = np.vstack(X_train_lpd)
y_train_lpd  = np.array(y_train_lpd)

X_train_full = np.vstack(X_train_full)
y_train_full = np.array(y_train_full)

X_test_fpd   = np.vstack(X_test_fpd)
y_test_fpd   = np.array(y_test_fpd)

X_test_lpd   = np.vstack(X_test_lpd)
y_test_lpd   = np.array(y_test_lpd)

X_test_full  = np.vstack(X_test_full)
y_test_full  = np.array(y_test_full)


print("\n[INFO] Training shapes:")
print("  FPD : ", X_train_fpd.shape)
print("  LPD : ", X_train_lpd.shape)
print("  FULL: ", X_train_full.shape)

print("\n[INFO] Test shapes:")
print("  FPD : ", X_test_fpd.shape)
print("  LPD : ", X_test_lpd.shape)
print("  FULL: ", X_test_full.shape)


# ==========================
# 3. Train three models
# ==========================

def make_model():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=400, random_state=42)
    )

model_fpd  = make_model()
model_lpd  = make_model()
model_full = make_model()

model_fpd.fit(X_train_fpd,  y_train_fpd)
model_lpd.fit(X_train_lpd,  y_train_lpd)
model_full.fit(X_train_full, y_train_full)


# ==========================
# 4. Evaluate
# ==========================

labels_order = ["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"]

print("\n========== FPD-only Test ==========")
y_pred_fpd = model_fpd.predict(X_test_fpd)
print(classification_report(y_test_fpd, y_pred_fpd))
print(confusion_matrix(y_test_fpd, y_pred_fpd, labels=labels_order))
print("Labels order:", labels_order)

print("\n========== LPD-only Test ==========")
y_pred_lpd = model_lpd.predict(X_test_lpd)
print(classification_report(y_test_lpd, y_pred_lpd))
print(confusion_matrix(y_test_lpd, y_pred_lpd, labels=labels_order))
print("Labels order:", labels_order)

print("\n========== FULL (FPD+LPD) Test ==========")
y_pred_full = model_full.predict(X_test_full)
print(classification_report(y_test_full, y_pred_full))
print(confusion_matrix(y_test_full, y_pred_full, labels=labels_order))
print("Labels order:", labels_order)

# ------------------------------------
# Per-file predictions for each model
# ------------------------------------

print("\n=== File-by-file predictions (FPD-only model) ===")
for (fp, lp, true_label), pred_label in zip(test_file_info, y_pred_fpd):
    print(f"{os.path.basename(fp):60s}  true={true_label:14s}  pred_FPD={pred_label}")

print("\n=== File-by-file predictions (LPD-only model) ===")
for (fp, lp, true_label), pred_label in zip(test_file_info, y_pred_lpd):
    print(f"{os.path.basename(lp):60s}  true={true_label:14s}  pred_LPD={pred_label}")

print("\n=== File-by-file predictions (FULL model) ===")
for (fp, lp, true_label), pred_label in zip(test_file_info, y_pred_full):
    print(f"{os.path.basename(fp):60s}  true={true_label:14s}  pred_FULL={pred_label}")

import os
import glob
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ==========================
# 1. Data loading utilities
# ==========================
def load_temp_file(path):
    """
    Load a temp trace file of the form:
    <temp> C <timestamp> ns
    Returns t_s (seconds) and temp_C (numpy arrays).
    """
    # Only read the numeric columns: 0 (temp) and 2 (timestamp)
    data = np.loadtxt(path, usecols=(0, 2))

    # If there's only one line, make sure it's 2D
    if data.ndim == 1:
        data = data.reshape(1, -1)

    temp = data[:, 0]
    t_ns = data[:, 1]
    t_s = (t_ns - t_ns[0]) * 1e-9   # relative seconds

    return t_s, temp


def extract_features_from_trace(t, temp):
    """
    Extract simple time-series features from a single sensor trace.
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
#    - train: idle + single_app_runs + single_app_multi_runs/run_1
#    - test : single_app_multi_runs/run_2
# ===============================

BASE_DIR = "."

X_train_all = []
y_train_all = []

X_test_multi = []   # only run_2 from single_app_multi_runs
y_test_multi = []

# ---- 2a. Idle runs (all used for training) ----

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

    fpd_files = glob.glob(os.path.join(full_idle_dir, "*fpd_measured_temp.txt"))
    lpd_files = glob.glob(os.path.join(full_idle_dir, "*lpd_measured_temp.txt"))

    if len(fpd_files) != 1 or len(lpd_files) != 1:
        print(f"[WARN] Unexpected idle files in {idle_dir}: fpd={fpd_files}, lpd={lpd_files}")
        continue

    feats = extract_features_fpd_lpd(fpd_files[0], lpd_files[0])
    X_train_all.append(feats)
    y_train_all.append("Idle")


# ---- 2b. Single-app runs (run_1...run_5) used for training ----

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

        fpd_files = sorted(glob.glob(os.path.join(run_path, "*fpd_measured_temp.txt")))
        lpd_files = sorted(glob.glob(os.path.join(run_path, "*lpd_measured_temp.txt")))

        if len(fpd_files) == 0 or len(lpd_files) == 0:
            print(f"[WARN] No temp files in {run_path} (fpd={fpd_files}, lpd={lpd_files})")
            continue

        for fpd_path, lpd_path in zip(fpd_files, lpd_files):
            feats = extract_features_fpd_lpd(fpd_path, lpd_path)
            X_train_all.append(feats)
            y_train_all.append(label)


# ---- 2c. single_app_multi_runs: run_1 → training, run_2 → external test ----

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

    # Expect run_1 and run_2
    for run_name in ["run_1", "run_2"]:
        run_path = os.path.join(app_dir, run_name)
        if not os.path.isdir(run_path):
            print(f"[WARN] Missing {run_name} in {app_dir}")
            continue

        fpd_files = sorted(glob.glob(os.path.join(run_path, "*fpd_measured_temp.txt")))
        lpd_files = sorted(glob.glob(os.path.join(run_path, "*lpd_measured_temp.txt")))

        if len(fpd_files) == 0 or len(lpd_files) == 0:
            print(f"[WARN] No temp files in {run_path} (fpd={fpd_files}, lpd={lpd_files})")
            continue

        for fpd_path, lpd_path in zip(fpd_files, lpd_files):
            feats = extract_features_fpd_lpd(fpd_path, lpd_path)

            if run_name == "run_1":
                # use for training
                X_train_all.append(feats)
                y_train_all.append(label)
            else:
                # run_2 → external test set
                X_test_multi.append(feats)
                y_test_multi.append(label)


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
    print(f"\n[INFO] External multi-run test set: X_test_multi shape = {X_test_multi.shape}, size = {len(y_test_multi)}")
    print("[INFO] Test set class counts:")
    unique_t, counts_t = np.unique(y_test_multi, return_counts=True)
    for cls, cnt in zip(unique_t, counts_t):
        print(f"   {cls:14s}: {cnt}")
else:
    print("\n[WARN] No external multi-run test set built.")


# ===============================
# 3. Train / validation split and model
# ===============================

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all,
    test_size=0.3,
    random_state=42,
    stratify=y_train_all
)

rf_clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42
)

model = make_pipeline(StandardScaler(), rf_clf)

model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)

print("\n=== Validation set (holdout from training data) ===")
print(classification_report(y_val, y_val_pred))
print("Confusion matrix (val):")
print(confusion_matrix(y_val, y_val_pred, labels=["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"]))
print("Labels order:", ["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"])


# ===============================
# 4. Evaluate on external multi-run test set (run_2)
# ===============================

if X_test_multi is not None:
    y_test_pred = model.predict(X_test_multi)
    print("\n=== External test: single_app_multi_runs run_2 only ===")
    print(classification_report(y_test_multi, y_test_pred))
    print("Confusion matrix (external multi-runs):")
    print(confusion_matrix(y_test_multi, y_test_pred, labels=["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"]))
    print("Labels order:", ["Idle", "Radar", "Pulse_Doppler", "SAR", "WiFi"])
else:
    print("\n[WARN] Skipping external multi-run test; none was built.")


# ===============================
# 5. Inference helper on new files
# ===============================

def predict_run(fpd_path, lpd_path):
    feats = extract_features_fpd_lpd(fpd_path, lpd_path).reshape(1, -1)
    pred = model.predict(feats)[0]
    proba = model.predict_proba(feats)[0]
    return pred, dict(zip(model.classes_, proba))


# Example usage (uncomment and adjust paths):
# fp = "single_app_runs/radar_correlator_run/run_1/radar_1_fpd_measured_temp.txt"
# lp = "single_app_runs/radar_correlator_run/run_1/radar_1_lpd_measured_temp.txt"
# p, pr = predict_run(fp, lp)
# print("Prediction:", p)
# print("Probabilities:", pr)

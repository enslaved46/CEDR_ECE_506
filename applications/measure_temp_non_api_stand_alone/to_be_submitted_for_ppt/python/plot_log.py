#!/usr/bin/env python3
"""
Batch plot temperature logs:
 - Reads all *_measured_temp.txt under ROOT_DIR
 - For each file, generates:
      * moving-average vs raw
      * Butterworth LPF vs raw
      * median filter vs raw
 - Saves PNGs under results/figures/<same/folder/structure>/
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
OUT_ROOT = ROOT_DIR / "results" / "figures"
FILE_SUFFIX = "_measured_temp.txt"


def read_temp_file(path: Path):
    """
    Read a temp log file with format:
        <temp> C <timestamp_ns> ns

    Returns:
        t_s  : np.ndarray of time in seconds (relative to first sample)
        temp : np.ndarray of temperature in °C
    """
    temps = []
    t_ns = []

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Expect: value, "C", timestamp, "ns"
            if len(parts) < 4:
                continue
            try:
                temp_val = float(parts[0])
                t_val = float(parts[2])
            except ValueError:
                continue
            temps.append(temp_val)
            t_ns.append(t_val)

    temps = np.array(temps, dtype=float)
    t_ns = np.array(t_ns, dtype=float)

    if len(temps) == 0:
        raise ValueError(f"No valid samples parsed from {path}")

    # Convert ns to seconds, relative to first sample
    t_s = (t_ns - t_ns[0]) * 1e-9
    return t_s, temps


def moving_average(x, window=5):
    """
    Moving average without zero-padding artifacts:
    - Use 'valid' convolution
    - Pad ends with edge values instead of zeros
    """
    n = len(x)
    if window <= 1 or window > n:
        return x.copy()

    kernel = np.ones(window) / window
    y_valid = np.convolve(x, kernel, mode="valid")

    pad_left = np.full(window // 2, y_valid[0])
    pad_right = np.full(window - 1 - window // 2, y_valid[-1])

    return np.concatenate((pad_left, y_valid, pad_right))


def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)


def process_file(path: Path):
    print(f"Processing: {path}")
    try:
        t_s, temp = read_temp_file(path)
    except Exception as e:
        print(f"  [WARN] Skipping {path} – read error: {e}")
        return

    # Build relative output directory:
    #   results/figures/<relative_folder_from_root>/
    rel_folder = path.parent.relative_to(ROOT_DIR)
    save_dir = OUT_ROOT / rel_folder
    ensure_dir(save_dir)

    base_name = path.stem
    out_base = save_dir / base_name

    # ----------------- 1) Moving Average -----------------
    try:
        y_mov = moving_average(temp, window=5)

        plt.figure()
        plt.plot(t_s, temp, label="Raw")
        plt.plot(t_s, y_mov, label="Moving Average", linewidth=2)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Moving Average: {base_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_base.with_name(base_name + "_movavg.png"))
        plt.close()
    except Exception as e:
        print(f"  [WARN] Moving average failed for {path}: {e}")

    # ----------------- 2) Butterworth LPF -----------------
    try:
        # 3rd order, cutoff = 0.1 (normalized frequency)
        b, a = butter(3, 0.1)
        y_lpf = filtfilt(b, a, temp)

        plt.figure()
        plt.plot(temp, label="Raw")
        plt.plot(y_lpf, label="Butterworth LPF", linewidth=2)
        plt.xlabel("Samples")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Butterworth LPF: {base_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_base.with_name(base_name + "_lpf.png"))
        plt.close()
    except Exception as e:
        print(f"  [WARN] LPF failed for {path}: {e}")

    # ----------------- 3) Median Filter -------------------
    try:
        y_med = medfilt(temp, kernel_size=5)

        plt.figure()
        plt.plot(temp, label="Raw")
        plt.plot(y_med, label="Median Filter", linewidth=2)
        plt.xlabel("Samples")
        plt.ylabel("Temperature (°C)")
        plt.title(f"Median Filter: {base_name}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_base.with_name(base_name + "_median.png"))
        plt.close()
    except Exception as e:
        print(f"  [WARN] Median filter failed for {path}: {e}")

    print(f"  ✔ Saved plots under: {save_dir}\n")


def main():
    print(f"Root:        {ROOT_DIR}")
    print(f"Output root: {OUT_ROOT}")
    ensure_dir(OUT_ROOT)

    temp_files = []
    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):
        # Skip results/figures to avoid re-processing outputs
        if "results" in dirpath and "figures" in dirpath:
            continue
        for fname in filenames:
            if fname.endswith(FILE_SUFFIX):
                temp_files.append(Path(dirpath) / fname)

    print(f"Found {len(temp_files)} measurement files.\n")

    for f in sorted(temp_files):
        process_file(f)

    print("=== All done. Check results/figures/ ===")


if __name__ == "__main__":
    main()

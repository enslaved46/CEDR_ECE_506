#!/usr/bin/env python3
"""
Batch plot temperature logs + summary averages

- Reads all *_measured_temp.txt under ROOT_DIR
- For each file, generates:
    * moving-average vs raw
    * Butterworth LPF vs raw
    * median filter vs raw
  and saves under:
    results/figures/<same/folder/structure>/

- Additionally:
    * For single_app_runs/*/run_X:
         computes mean temperature per run and plots:
           results/figures/summary/mean_temp_single_app_fpd.png
           results/figures/summary/mean_temp_single_app_lpd.png
    * For idle directories:
         idle_after_multi_apps_run_1
         idle_after_multi_apps_run_2
         ilde_nov_29_night
      computes one mean per file and plots:
           results/figures/summary/mean_temp_idle_fpd.png
           results/figures/summary/mean_temp_idle_lpd.png
"""

import os
import re
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

IDLE_DIR_NAMES = {
    "idle_after_multi_apps_run_1",
    "idle_after_multi_apps_run_2",
    "ilde_nov_29_night",
}


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

    t_s = (t_ns - t_ns[0]) * 1e-9  # ns → s, relative
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


def parse_run_number(run_name: str) -> int:
    """
    Extract integer from a run directory name like 'run_1'.
    If not found, return 0 so sorting is still deterministic.
    """
    m = re.search(r"(\d+)", run_name)
    if m:
        return int(m.group(1))
    return 0


def process_file(
    path: Path,
    stats_single_app,
    stats_idle,
):
    """
    Process a single *_measured_temp.txt file:
    - generate filtered plots
    - accumulate mean temperature stats for summaries
    """
    print(f"Processing: {path}")
    try:
        t_s, temp = read_temp_file(path)
    except Exception as e:
        print(f"  [WARN] Skipping {path} – read error: {e}")
        return

    # ---------- Determine zone (FPD / LPD) ----------
    fname_lower = path.name.lower()
    if "_fpd_" in fname_lower:
        zone = "fpd"
    elif "_lpd_" in fname_lower:
        zone = "lpd"
    else:
        zone = "unknown"

    # ---------- Save per-file filtered plots ----------
    rel_folder = path.parent.relative_to(ROOT_DIR)
    save_dir = OUT_ROOT / rel_folder
    ensure_dir(save_dir)

    base_name = path.stem
    out_base = save_dir / base_name

    # 1) Moving average
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

    # 2) Butterworth LPF
    try:
        b, a = butter(3, 0.1)  # 3rd order, cutoff = 0.1 (normalized)
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

    # 3) Median filter
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

    print(f"  ✔ Saved plots under: {save_dir}")

    # ---------- Accumulate mean temperature for summary ----------
    mean_temp = float(temp.mean())

    rel_parts = rel_folder.parts
    if len(rel_parts) >= 3 and rel_parts[0] == "single_app_runs":
        # Example: single_app_runs / radar_correlator_run / run_1 / <file>
        app_name = rel_parts[1]
        run_name = rel_parts[2]

        if zone not in stats_single_app:
            stats_single_app[zone] = {}
        if app_name not in stats_single_app[zone]:
            stats_single_app[zone][app_name] = {}
        stats_single_app[zone][app_name][run_name] = mean_temp

    elif len(rel_parts) >= 1 and rel_parts[0] in IDLE_DIR_NAMES:
        # Idle directories at root level
        idle_label = rel_parts[0]  # e.g., idle_after_multi_apps_run_1

        if zone not in stats_idle:
            stats_idle[zone] = {}
        stats_idle[zone][idle_label] = mean_temp

    print(f"  ⓘ Mean temp ({zone}): {mean_temp:.3f} °C\n")


def plot_single_app_summary(stats_single_app):
    """
    Create summary plots for single_app_runs:
    x-axis: run index (1..5)
    y-axis: mean temperature
    one line per app
    """
    summary_dir = OUT_ROOT / "summary"
    ensure_dir(summary_dir)

    for zone in ("fpd", "lpd"):
        if zone not in stats_single_app:
            continue

        plt.figure()
        for app_name, runs_dict in sorted(stats_single_app[zone].items()):
            # Sort runs by run number
            run_items = sorted(
                runs_dict.items(), key=lambda kv: parse_run_number(kv[0])
            )
            x = [parse_run_number(rn) for rn, _ in run_items]
            y = [val for _, val in run_items]
            plt.plot(x, y, marker="o", label=app_name)

        plt.xlabel("Run number")
        plt.ylabel("Mean temperature (°C)")
        plt.title(f"Mean Temperature per Run (single_app_runs, {zone.upper()})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        out_path = summary_dir / f"mean_temp_single_app_{zone}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"✔ Saved single-app summary: {out_path}")


def plot_idle_summary(stats_idle):
    """
    Create summary plots for idle directories:
    x-axis: run label index
    y-axis: mean temperature
    """
    summary_dir = OUT_ROOT / "summary"
    ensure_dir(summary_dir)

    for zone in ("fpd", "lpd"):
        if zone not in stats_idle:
            continue

        labels = sorted(stats_idle[zone].keys())
        y = [stats_idle[zone][lab] for lab in labels]
        x = np.arange(1, len(labels) + 1)

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.xlabel("Idle run")
        plt.ylabel("Mean temperature (°C)")
        plt.title(f"Mean Temperature per Idle Run ({zone.upper()})")
        plt.grid(True)
        plt.tight_layout()

        out_path = summary_dir / f"mean_temp_idle_{zone}.png"
        plt.savefig(out_path)
        plt.close()
        print(f"✔ Saved idle summary: {out_path}")


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

    # Stats containers
    stats_single_app = {}  # {zone: {app_name: {run_name: mean_temp}}}
    stats_idle = {}        # {zone: {idle_label: mean_temp}}

    for f in sorted(temp_files):
        process_file(f, stats_single_app, stats_idle)

    # Summary plots
    plot_single_app_summary(stats_single_app)
    plot_idle_summary(stats_idle)

    print("=== All done. Check results/figures/ and results/figures/summary/ ===")


if __name__ == "__main__":
    main()

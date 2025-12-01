#!/usr/bin/env python3
"""
idle_mean_scatter.py

Compute mean temperature for idle / ideal runs and plot as
connected scatter (line + markers), with night runs first
and day runs later.

Idle labels matched (anywhere in the path):
    - idle_after_multi_apps_run_1_night
    - idle_after_multi_apps_run_2_night
    - idle_nov_28_night_run
    - ilde_nov_29_night
    - idle_day_run_nov_29  (inside nov_29_day_runs_raw_data)

For each *_fpd_measured_temp.txt / *_lpd_measured_temp.txt:
    1 file = 1 sample (mean temp)

Outputs:
    results/figures/summary/idle_means_scatter_fpd.png
    results/figures/summary/idle_means_scatter_lpd.png
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Root directory = where this script lives
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results" / "figures" / "summary"

# Folder names that mean "this is an idle run"
IDLE_LABELS = {
    "idle_after_multi_apps_run_1_night",
    "idle_after_multi_apps_run_2_night",
    "idle_nov_28_night_run",
    "ilde_nov_29_night",
    "idle_day_run_nov_29",  # inside nov_29_day_runs_raw_data
}

FILE_SUFFIX = "_measured_temp.txt"


def read_temps(path: Path) -> np.ndarray:
    """Read temperature column from '<temp> C <timestamp> ns' file."""
    temps = []
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                temps.append(float(parts[0]))
            except ValueError:
                continue
    arr = np.array(temps, float)
    if arr.size == 0:
        raise ValueError(f"No valid temps in {path}")
    return arr


def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def is_day_label(label: str) -> bool:
    """Return True if label should be considered 'day' run."""
    return "day" in label.lower()


def main():
    print(f"[INFO] ROOT: {ROOT}")
    ensure_dir(OUT_DIR)

    # idle_means[zone][idle_label] = mean_temp
    idle_means = {"fpd": {}, "lpd": {}}

    # Walk tree
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # Skip our own figures directory
        if "results" in dirpath and "figures" in dirpath:
            continue

        for fname in filenames:
            if not fname.endswith(FILE_SUFFIX):
                continue

            full_path = Path(dirpath) / fname
            rel_path = full_path.relative_to(ROOT)
            parts = rel_path.parts

            # Which idle label does this belong to?
            idle_label = None
            for part in parts:
                if part in IDLE_LABELS:
                    idle_label = part
                    break

            if idle_label is None:
                continue  # not an idle run we care about

            name_lower = fname.lower()
            if "fpd" in name_lower:
                zone = "fpd"
            elif "lpd" in name_lower:
                zone = "lpd"
            else:
                zone = "unknown"

            if zone not in ("fpd", "lpd"):
                continue

            try:
                temps = read_temps(full_path)
            except Exception as e:
                print(f"[WARN] Skipping {rel_path}: {e}")
                continue

            mean_val = float(temps.mean())
            idle_means[zone][idle_label] = mean_val
            print(f"[INFO] {rel_path} → {zone} mean = {mean_val:.3f} °C")

    # === Make connected-scatter plots ===
    for zone in ("fpd", "lpd"):
        zone_data = idle_means[zone]
        if not zone_data:
            print(f"[INFO] No {zone.upper()} idle data found, skipping.")
            continue

        # Order: all night labels first, then day labels
        labels_sorted = sorted(
            zone_data.keys(),
            key=lambda lbl: (is_day_label(lbl), lbl)
        )
        values = [zone_data[lbl] for lbl in labels_sorted]
        x = np.arange(1, len(labels_sorted) + 1)

        plt.figure(figsize=(8, 5))
        plt.plot(x, values, "-o")
        plt.xticks(x, labels_sorted, rotation=20, ha="right")
        plt.ylabel("Mean Temperature (°C)")
        plt.xlabel("Idle runs (night first, day later)")
        plt.title(f"Idle / Ideal Mean Temperature ({zone.upper()})")
        plt.grid(True)
        plt.tight_layout()

        out_file = OUT_DIR / f"idle_means_scatter_{zone}.png"
        plt.savefig(out_file)
        plt.close()
        print(f"[OK] Saved scatter figure → {out_file}")

    print("\n[DONE] Check figures in:")
    print(f"   {OUT_DIR}\n")


if __name__ == "__main__":
    main()

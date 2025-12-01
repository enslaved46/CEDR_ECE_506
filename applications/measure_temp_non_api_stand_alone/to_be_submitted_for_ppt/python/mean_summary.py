#!/usr/bin/env python3
"""
mean_summary.py

Computes ONE mean temperature PER RUN FILE and plots only the MEANS.

- single_app_runs (under nov_29_night_runs/single_app_runs/...):
    5 runs per app → 5 dots per app
    OUTPUT:
        results/figures/summary/mean_temp_single_app_fpd.png
        results/figures/summary/mean_temp_single_app_lpd.png

- idle runs:
    3 dots total (one per folder)
    OUTPUT:
        results/figures/summary/mean_temp_idle_fpd.png
        results/figures/summary/mean_temp_idle_lpd.png
"""

import os
import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ========================= CONFIG =========================
ROOT = Path(__file__).resolve().parent
OUT_SUMMARY = ROOT / "results" / "figures" / "summary"
FILE_SUFFIX = "_measured_temp.txt"

IDLE_FOLDERS = {
    "idle_after_multi_apps_run_1",
    "idle_after_multi_apps_run_2",
    "ilde_nov_29_night",
}
# ==========================================================


def read_temp(path: Path):
    """Load temperatures only."""
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
        raise ValueError(f"No temperature values could be parsed in {path}")
    return arr


def run_num(run_dir: str) -> int:
    m = re.search(r"(\d+)", run_dir)
    return int(m.group(1)) if m else 0


def ensure_dir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)


def plot_single_app(stats):
    """Plot mean temperatures for each app vs run number."""
    for zone in ("fpd", "lpd"):
        if zone not in stats or not stats[zone]:
            print(f"[INFO] No single_app data for {zone.upper()}")
            continue

        plt.figure(figsize=(9,6))
        for app, runs in sorted(stats[zone].items()):
            ordered = sorted(runs.items(), key=lambda kv: run_num(kv[0]))
            x = [run_num(r) for r,_ in ordered]
            y = [v for _,v in ordered]
            plt.plot(x,y,"-o",label=app)

        ensure_dir(OUT_SUMMARY)
        fp = OUT_SUMMARY / f"mean_temp_single_app_{zone}.png"
        plt.title(f"Single-App Mean Temp ({zone.upper()})")
        plt.xlabel("Run Number (each run = 1 sample)")
        plt.ylabel("Mean Temperature (°C)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fp)
        plt.close()

        print(f"[OK] Saved → {fp}")


def plot_idle(stats):
    """Plot 3 mean points — one for each idle log folder."""
    for zone in ("fpd", "lpd"):
        if zone not in stats or not stats[zone]:
            print(f"[INFO] No IDLE data for {zone.upper()}")
            continue

        labels = sorted(stats[zone].keys())
        y = [stats[zone][name] for name in labels]
        x = list(range(1,len(y)+1))

        plt.figure(figsize=(8,5))
        plt.plot(x,y,"-o")
        plt.xticks(x, labels, rotation=15)
        plt.ylabel("Mean Temperature (°C)")
        plt.title(f"Idle Mean Temp ({zone.upper()})")
        plt.grid(True)
        plt.tight_layout()

        ensure_dir(OUT_SUMMARY)
        fp = OUT_SUMMARY / f"mean_temp_idle_{zone}.png"
        plt.savefig(fp)
        plt.close()

        print(f"[OK] Saved → {fp}")


def main():

    print(f"[INFO] Scanning for temperature logs in:\n      {ROOT}")

    single_app = {}   # {zone:{app:{run:mean}}}
    idle = {}         # {zone:{folder:mean}}

    found = []
    for root,_,files in os.walk(ROOT):
        if "results" in root and "figures" in root:    # skip outputs
            continue
        for f in files:
            if f.endswith(FILE_SUFFIX):
                found.append(Path(root)/f)

    print(f"[INFO] Found {len(found)} measurement files\n")

    for path in sorted(found):
        rel = path.relative_to(ROOT)
        parts = rel.parts
        print(f"  → {rel}")

        # get zone
        name = path.name.lower()
        zone = "fpd" if "_fpd_" in name else "lpd" if "_lpd_" in name else "unknown"

        try:
            mean = read_temp(path).mean()
        except Exception as e:
            print(f"     [WARN] failed → {e}")
            continue

        print(f"     mean = {mean:.3f} °C ({zone})")

        # detect single_app anywhere
        if "single_app_runs" in parts:
            idx = parts.index("single_app_runs")
            app = parts[idx+1]
            run = parts[idx+2]

            single_app.setdefault(zone,{})
            single_app[zone].setdefault(app,{})
            single_app[zone][app][run] = mean
            continue

        # detect idle anywhere
        for part in parts:
            if part in IDLE_FOLDERS:
                idle.setdefault(zone,{})
                idle[zone][part] = mean
                break

    print("\n[SUMMARY]")
    print("  single_app zones:", list(single_app.keys()))
    print("  idle zones:", list(idle.keys()))

    plot_single_app(single_app)
    plot_idle(idle)

    print(f"\n[DONE] View results at:\n  {OUT_SUMMARY}\n")


if __name__ == "__main__":
    main()

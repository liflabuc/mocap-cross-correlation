# Motion Capture (MOCAP) Processing Script
# Developed by: Marisol Correa (2025)
# Supervised by: Esteban Hurtado

# Description:
# Computes cross-correlations between paired MoCap time series
# across condition folders, applies Fisher-z weighted averaging,
# confidence intervals (95%), and FDR correction.

# Usage (PowerShell / CMD):
# py -3.10 "path_to_script\Mocap_processing_W3.py" "path_to_data" -fs 180
# py -3.10 "path_to_script\Mocap_processing_W3.py" "path_to_data" -fs 120 -min-lag -2 -max-lag 2

# Arguments:
#   folder          Parent folder with condition subfolders
#   -fs <float>     Sampling frequency (Hz)
#   -fc <float>     Low-pass cutoff (Hz, default=10)
#   -interpolate    Apply cubic smoothing for visualization

# Output:
#   - <Condition>_Mocap_W3.png (per condition)
#   - All_Conditions_W3.png (combined summary)

import os
import math
import colorsys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from scipy.stats import norm, pearsonr
from statsmodels.stats.multitest import multipletests


def load_files(condition):
    """Read all TSV-like CSV files in a condition subfolder (no header, 2 columns)."""
    condition_folder = os.path.join(folder_path, condition)
    files = [file for file in os.listdir(condition_folder) if file.endswith('.csv')]
    condition_data = {}
    for file in files:
        file_path = os.path.join(condition_folder, file)
        series_name = file.split('.')[0]
        condition_data[series_name] = pd.read_csv(file_path, delimiter='\t', header=None)
    return condition_data


def generate_pastel_colors(n):
    """Generate readable pastel colors for per-condition plots."""
    return [
        colorsys.hls_to_rgb(hue, 0.7, 0.6)
        for hue in np.linspace(0, 1, n, endpoint=False)
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="Motion cross-correlation",
        description=(
            "Reads two time series from each CSV file (2 columns, as many rows as samples) "
            "in each of several subfolders. Each subfolder is one experimental condition. "
            "Outputs a condition-level cross-correlation curve with CI and FDR markers."
        ),
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Parent folder that contains condition subfolders."
    )
    parser.add_argument(
        "-fs",
        type=float,
        help="Sampling frequency (Hz)."
    )
    parser.add_argument(
        "-fc",
        type=float,
        default=10,
        help="Low-pass cutoff (Hz)."
    )
    parser.add_argument(
        "-min-lag",
        type=float,
        default=-1.5,
        help="Lag lower bound (s, inclusive)."
    )
    parser.add_argument(
        "-max-lag",
        type=float,
        default=1.6,
        help="Lag upper bound (s, exclusive)."
    )
    parser.add_argument(
        "-lag-step",
        type=float,
        default=0.1,
        help="Lag step (s)."
    )
    parser.add_argument(
        "-interpolate",
        action="store_true",
        help="Cubic spline smoothing."
    )
    args = parser.parse_args()

    folder_path = args.folder
    fs = args.fs
    fc = args.fc
    start_shift = args.min_lag
    end_shift = args.max_lag
    shift_step = args.lag_step

    # Precompute a single lag grid and reuse it everywhere.
    shifts = np.arange(start_shift, end_shift, shift_step)

    # Find condition subfolders and color palette.
    subfolders = next(os.walk(folder_path))[1]
    colors = generate_pastel_colors(len(subfolders))
    color_index = 0

    # Figure that aggregates all conditions together.
    global_fig, global_ax = plt.subplots()

    for subfolder in subfolders:
        # Per-file correlation arrays (aligned to `shifts`) and per-file effective sample sizes.
        total_correlations = []
        total_neff = []

        cond = load_files(subfolder)
        files_cond = list(cond.keys())

        # 2nd-order low-pass Butterworth and effective bandwidth (Hz) for EDF.
        b, a = butter(N=2, Wn=(2 * fc) / fs, btype="low", analog=False)
        W = 3  # Effective bandwidth used to build N*.

        for file in files_cond:
            # Velocity-like derivation (first difference), then low-pass filter.
            subject_A = cond[file][0].diff().dropna()
            subject_B = cond[file][1].diff().dropna()

            subject_A = filtfilt(b, a, subject_A)
            subject_B = filtfilt(b, a, subject_B)

            correlations = []
            neff_this_file = []

            # Cross-correlation across fixed lag grid.
            for shift in shifts:
                shift_indices = abs(int(shift * fs))

                # Align segments per lag sign.
                if shift < 0:
                    deviation_A = subject_A[shift_indices:]
                    deviation_B = subject_B[:len(deviation_A)]
                elif shift > 0:
                    deviation_B = subject_B[shift_indices:]
                    deviation_A = subject_A[:len(deviation_B)]
                else:
                    deviation_B = subject_B
                    deviation_A = subject_A

                # De-mean before Pearson correlation.
                deviation_A = deviation_A - np.mean(deviation_A)
                deviation_B = deviation_B - np.mean(deviation_B)

                r, _ = pearsonr(deviation_A, deviation_B)
                correlations.append(r)

                # Effective sample size per lag for this file: N* = 2 * W * T_lag
                T_lag = len(deviation_A) / fs
                neff_this_file.append(2 * W * T_lag)

            total_correlations.append(correlations)
            total_neff.append(neff_this_file)

        # Combine files per lag in Fisher-z space using N* as weights.
        R_by_lag = list(zip(*total_correlations))
        Neff_by_lag = list(zip(*total_neff))

        avg_z = []
        sum_neff = []
        for r_list, ne_list in zip(R_by_lag, Neff_by_lag):
            z_list = [np.arctanh(r) for r in r_list]
            w_list = list(ne_list)  # weights = N*
            z_bar = np.average(z_list, weights=w_list)
            avg_z.append(z_bar)
            sum_neff.append(np.sum(w_list))

        # Standard error on z using EDF: SE_z = 1 / sqrt(N* - 3).
        standard_error = [1 / math.sqrt(max(ne - 3, 1)) for ne in sum_neff]

        # z-stats and two-sided p-values.
        z_stats = [
            z / se if (se > 0 and np.isfinite(z)) else np.nan
            for z, se in zip(avg_z, standard_error)
        ]
        p_values = [
            2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            for z in z_stats
        ]

        # FDR (Benjamini–Hochberg) across lags.
        p_values = np.array(p_values)
        fdr_mask = np.full_like(p_values, False, dtype=bool)
        if np.any(np.isfinite(p_values)):
            reject, _, _, _ = multipletests(
                p_values[np.isfinite(p_values)],
                alpha=0.05,
                method="fdr_bh"
            )
            fdr_mask[np.isfinite(p_values)] = reject

        # 95% CI in z, then back-transform to r.
        z_crit = norm.ppf(0.975)
        upper_limits = [z + z_crit * se for z, se in zip(avg_z, standard_error)]
        lower_limits = [z - z_crit * se for z, se in zip(avg_z, standard_error)]

        averages = [np.tanh(z) for z in avg_z]
        upper_limits = [np.tanh(u) for u in upper_limits]
        lower_limits = [np.tanh(l) for l in lower_limits]

        color = colors[color_index]
        fig, ax = plt.subplots()

        if args.interpolate:
            # Smooth with cubic interpolation for visual clarity (no change in stats).
            f_avg = interp1d(shifts, averages, kind="cubic")
            f_lower = interp1d(shifts, lower_limits, kind="cubic")
            f_upper = interp1d(shifts, upper_limits, kind="cubic")

            shifts_new = np.linspace(min(shifts), max(shifts), 100)

            averages_smooth = f_avg(shifts_new)
            lower_limits_smooth = f_lower(shifts_new)
            upper_limits_smooth = f_upper(shifts_new)

            ax.plot(shifts_new, averages_smooth, "-", color=color)
            ax.fill_between(
                shifts_new,
                lower_limits_smooth,
                upper_limits_smooth,
                color=color,
                alpha=0.4
            )

            global_ax.plot(shifts_new, averages_smooth, "-", color=color, label=subfolder)
            global_ax.fill_between(
                shifts_new,
                lower_limits_smooth,
                upper_limits_smooth,
                color=color,
                alpha=0.4
            )
        else:
            # Raw (unsmoothed) curve + CI.
            ax.plot(shifts, averages, "-", color=color)
            ax.fill_between(shifts, lower_limits, upper_limits, color=color, alpha=0.4)

            global_ax.plot(shifts, averages, "-", color=color, label=subfolder)
            global_ax.fill_between(shifts, lower_limits, upper_limits, color=color, alpha=0.4)

        # Mark lags that pass FDR.
        sig_shifts = np.array(shifts)[fdr_mask]
        sig_values = np.array(averages)[fdr_mask]
        if sig_shifts.size > 0:
            ax.scatter(
                sig_shifts,
                sig_values,
                s=30,
                facecolors="none",
                edgecolors="k",
                linewidths=1.2,
                label="FDR < 0.05"
            )

        color_index += 1

        ax.set_xlabel("Lag [s]")
        ax.set_ylabel("Pearson correlation")
        fig.suptitle("Cross-correlation")
        ax.set_title(subfolder)
        ax.grid(True)
        ax.legend(fontsize=7)

        name_fig = os.path.join(folder_path, subfolder + "_Mocap_W3.png")
        fig.savefig(name_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Combined figure across all conditions.
    global_ax.set_xlabel("Lag [s]")
    global_ax.set_ylabel("Pearson correlation")
    global_ax.set_title("Cross-correlation of all conditions")
    global_ax.grid(True)
    global_ax.legend(fontsize=7)

    global_fig_name = os.path.join(folder_path, "All_Conditions_W3.png")
    global_fig.savefig(global_fig_name, dpi=150, bbox_inches="tight")
    plt.close(global_fig)
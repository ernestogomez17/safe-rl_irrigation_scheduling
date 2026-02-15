#!/usr/bin/env python3
"""
Plotting utilities for Safe RL Irrigation Scheduling.

This module serves two purposes:

1. **Training-time helpers** — lightweight function imported by ``train.py``
   to save simulation CSVs and diagnostic PNGs during training.
   This lives in Section 1 and has *no side-effects* at import time.

2. **Post-hoc analysis** — publication-quality TensorBoard aggregation,
   performance grids, simulation overlays, and statistics tables.
   These live in Section 2 and are only executed when this file is run
   directly (``python results/plots.py``) or called from a notebook.

Usage from a notebook / CLI::

    from results.plots import run_analysis
    run_analysis()                                        # full pipeline
    run_analysis(episode=499, chance_filters=["0.95"])     # + filtered grid
"""

# ---------------------------------------------------------------------------
# Standard library / lightweight imports (always available)
# ---------------------------------------------------------------------------
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


# ###########################################################################
#                                                                           #
#  SECTION 1 — TRAINING-TIME PLOTTING UTILITIES                            #
#  Imported by  models/safe_rl/train.py  (lightweight, no side-effects)    #
#                                                                           #
# ###########################################################################


def plot_simulation_data(
    history_It, history_Rain, history_st,
    history_ET_o, history_ETmax, history_Kc, history_rho,
    s_star, sfc, sw,
    model_directory, episode_count, model_name,
):
    """
    Save simulation data (CSV) and a diagnostic plot (PNG).

    Called at the end of each evaluation epoch during training.  Uses the
    same renderer (``_populate_simulation_axes``) as the publication
    analysis pipeline so that training-time diagnostics and post-hoc
    figures are visually consistent.
    """
    # --- Prepare arrays ---
    original_It = np.asarray(history_It)
    Rain = np.asarray(history_Rain)
    st = np.asarray(history_st)
    ET_o = np.asarray(history_ET_o)
    ETmax = np.asarray(history_ETmax)
    Kc = np.asarray(history_Kc)
    rho = np.asarray(history_rho)

    min_len = min(len(original_It), len(Rain), len(st),
                  len(ET_o), len(ETmax), len(Kc), len(rho))
    if min_len == 0:
        print("No data available to plot.")
        return

    original_It = original_It[:min_len]
    Rain, st = Rain[:min_len], st[:min_len]
    ET_o, ETmax = ET_o[:min_len], ETmax[:min_len]
    Kc, rho = Kc[:min_len], rho[:min_len]

    # --- Build DataFrame (shared format with analysis pipeline) ---
    df = pd.DataFrame({
        "History It (before scaling)": original_It,
        "History It (after scaling)": original_It * 1000,
        "History Rainfall": Rain,
        "History Soil Moisture": st,
        "History ET_o": ET_o,
        "History ETmax": ETmax,
        "History Kc": Kc,
        "History rho": rho,
    })

    # --- Save CSV ---
    csv_path = os.path.join(model_directory, f"simulation_data_episode{episode_count}.csv")
    df.to_csv(csv_path, index=False)

    # --- Diagnostic plot (same renderer as publication figures) ---
    constants = {"s_star": s_star, "sfc": sfc, "sw": sw}
    fig, ax1 = plt.subplots(figsize=(14, 6))
    moisture_line, rain_bars, irr_bars, ax2_lines = _populate_simulation_axes(
        ax1, df, constants,
    )
    plt.title(f"{model_name}: Evaluation (Episode {episode_count})", fontsize=14)
    _build_simulation_legend(fig, moisture_line, rain_bars, irr_bars, ax2_lines)
    fig.tight_layout(rect=[0, 0.15, 1, 0.95])

    png_path = os.path.join(model_directory, f"simulation_episode{episode_count}_plot.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Simulation data saved to {csv_path}")


# ###########################################################################
#                                                                           #
#  SECTION 2 — POST-HOC ANALYSIS (publication-quality figures & stats)     #
#  Only executed when calling  run_analysis()  or  __main__                #
#                                                                           #
# ###########################################################################


# ---- Shared colour palette for simulation plots ----------------------------

_SIM_COLORS = {
    "moisture":      "#0D7607",
    "rain":          "#1916ED",
    "irrigation":    "#4B3633",
    "feasible_zone": "#B5E6B8",
    "chance_zone":   "#FFB46E",
    "hard_zone":     "#FD9286",
    "sfc_line":      "#BC850E",
    "s_star_line":   "#7746D9",
    "sw_line":       "#9F4500",
}

# ---- Soil-moisture thresholds (shared across all analysis functions) -------

SIMULATION_CONSTANTS = {
    "s_star": 0.35,
    "sfc":    0.65,
    "sw":     0.30,
}


def _default_config() -> dict:
    """Return the default analysis configuration dictionary."""
    return {
        "base_directory": "/scratch/egomez/irrigation_project_output/models",
        "smoothing_window": 21,
        "smoothing_polyorder": 5,
        "output_dir": "./plots",

        "model_name_pattern": re.compile(
            r"exp\d+_(?P<algo>DDPG|SAC|DDPGLagrangian|SACLagrangian)_chance_"
            r"(?P<chance>[\d\.]+)_"
            r"(?P<days>days\d+)"
            r"(?:_s(?P<seed>\d+))?"
        ),

        "tag_mapping": {
            "Averageworker/EpRet":              "Return (Training)",
            "Averageworker/EpNumViolations":    "Number of Violations (Training)",
            "Averageeval/TestEpRet":            "Return (Evaluation)",
            "Averageeval/TestEpNumViolations":  "Number of Violations (Evaluation)",
            "Stdworker/EpRet":                  "Return Std (Training)",
            "Stdworker/EpNumViolations":        "Number of Violations Std (Training)",
            "Stdeval/TestEpRet":                "Return Std (Evaluation)",
            "Stdeval/TestEpNumViolations":      "Number of Violations Std (Evaluation)",
        },

        "plot_styles": {
            "SAC":  {"color": "#1f77b4", "linestyle": "-", "label": "SAC"},
            "DDPG": {"color": "#ff7f0e", "linestyle": "-", "label": "DDPG"},
            "SACLagrangian": {
                "1.0":  {"color": "#2ca02c", "linestyle": "-", "label": r"SAC Lagrangian ($\alpha=1.0$)"},
                "0.95": {"color": "#d62728", "linestyle": "-", "label": r"SAC Lagrangian ($\alpha=0.95$)"},
                "0.85": {"color": "#9467bd", "linestyle": "-", "label": r"SAC Lagrangian ($\alpha=0.85$)"},
                "0.75": {"color": "#8c564b", "linestyle": "-", "label": r"SAC Lagrangian ($\alpha=0.75$)"},
            },
            "DDPGLagrangian": {
                "1.0":  {"color": "#e377c2", "linestyle": "-", "label": r"DDPG Lagrangian ($\alpha=1.0$)"},
                "0.95": {"color": "#7f7f7f", "linestyle": "-", "label": r"DDPG Lagrangian ($\alpha=0.95$)"},
                "0.85": {"color": "#bcbd22", "linestyle": "-", "label": r"DDPG Lagrangian ($\alpha=0.85$)"},
                "0.75": {"color": "#17becf", "linestyle": "-", "label": r"DDPG Lagrangian ($\alpha=0.75$)"},
            },
        },
    }


# ---------------------------------------------------------------------------
#  Lazy-loaded heavy imports (TensorBoard, scipy, tqdm, IPython)
# ---------------------------------------------------------------------------

def _lazy_imports():
    """Import heavy analysis dependencies on first use, not at module load."""
    from scipy.signal import savgol_filter
    from tensorboard.backend.event_processing import event_accumulator
    from tqdm.auto import tqdm
    try:
        from IPython.display import display
    except ImportError:
        display = print
    return savgol_filter, event_accumulator, tqdm, display


# ---------------------------------------------------------------------------
#  Publication rcParams
# ---------------------------------------------------------------------------

def _apply_publication_style():
    """Apply LaTeX-based matplotlib rcParams for publication-quality figures."""
    mpl.rcParams.update({
        "text.usetex":      True,
        "font.family":      "serif",
        "font.serif":       ["Computer Modern Roman"],
        "font.size":        12,
        "axes.labelsize":   14,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  11,
        "pgf.texsystem":    "pdflatex",
        "pgf.preamble":     r"\usepackage[utf8x]{inputenc}\usepackage[T1]{fontenc}",
    })


# ---------------------------------------------------------------------------
#  TensorBoard data loading
# ---------------------------------------------------------------------------

def parse_model_info(dir_name: str, config: dict) -> dict | None:
    """Parse an experiment directory name into algo / chance / days metadata."""
    match = config["model_name_pattern"].search(dir_name)
    if not match:
        return None
    info = match.groupdict()
    info["name"] = f"{info['algo']}_chance{info['chance']}_{info['days']}"
    return info


def _process_scalar_data(tb_path: str, config: dict) -> pd.DataFrame | None:
    """Load scalar data from a single TensorBoard log directory."""
    _, event_accumulator, _, _ = _lazy_imports()
    try:
        ea = event_accumulator.EventAccumulator(
            tb_path, size_guidance={event_accumulator.SCALARS: 0}
        )
        ea.Reload()
        if not ea.Tags().get("scalars"):
            return None

        ref_tag = "Averageeval/TestEpRet"
        if ref_tag not in ea.Tags()["scalars"]:
            return None

        data = {"TrainingSteps": [e.step for e in ea.Scalars(ref_tag)]}
        for tag, col_name in config["tag_mapping"].items():
            if tag in ea.Tags()["scalars"]:
                values = [e.value for e in ea.Scalars(tag)]
                if len(values) == len(data["TrainingSteps"]):
                    data[col_name] = values

        return pd.DataFrame(data)
    except Exception as e:
        print(f"  Error processing {tb_path}: {e}")
        return None


def load_all_runs(base_dir: str, config: dict | None = None) -> dict:
    """
    Scan *base_dir* for TensorBoard logs, group by experiment configuration.

    Returns ``{exp_name: {"dfs": [DataFrames], "info": {metadata}}}``.
    """
    if config is None:
        config = _default_config()
    _, _, tqdm, _ = _lazy_imports()

    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    grouped = defaultdict(lambda: {"dfs": [], "info": {}})
    tb_paths = [
        os.path.join(root, "tb")
        for root, dirs, _ in os.walk(base_dir) if "tb" in dirs
    ]

    print(f"Found {len(tb_paths)} TensorBoard directories. Processing...")
    for tb_path in tqdm(tb_paths, desc="Loading runs"):
        model_dir = os.path.basename(os.path.dirname(tb_path))
        parent_dir = os.path.basename(os.path.dirname(os.path.dirname(tb_path)))

        info = parse_model_info(parent_dir, config) or parse_model_info(model_dir, config)
        if info is None:
            continue

        df = _process_scalar_data(tb_path, config)
        if df is not None and not df.empty:
            grouped[info["name"]]["dfs"].append(df)
            grouped[info["name"]]["info"] = info

    return dict(grouped)


# ---------------------------------------------------------------------------
#  Performance-grid helpers (TensorBoard learning curves)
# ---------------------------------------------------------------------------

def _get_legend_order(algo: str, chance: str | None = None) -> int:
    """Return a sort key so the legend reads DDPG -> SAC -> Lagrangians by alpha."""
    if "Lagrangian" not in algo:
        return 0 if algo == "DDPG" else 1
    base = {"0.75": 2, "0.85": 4, "0.95": 6, "1.0": 8}.get(chance, 10)
    return base if "DDPG" in algo else base + 1


def _plot_single_metric(
    ax: plt.Axes,
    mean_metric: str,
    std_metric: str,
    day_data: dict,
    config: dict,
    *,
    scale_factor: float = 1.0,
    chance_filter: str | None = None,
):
    """
    Plot one metric on *ax* for every experiment in *day_data*.

    Parameters
    ----------
    chance_filter : str or None
        If given (e.g. ``"0.95"``), only regular models and Lagrangian
        models matching that chance value are plotted.
    """
    savgol_filter, _, _, _ = _lazy_imports()

    # Optional filtering
    filtered = {}
    for name, exp in day_data.items():
        algo = exp["info"]["algo"]
        chance = exp["info"].get("chance")
        if chance_filter is not None:
            if "Lagrangian" in algo and chance != chance_filter:
                continue
        filtered[name] = exp

    # Sort for deterministic legend order
    items = sorted(
        filtered.items(),
        key=lambda kv: _get_legend_order(kv[1]["info"]["algo"], kv[1]["info"].get("chance")),
    )

    for name, exp in items:
        info, dfs = exp["info"], exp["dfs"]
        if not dfs:
            continue

        all_df = pd.concat(dfs).sort_values("TrainingSteps").reset_index(drop=True)

        if std_metric in all_df.columns:
            agg = all_df.groupby("TrainingSteps")[[mean_metric, std_metric]].mean().reset_index()
            mean_vals = agg[mean_metric]
            std_vals = agg[std_metric]
        else:
            agg = all_df.groupby("TrainingSteps")[mean_metric].agg(["mean", "std"]).reset_index()
            agg.columns = ["TrainingSteps", "mean", "std"]
            mean_vals = agg["mean"]
            std_vals = agg["std"].fillna(0)

        smooth = savgol_filter(mean_vals, config["smoothing_window"], config["smoothing_polyorder"])
        std_vals = np.maximum(std_vals, 0)

        style_group = config["plot_styles"].get(info["algo"], {})
        style = style_group.get(info["chance"], {}) if "Lagrangian" in info["algo"] else style_group
        color = style.get("color", "gray")
        ls = style.get("linestyle", "-")
        label = style.get("label", name)

        ax.plot(agg["TrainingSteps"], smooth, label=label, color=color, linestyle=ls, linewidth=1.5)
        ax.fill_between(agg["TrainingSteps"], smooth - std_vals, smooth + std_vals, alpha=0.15, color=color)

    ax.set_ylabel(mean_metric)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, _: f"{x / scale_factor:.1f}"))


def plot_performance_grid(
    data: dict,
    config: dict,
    *,
    chance_filter: str | None = None,
    output_suffix: str = "",
):
    """
    Generate a multi-page PDF with training & evaluation performance grids.

    Parameters
    ----------
    chance_filter : str or None
        Pass ``"0.95"`` to only show regular + chance=0.95 Lagrangian models.
    output_suffix : str
        Appended to the filename, e.g. ``"_chance95"``.
    """
    _, _, _, display = _lazy_imports()

    data_by_day = defaultdict(dict)
    for name, exp in data.items():
        data_by_day[exp["info"]["days"]][name] = exp
    if not data_by_day:
        print("No data to plot.")
        return

    fname = f"performance_grid{output_suffix}.pdf"
    save_path = os.path.join(config["output_dir"], fname)

    with PdfPages(save_path) as pdf:
        print(f"\n--- Generating {save_path} ---")

        for metric_type in ("Evaluation", "Training"):
            print(f"  -> {metric_type} page...")
            day_keys = sorted(data_by_day)
            n_rows = len(day_keys)

            fig, axes = plt.subplots(n_rows, 2, figsize=(10, 4 * n_rows), squeeze=False)
            fig.suptitle(
                fr"\textbf{{Performance Metrics ({metric_type})}}",
                fontsize=16, y=0.96,
            )

            for i, day in enumerate(day_keys):
                day_data = data_by_day[day]
                day_num = day.replace("days", "")
                ax_ret, ax_viol = axes[i, 0], axes[i, 1]

                # Determine x-axis scale
                max_steps = max(
                    (df["TrainingSteps"].max()
                     for exp in day_data.values() for df in exp["dfs"] if not df.empty),
                    default=0,
                )
                if max_steps > 1.5e6:
                    sf, sl = 1e6, r"($\times 10^6$)"
                elif max_steps > 1.5e3:
                    sf, sl = 1e3, r"($\times 10^3$)"
                else:
                    sf, sl = 1.0, ""

                ret_mean  = f"Return ({metric_type})"
                ret_std   = f"Return Std ({metric_type})"
                viol_mean = f"Number of Violations ({metric_type})"
                viol_std  = f"Number of Violations Std ({metric_type})"

                _plot_single_metric(ax_ret,  ret_mean,  ret_std,  day_data, config, scale_factor=sf, chance_filter=chance_filter)
                _plot_single_metric(ax_viol, viol_mean, viol_std, day_data, config, scale_factor=sf, chance_filter=chance_filter)

                ax_ret.set_ylabel(f"$d={day_num}$", fontsize=12)
                ax_viol.set_ylabel("")
                if i == 0:
                    ax_ret.set_title(r"\textbf{Return}", fontsize=14)
                    ax_viol.set_title(r"\textbf{Number of Violations}", fontsize=14)
                ax_ret.set_xlabel(f"Training Steps {sl}", fontsize=14)
                ax_viol.set_xlabel(f"Training Steps {sl}", fontsize=14)

            handles, labels = axes[0, 0].get_legend_handles_labels()
            n_cols_legend = max(len(handles) // 2, 1)
            fig.legend(
                handles, labels,
                loc="lower center", bbox_to_anchor=(0.5, 0.01),
                ncol=n_cols_legend, frameon=False,
                title=r"\textbf{Model}", fontsize=10, title_fontsize=14,
            )
            fig.tight_layout(rect=[0, 0.08, 1, 0.98], h_pad=2.0, w_pad=1.0)
            pdf.savefig(fig, bbox_inches="tight")
            display(fig)
            plt.close(fig)

    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
#  Simulation visualisation helpers (soil-moisture environment plots)
# ---------------------------------------------------------------------------

def _populate_simulation_axes(ax1, df, constants):
    """
    Core rendering logic: draw soil-moisture simulation onto *ax1*.

    Returns ``(moisture_line, rain_bars, irrigation_bars, ax2_lines)``
    for legend construction.
    """
    C = _SIM_COLORS
    It_mm = df["History It (before scaling)"] * 1000
    Rain = df["History Rainfall"]
    st = df["History Soil Moisture"]
    days = np.arange(len(st))
    bar_w = 0.25

    ax2 = ax1.twinx()

    # Background zones
    ax2.axhspan(0, constants["sw"],                   color=C["hard_zone"],     alpha=0.3, zorder=0)
    ax2.axhspan(constants["sw"], constants["s_star"],  color=C["chance_zone"],   alpha=0.3, zorder=0)
    ax2.axhspan(constants["s_star"], constants["sfc"], color=C["feasible_zone"], alpha=0.3, zorder=0)
    ax2.axhspan(constants["sfc"], 1,                   color=C["chance_zone"],   alpha=0.3, zorder=0)

    # Bars
    rain_bars = ax1.bar(days - bar_w / 2, Rain,  width=bar_w, color=C["rain"],       label="Rainfall (mm)",   zorder=5)
    irr_bars  = ax1.bar(days + bar_w / 2, It_mm, width=bar_w, color=C["irrigation"], label="Irrigation (mm)", zorder=5)

    # Threshold lines
    ax2.axhline(y=constants["sfc"],    color=C["sfc_line"],    linestyle="--", linewidth=1.5, zorder=2)
    ax2.axhline(y=constants["s_star"], color=C["s_star_line"], linestyle="--", linewidth=1.5, zorder=2)
    ax2.axhline(y=constants["sw"],     color=C["sw_line"],     linestyle="--", linewidth=1.5, zorder=2)

    # Soil moisture
    moisture_line = ax2.plot(days, st, color=C["moisture"], linewidth=1.5, zorder=6)[0]

    # Formatting
    ax1.set_ylabel(r"Precipitation \& Irrigation (mm)", fontsize=14, color=C["rain"])
    ax2.set_ylabel(r"Soil Moisture Level", fontsize=14, color=C["moisture"])
    ax1.set_ylim(0, max(np.max(Rain), np.max(It_mm)) * 1.25 if len(It_mm) > 0 else 10)
    ax2.set_ylim(0, 1)
    ax1.set_zorder(10)
    ax1.patch.set_visible(False)
    ax1.tick_params(axis="y", labelsize=12, colors=C["rain"])
    ax1.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12, colors=C["moisture"])
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)

    return moisture_line, rain_bars, irr_bars, ax2.get_lines()


def _build_simulation_legend(fig, moisture_line, rain_bars, irr_bars, ax2_lines):
    """Add a three-row shared legend below a simulation figure."""
    C = _SIM_COLORS
    row1 = [
        (moisture_line, "Soil Moisture"),
        (rain_bars,     "Rainfall (mm)"),
        (irr_bars,      "Irrigation (mm)"),
    ]
    row2 = [
        (ax2_lines[0], r"Field Capacity ($s_{fc} = 0.65$)"),
        (ax2_lines[1], r"Water Stress Point ($s^* = 0.35$)"),
        (ax2_lines[2], r"Permanent Wilting Point ($s_w = 0.3$)"),
    ]
    row3 = [
        (plt.Rectangle((0, 0), 1, 1, fc=C["feasible_zone"], alpha=0.3), "Feasible Region"),
        (plt.Rectangle((0, 0), 1, 1, fc=C["chance_zone"],   alpha=0.3), "Chance-Constrained Region"),
        (plt.Rectangle((0, 0), 1, 1, fc=C["hard_zone"],     alpha=0.3), "Hard-Constrained Region"),
    ]
    for items, y_off in [(row1, 0.02), (row2, -0.03), (row3, -0.08)]:
        handles, labels = zip(*items)
        fig.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, y_off),
            ncol=len(handles), frameon=False, fontsize=12,
        )


def batch_plot_simulations(base_dir: str, episode: int, config: dict):
    """Find all simulation CSVs for *episode* and render each as a standalone PDF."""
    print(f"\n--- Simulation plots for episode {episode} ---")
    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    csv_name = f"simulation_data_episode{episode}.csv"

    for root, _, files in os.walk(base_dir):
        if csv_name not in files:
            continue
        csv_path = os.path.join(root, csv_name)
        dir_name = os.path.basename(root)
        print(f"\nFound: {dir_name}")

        try:
            df = pd.read_csv(csv_path)
            info = parse_model_info(dir_name, config)
            if info:
                algo_disp = (info["algo"]
                             .replace("DDPGLagrangian", "DDPG Lagrangian")
                             .replace("SACLagrangian", "SAC Lagrangian"))
                day_num = info["days"].replace("days", "")
                if "Lagrangian" in info["algo"]:
                    title = (fr"\textbf{{{algo_disp} ($\alpha={info['chance']}$, "
                             fr"$d={day_num}$) | Evaluation (Seed 59)}}")
                else:
                    title = fr"\textbf{{{algo_disp} ($d={day_num}$) | Evaluation (Seed 59)}}"
            else:
                title = fr"\textbf{{{dir_name} — Evaluation (Seed 59)}}"

            save_path = os.path.join(out_dir, f"simulation_{dir_name}_e{episode}.pdf")
            fig, ax1 = plt.subplots(figsize=(14, 6))
            moisture_line, rain_bars, irr_bars, ax2_lines = _populate_simulation_axes(
                ax1, df, SIMULATION_CONSTANTS,
            )
            plt.title(title, fontsize=18)
            _build_simulation_legend(fig, moisture_line, rain_bars, irr_bars, ax2_lines)
            fig.tight_layout(rect=[0, 0.15, 1, 0.95])
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"  -> Saved: {save_path}")
            plt.show()
            plt.close(fig)
        except Exception as e:
            print(f"  ERROR: {e}")


def plot_all_days_comparison(base_dir: str, episode: int, config: dict):
    """
    Multi-page PDF: each page is a 2x2 grid (DDPG, SAC, DDPG-Lag, SAC-Lag)
    comparing regular and chance=0.95 models for one day configuration.
    """
    _, _, _, display = _lazy_imports()

    day_keys = set()
    for root, _, _ in os.walk(base_dir):
        info = parse_model_info(os.path.basename(root), config)
        if info:
            day_keys.add(info["days"])
    if not day_keys:
        print("No model data found.")
        return

    save_path = os.path.join(config["output_dir"], f"simulation_comparison_all_days_e{episode}.pdf")
    csv_name = f"simulation_data_episode{episode}.csv"

    with PdfPages(save_path) as pdf:
        print(f"\n--- Creating {save_path} ---")

        for day_key in sorted(day_keys):
            print(f"  -> Page: {day_key}")

            # Find the four models for this day
            models = {"DDPG": None, "SAC": None, "DDPG Lagrangian": None, "SAC Lagrangian": None}
            for root, _, files in os.walk(base_dir):
                if day_key not in root or csv_name not in files:
                    continue
                info = parse_model_info(os.path.basename(root), config)
                if not info:
                    continue
                key = (info["algo"]
                       .replace("DDPGLagrangian", "DDPG Lagrangian")
                       .replace("SACLagrangian", "SAC Lagrangian"))
                is_regular = "Lagrangian" not in info["algo"]
                is_95 = "Lagrangian" in info["algo"] and info["chance"] == "0.95"
                if (is_regular or is_95) and key in models:
                    models[key] = (os.path.join(root, csv_name), info["chance"])

            # 2x2 grid
            fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
            layout = {
                (0, 0): "DDPG",            (0, 1): "SAC",
                (1, 0): "DDPG Lagrangian",  (1, 1): "SAC Lagrangian",
            }

            moisture_line = rain_bars = irr_bars = ax2_lines = None
            for (r, c), model_name in layout.items():
                ax = axes[r, c]
                found = models.get(model_name)
                if found and os.path.exists(found[0]):
                    df = pd.read_csv(found[0])
                    ml, rb, ib, a2l = _populate_simulation_axes(ax, df, SIMULATION_CONSTANTS)
                    moisture_line, rain_bars, irr_bars, ax2_lines = ml, rb, ib, a2l
                    chance_val = found[1]
                    if model_name.endswith("Lagrangian"):
                        ax.set_title(fr"\textbf{{{model_name} ($\alpha={chance_val}$)}}", fontsize=16)
                    else:
                        ax.set_title(fr"\textbf{{{model_name} (Regular)}}", fontsize=16)
                else:
                    ax.text(0.5, 0.5, f"Data not found for\n{model_name}",
                            ha="center", va="center", fontsize=12)
                    ax.set_xticks([])
                    ax.set_yticks([])

            axes[1, 0].set_xlabel(r"Day", fontsize=14)
            axes[1, 1].set_xlabel(r"Day", fontsize=14)

            if moisture_line is not None:
                _build_simulation_legend(fig, moisture_line, rain_bars, irr_bars, ax2_lines)

            day_num = day_key.replace("days", "")
            fig.suptitle(
                fr"\textbf{{Model Performance During Evaluation, Seed 59 ($d={day_num}$)}}",
                fontsize=18, y=0.98,
            )
            fig.tight_layout(rect=[0, 0.1, 1, 0.99], h_pad=1.5, w_pad=1.0)
            pdf.savefig(fig, bbox_inches="tight")
            display(fig)
            plt.close(fig)

    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
#  Simulation statistics
# ---------------------------------------------------------------------------

def _compute_relative_yield(df) -> float | None:
    """
    Compute the Relative Yield (RY) from a simulation CSV DataFrame.

    Uses FAO-56 methodology: actual ET is derived from the recorded
    soil-moisture loss rate (``rho``), capped at potential ET (``ETmax``)
    to exclude deep-percolation losses.  Growth-stage boundaries and
    :math:`K_y` values are read from ``env.params``.
    """
    try:
        from env.params import (
            N, ZR, LINI, LDEV, LMID, LLATE,
            KY_INI, KY_DEV, KY_MID, KY_LATE,
        )
        from results.relative_yield import calculate_relative_yield
    except ImportError:
        return None  # graceful degradation if params unavailable

    ETmax = df["History ETmax"].values
    rho = df["History rho"].values
    n_days = len(ETmax)
    if n_days == 0:
        return None

    # Actual ET (mm/day) = rho * N * ZR, capped at ETmax to strip deep percolation
    actual_et = np.minimum(rho * N * ZR, ETmax)

    # Cumulative sums
    cpet_cum = np.cumsum(ETmax)
    caet_cum = np.cumsum(actual_et)

    # Growth-stage end indices (0-based cumulative day counts)
    stage_ends = [LINI, LINI + LDEV, LINI + LDEV + LMID,
                  LINI + LDEV + LMID + LLATE]
    ky_per_stage = [KY_INI, KY_DEV, KY_MID, KY_LATE]

    # Collect cumulative values at the end of each stage covered by the data
    cpet_stages, caet_stages, ky_used = [], [], []
    for ky, end_day in zip(ky_per_stage, stage_ends):
        if end_day > n_days:
            # Partial coverage — use whatever data we have for this stage
            prev_end = stage_ends[len(ky_used) - 1] if ky_used else 0
            if n_days > prev_end:
                cpet_stages.append(float(cpet_cum[-1]))
                caet_stages.append(float(caet_cum[-1]))
                ky_used.append(ky)
            break
        idx = end_day - 1  # 0-based
        cpet_stages.append(float(cpet_cum[idx]))
        caet_stages.append(float(caet_cum[idx]))
        ky_used.append(ky)

    if not cpet_stages:
        return None

    return calculate_relative_yield(ky_used, caet_stages, cpet_stages)


def calculate_simulation_stats(df, constants) -> dict:
    """Compute key summary statistics from a simulation CSV DataFrame."""
    irrigation_mm = df["History It (before scaling)"] * 1000
    st = df["History Soil Moisture"]
    n = len(st)

    total_irr = irrigation_mm.sum()
    prob_above_s_star = (st >= constants["s_star"]).sum() / n
    prob_below_sfc    = (st <= constants["sfc"]).sum()    / n
    prob_above_sw     = (st >= constants["sw"]).sum()     / n
    n_violations = int(
        (st < constants["s_star"]).sum()
        + (st < constants["sw"]).sum()
        + (st > constants["sfc"]).sum()
    )

    ep_return = df["Episode Return"].iloc[-1] if "Episode Return" in df.columns else None

    # Relative Yield (may return None if crop params are unavailable)
    ry = _compute_relative_yield(df)

    return {
        "total_irrigation_mm": total_irr,
        "prob_above_s_star":   prob_above_s_star,
        "prob_below_sfc":      prob_below_sfc,
        "prob_above_sw":       prob_above_sw,
        "num_violations":      n_violations,
        "episode_return":      ep_return,
        "relative_yield":      ry,
    }


def print_all_simulation_stats(
    base_dir: str, episode: int, constants: dict, config: dict,
) -> pd.DataFrame | None:
    """
    Print a Markdown table of simulation statistics and return a DataFrame.

    Saves results as both CSV and a publication-ready LaTeX booktabs table
    in ``config["output_dir"]``.
    """
    _, event_accumulator, _, _ = _lazy_imports()

    print(f"\n--- Statistics for Episode {episode} ---")
    print("| Model | Day | α | Irrig. (mm) | P(s>=s*) | P(s<=sfc) | P(s>=sw) | Violations | RY | Return |")
    print("|-------|-----|---|-------------|----------|-----------|----------|------------|----|--------|")

    all_stats = []
    csv_name = f"simulation_data_episode{episode}.csv"

    for root, _, files in os.walk(base_dir):
        if csv_name not in files:
            continue
        csv_path = os.path.join(root, csv_name)
        dir_name = os.path.basename(root)

        try:
            df = pd.read_csv(csv_path)
            info = parse_model_info(dir_name, config)
            if not info:
                continue

            algo_disp = (info["algo"]
                         .replace("DDPGLagrangian", "DDPG-L")
                         .replace("SACLagrangian", "SAC-L"))
            day_num = info["days"].replace("days", "")
            stats = calculate_simulation_stats(df, constants)

            # Fallback: read return from TensorBoard if missing in CSV
            if stats["episode_return"] is None:
                tb_path = os.path.join(root, "tb")
                if os.path.exists(tb_path):
                    try:
                        ea = event_accumulator.EventAccumulator(tb_path)
                        ea.Reload()
                        if "Averageeval/TestEpRet" in ea.Tags().get("scalars", []):
                            stats["episode_return"] = ea.Scalars("Averageeval/TestEpRet")[-1].value
                    except Exception:
                        pass

            ret_str = f"{stats['episode_return']:.1f}" if stats["episode_return"] is not None else "N/A"
            ry_str = f"{stats['relative_yield']:.4f}" if stats["relative_yield"] is not None else "N/A"

            print(
                f"| {algo_disp} | {day_num} | {info['chance']} "
                f"| {stats['total_irrigation_mm']:.2f} "
                f"| {stats['prob_above_s_star'] * 100:.2f}% "
                f"| {stats['prob_below_sfc'] * 100:.2f}% "
                f"| {stats['prob_above_sw'] * 100:.2f}% "
                f"| {stats['num_violations']} "
                f"| {ry_str} "
                f"| {ret_str} |"
            )

            all_stats.append({
                "Model": algo_disp,
                "Days": int(day_num),
                "Alpha": info["chance"],
                "Total Irrigation (mm)": round(stats["total_irrigation_mm"], 2),
                "P(s >= s*)": round(stats["prob_above_s_star"], 4),
                "P(s <= sfc)": round(stats["prob_below_sfc"], 4),
                "P(s >= sw)": round(stats["prob_above_sw"], 4),
                "Violations": stats["num_violations"],
                "RY": round(stats["relative_yield"], 4) if stats["relative_yield"] is not None else None,
                "Return": round(stats["episode_return"], 1) if stats["episode_return"] is not None else None,
            })
        except Exception as e:
            print(f"  ERROR: {e}")

    if not all_stats:
        print("No statistics computed.")
        return None

    stats_df = pd.DataFrame(all_stats).sort_values(["Days", "Alpha", "Model"])

    # --- Save CSV ---
    csv_out = os.path.join(config["output_dir"], f"simulation_stats_e{episode}.csv")
    stats_df.to_csv(csv_out, index=False)
    print(f"\nStatistics saved to: {csv_out}")

    # --- Save publication-ready LaTeX table ---
    tex_out = os.path.join(config["output_dir"], f"simulation_stats_e{episode}.tex")
    _export_stats_latex(stats_df, episode, tex_out)
    print(f"LaTeX table saved to: {tex_out}")

    return stats_df


def _export_stats_latex(stats_df: pd.DataFrame, episode: int, output_path: str):
    """Write *stats_df* as a publication-ready LaTeX booktabs table."""
    # LaTeX-friendly column names
    col_map = {
        "Model": "Model",
        "Days": "$d$",
        "Alpha": r"$\alpha$",
        "Total Irrigation (mm)": r"Irrig.~(mm)",
        "P(s >= s*)": r"$\Pr(s \!\geq\! s^*)$",
        "P(s <= sfc)": r"$\Pr(s \!\leq\! s_{fc})$",
        "P(s >= sw)": r"$\Pr(s \!\geq\! s_w)$",
        "Violations": "Viol.",
        "RY": "RY",
        "Return": "Return",
    }
    tex_df = stats_df.rename(columns=col_map)

    # Format numeric columns
    fmt = {
        r"Irrig.~(mm)": "{:.2f}",
        r"$\Pr(s \!\geq\! s^*)$": "{:.4f}",
        r"$\Pr(s \!\leq\! s_{fc})$": "{:.4f}",
        r"$\Pr(s \!\geq\! s_w)$": "{:.4f}",
        "RY": "{:.4f}",
        "Return": "{:.1f}",
    }
    for col, f in fmt.items():
        if col in tex_df.columns:
            tex_df[col] = tex_df[col].map(
                lambda x, _f=f: _f.format(x) if pd.notna(x) else "--"
            )
    tex_df = tex_df.fillna("--")

    # Build the LaTeX string manually for full control
    n_cols = len(tex_df.columns)
    col_fmt = "ll" + "r" * (n_cols - 2)
    header = " & ".join(tex_df.columns) + r" \\"
    rows = "\n".join(
        " & ".join(str(v) for v in row) + r" \\"
        for row in tex_df.values
    )

    tex = (
        r"\begin{table}[ht]" "\n"
        r"\centering" "\n"
        f"\\caption{{Simulation statistics at episode {episode}.}}\n"
        f"\\label{{tab:sim_stats_e{episode}}}\n"
        f"\\begin{{tabular}}{{{col_fmt}}}\n"
        r"\toprule" "\n"
        f"{header}\n"
        r"\midrule" "\n"
        f"{rows}\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )

    with open(output_path, "w") as f:
        f.write(tex)


# ###########################################################################
#                                                                           #
#  SECTION 3 — MAIN ENTRY POINT                                           #
#  ``python results/plots.py`` or ``run_analysis()`` from a notebook       #
#                                                                           #
# ###########################################################################


def run_analysis(
    *,
    episode: int = 499,
    chance_filters: list[str] | None = None,
    config: dict | None = None,
):
    """
    Run the full post-hoc analysis pipeline.

    Parameters
    ----------
    episode : int
        Which training episode's simulation data to analyse.
    chance_filters : list of str or None
        Each value (e.g. ``["0.95"]``) produces an extra performance-grid
        PDF restricted to regular models + Lagrangian models with that
        chance constraint.  The full (unfiltered) grid is always generated.
        Pass ``None`` to skip filtered grids.
    config : dict or None
        Override the default CONFIG.  Pass ``None`` for defaults.
    """
    if config is None:
        config = _default_config()

    _apply_publication_style()
    os.makedirs(config["output_dir"], exist_ok=True)

    # 1. Load TensorBoard data
    print("=" * 60)
    print("  Loading TensorBoard data")
    print("=" * 60)
    data = load_all_runs(config["base_directory"], config)

    print("\n--- Data Loading Summary ---")
    if not data:
        print("No data loaded. Check CONFIG['base_directory'].")
        return
    for name, exp in sorted(data.items()):
        print(f"  {name}: {len(exp['dfs'])} run(s)")

    # 2. Performance grids (all models)
    print("\n" + "=" * 60)
    print("  Performance Grids")
    print("=" * 60)
    plot_performance_grid(data, config)

    for cf in (chance_filters or []):
        suffix = f"_chance{cf.replace('.', '')}"
        plot_performance_grid(data, config, chance_filter=cf, output_suffix=suffix)

    # 3. Individual simulation plots
    print("\n" + "=" * 60)
    print("  Individual Simulation Plots")
    print("=" * 60)
    batch_plot_simulations(config["base_directory"], episode, config)

    # 4. Comparative 2x2 grids
    print("\n" + "=" * 60)
    print("  Simulation Comparison Grids")
    print("=" * 60)
    plot_all_days_comparison(config["base_directory"], episode, config)

    # 5. Statistics table
    print("\n" + "=" * 60)
    print("  Simulation Statistics")
    print("=" * 60)
    stats_df = print_all_simulation_stats(
        config["base_directory"], episode, SIMULATION_CONSTANTS, config,
    )
    if stats_df is not None:
        _, _, _, display = _lazy_imports()
        display(stats_df)

    print("\n" + "=" * 60)
    print("  Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_analysis(chance_filters=["0.95"])

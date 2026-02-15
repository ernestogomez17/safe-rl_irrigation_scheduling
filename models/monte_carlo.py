"""
Monte-Carlo Stochastic MPC Baseline for Irrigation Scheduling
=============================================================

This module implements a Stochastic Model Predictive Control (SMPC) baseline 
for agricultural irrigation, evaluating chance-constrained water scheduling 
using Monte-Carlo rainfall simulations.

Reference
---------
Based on the simulation-optimization framework detailed in:
Roy, A., Narvekar, P., Murtugudde, R., Shinde, V., & Ghosh, S. (2021). 
"Short and Medium Range Irrigation Scheduling Using Stochastic 
Simulation-Optimization Framework With Farm-Scale Ecohydrological Model 
and Weather Forecasts." Water Resources Research. 
DOI: 10.1029/2020WR029004

Methodology
-----------
1. Forecast Error Modeling (Training: 2015-2016):
   Fits a forecast error model to historical data. For each lead time `d`, 
   a linear regression maps the simulated forecast (shifted observed rain + noise) 
   to the actual observed rain. An exponential distribution rate parameter `gamma` 
   and rain probability `p` are estimated per forecast-intensity bin.

2. SMPC Evaluation (Simulation: 2017):
   Simulates the environment day-by-day. At each step, the controller:
   - Retrieves `p` and `gamma` based on the current weather forecast.
   - Generates Monte-Carlo rainfall sample paths.
   - Propagates soil moisture dynamics forward.
   - Performs a bisection search to find the minimum irrigation action that 
     satisfies the chance constraint (e.g., maintaining moisture above `S_STAR`).

Usage
-----
The script functions as a standalone CLI tool and outputs a summary to `stdout` 
along with a detailed CSV. This is designed for direct benchmarking against 
reinforcement learning (RL) evaluation episodes.

    $ python mc_irrigation_baseline.py --n-days-ahead 7 --chance-pct 0.75

Notes
-----
Soil parameters are strictly imported from `water_environment.py` to ensure 
both the MPC and RL approaches are evaluated on identical physical dynamics.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Ensure project root import path (for env.params)
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from env.params import (
    SW,
    SH,
    SFC,
    S_STAR,
    N as N_SOIL,
    ZR,
    KS,
    BETA,
    SEASON_START_DATE as SEASON_START_DOY,
    LINI,
    LDEV,
    LMID,
    LLATE,
    KCINI,
    KCMID,
    KCEND,
    KY_INI,
    KY_DEV,
    KY_MID,
    KY_LATE,
)


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def kc_function(day_of_year: int) -> float:
    adjusted = (day_of_year - SEASON_START_DOY) % 365
    total_growth = LINI + LDEV + LMID + LLATE
    trans = 365 - total_growth

    if adjusted <= LINI:
        return KCINI
    if adjusted <= LINI + LDEV:
        return KCINI + (KCMID - KCINI) * (adjusted - LINI) / LDEV
    if adjusted <= LINI + LDEV + LMID:
        return KCMID
    if adjusted <= total_growth:
        return KCMID - (KCMID - KCEND) * (adjusted - LINI - LDEV - LMID) / LLATE
    if adjusted <= total_growth + trans:
        return KCEND - (KCEND - KCINI) * (adjusted - total_growth) / trans
    return KCINI


def calculate_ET_o(row: pd.Series) -> float:
    Tmax = row["Daily Tmax (C)"]
    Tmin = row["Daily Tmin (C)"]
    DSWR = row["Daily DSWR"]
    DLWR = row["Daily DLWR"]
    USWR = row["Daily USWR"]
    ULWR = row["Daily ULWR"]
    UGRD = row["Daily UGRD"]
    VGRD = row["Daily VGRD"]
    Pres = row["Daily Pres (kPa)"]
    date = row["Date"]

    tmean = (Tmax + Tmin) / 2.0
    Rs = (DSWR + DLWR - USWR - ULWR) * 0.0864
    u2 = (np.sqrt(UGRD**2 + VGRD**2) * 4.87) / np.log(67.8 * 10 - 5.42)

    delta_val = (
        4098 * 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3)) / (tmean + 237.3) ** 2
    )
    psy = 0.000665 * Pres
    DT = delta_val / (delta_val + psy * (1 + 0.34 * u2))
    PT = psy / (delta_val + psy * (1 + 0.34 * u2))
    TT = 900 * u2 / (tmean + 273)

    et_max = 0.6108 * np.exp(17.27 * Tmax / (Tmax + 237.3))
    et_min = 0.6108 * np.exp(17.27 * Tmin / (Tmin + 237.3))
    es = (et_max + et_min) / 2.0
    ea = et_min

    jday = date.timetuple().tm_yday
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * jday)
    yen = 0.409 * np.sin(2 * np.pi / 365 * jday - 1.39)
    phi = np.pi * 20 / 180
    ws = np.arccos(-np.tan(phi) * np.tan(yen))
    Ra = (24 * 60 / np.pi) * 0.082 * dr * (
        np.sin(phi) * np.sin(yen) + np.cos(phi) * np.cos(yen) * np.sin(ws)
    )
    Rso = (0.75 + 2e-5 * 602) * Ra
    Rns = 0.77 * Rs
    Rnl = (
        4.903e-9
        * ((Tmax + 273.16) ** 4 + (Tmin + 273.16) ** 4)
        / 2
        * (0.34 - 0.14 * np.sqrt(ea))
        * (1.35 * Rs / max(Rso, 1e-6) - 0.35)
    )
    Rn = Rns - Rnl

    ET_o = DT * (0.408 * Rn) + PT * TT * (es - ea)
    return float(ET_o)


def rho(s: float | np.ndarray, ETmax: float) -> float | np.ndarray:
    """Normalized total soil-moisture loss rate (1/day)."""
    eta = ETmax / (N_SOIL * ZR)
    eta_w = 0.15 * ETmax / (N_SOIL * ZR)
    m = KS / (N_SOIL * ZR * (np.exp(BETA * (1 - SFC)) - 1))

    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)

    mask_sh = (s > 0) & (s <= SH)
    mask_sw = (s > SH) & (s <= SW)
    mask_ss = (s > SW) & (s <= S_STAR)
    mask_fc = (s > S_STAR) & (s <= SFC)
    mask_hi = s > SFC

    out[mask_sh] = 0.0
    out[mask_sw] = eta_w * (s[mask_sw] - SH) / (SW - SH)
    out[mask_ss] = eta_w + (eta - eta_w) * (s[mask_ss] - SW) / (S_STAR - SW)
    out[mask_fc] = eta
    out[mask_hi] = eta + m * (np.exp(BETA * (s[mask_hi] - SFC)) - 1)

    return float(out) if out.ndim == 0 else out


def soil_update(
    s: float | np.ndarray,
    rain_mm: float | np.ndarray,
    irrig_mm: float | np.ndarray,
    ETmax: float,
) -> np.ndarray:
    total_water = np.asarray(rain_mm, dtype=float) + np.asarray(irrig_mm, dtype=float)
    h = total_water / (N_SOIL * ZR)
    ds = h - rho(s, ETmax)
    return np.clip(np.asarray(s, dtype=float) + ds, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Forecast error model
# ---------------------------------------------------------------------------


def _default_forecast_column_candidates(lead_1_based: int) -> list[str]:
    d = lead_1_based
    return [
        f"Rain_fc_d{d}",
        f"Rain_fc_{d}",
        f"Forecast Rain Lead {d} (mm)",
        f"Forecast Rainfall Lead {d} (mm)",
        f"Forecast Rain D{d} (mm)",
        f"Forecast D{d} Rain (mm)",
        f"GEFS Rain Lead {d} (mm)",
        f"Lead{d}_Forecast_Rain_mm",
        f"Lead_{d}_Forecast_Rain_mm",
    ]


def _resolve_forecast_columns(
    columns: Sequence[str],
    n_leads: int,
    explicit_cols: Sequence[str] | None = None,
) -> list[str]:
    cols = list(columns)

    if explicit_cols is not None:
        if len(explicit_cols) < n_leads:
            raise ValueError(
                f"Need at least {n_leads} forecast columns, got {len(explicit_cols)}"
            )
        missing = [c for c in explicit_cols[:n_leads] if c not in cols]
        if missing:
            raise ValueError(f"Forecast columns not found in dataframe: {missing}")
        return list(explicit_cols[:n_leads])

    resolved: list[str] = []
    for d in range(1, n_leads + 1):
        found = None
        for cand in _default_forecast_column_candidates(d):
            if cand in cols:
                found = cand
                break
        if found is None:
            # fallback: regex match contains lead index and forecast+rain terms
            patt = re.compile(rf"(?i)(forecast|hindcast).*(lead|d)?[_\s-]*{d}.*rain")
            matches = [c for c in cols if patt.search(c)]
            if matches:
                found = matches[0]
        if found is None:
            head = ", ".join(cols[:20])
            raise ValueError(
                f"Could not resolve forecast column for lead {d}. "
                f"Pass --forecast-cols explicitly. First columns: {head}"
            )
        resolved.append(found)

    return resolved


@dataclass
class ForecastModel:
    bin_centers: np.ndarray
    p_table: np.ndarray
    mean_intercepts: np.ndarray
    mean_slopes: np.ndarray
    n_soil: float
    z_r: float

    def lookup(self, lead: int, forecast_mm: float) -> tuple[float, float, float]:
        x = float(forecast_mm)
        p = float(np.interp(x, self.bin_centers, self.p_table[lead]))
        p = float(np.clip(p, 0.0, 1.0))

        mean_rain_mm = self.mean_intercepts[lead] + self.mean_slopes[lead] * x
        mean_rain_mm = float(max(mean_rain_mm, 1e-6))

        # Equation 8 in normalized rainfall depth h = R/(n*Zr): E[h] = 1/gamma
        gamma_h = float((self.n_soil * self.z_r) / mean_rain_mm)
        gamma_h = max(gamma_h, 1e-9)

        return p, gamma_h, mean_rain_mm


def _build_forecast_model_from_hindcasts(
    df_train: pd.DataFrame,
    forecast_cols: Sequence[str],
    n_leads: int,
    bin_step: float = 1.0,
) -> ForecastModel:
    obs = df_train["Observed Rainfall (mm)"].to_numpy(dtype=float)
    n = len(obs)

    if n < n_leads + 30:
        raise ValueError("Training window too short for lead-specific model fitting")

    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    max_fc = 0.0
    for d in range(n_leads):
        fc = df_train[forecast_cols[d]].to_numpy(dtype=float)
        fc = np.nan_to_num(fc, nan=0.0)
        fc = np.clip(fc, 0.0, None)

        # Align lead-d forecast with observed rain at +d
        fc_aligned = fc[: n - d]
        ob_aligned = obs[d:]

        if len(fc_aligned) == 0:
            raise ValueError(f"No aligned samples for lead {d + 1}")

        pairs.append((fc_aligned, ob_aligned))
        max_fc = max(max_fc, float(np.max(fc_aligned)))

    max_fc = max(max_fc, bin_step)
    bin_centers = np.arange(0.0, max_fc + bin_step, bin_step)
    n_bins = len(bin_centers)

    p_table = np.zeros((n_leads, n_bins), dtype=float)
    mean_intercepts = np.zeros(n_leads, dtype=float)
    mean_slopes = np.zeros(n_leads, dtype=float)

    for lead, (fc, ob) in enumerate(pairs):
        order = np.argsort(fc)
        fc_s = fc[order]
        ob_s = ob[order]

        prev = float(np.mean(ob > 0))
        left = 0

        for b, center in enumerate(bin_centers):
            right_edge = center + bin_step
            right = int(np.searchsorted(fc_s, right_edge, side="right"))
            if right <= left:
                p_table[lead, b] = prev
                continue

            seg = ob_s[left:right]
            p_bin = float(np.mean(seg > 0))
            p_table[lead, b] = p_bin
            prev = p_bin
            left = right

        mask = ob > 0
        if int(mask.sum()) >= 10:
            x = fc[mask]
            y = ob[mask]
            A = np.vstack([np.ones_like(x), x]).T
            coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            mean_intercepts[lead] = float(coef[0])
            mean_slopes[lead] = float(coef[1])
        else:
            # Fallback to unconditional positive-rain mean
            pos = ob[ob > 0]
            mean_intercepts[lead] = float(pos.mean()) if len(pos) else 1.0
            mean_slopes[lead] = 0.0

    return ForecastModel(
        bin_centers=bin_centers,
        p_table=p_table,
        mean_intercepts=mean_intercepts,
        mean_slopes=mean_slopes,
        n_soil=N_SOIL,
        z_r=ZR,
    )


# ---------------------------------------------------------------------------
# MC rainfall generation (Eq. 9 mixed distribution)
# ---------------------------------------------------------------------------

def sample_rainfall_mixed(
    p_nonzero: float,
    gamma_h: float,
    n_samples: int,
    n_soil: float,
    z_r: float,
    rng: np.random.Generator,
) -> np.ndarray:
    p = float(np.clip(p_nonzero, 0.0, 1.0))
    rain_mm = np.zeros(n_samples, dtype=float)
    occurs = rng.random(n_samples) < p
    n_occ = int(occurs.sum())
    if n_occ == 0:
        return rain_mm

    if gamma_h <= 0:
        return rain_mm

    h = rng.exponential(scale=1.0 / gamma_h, size=n_occ)
    rain_mm[occurs] = h * (n_soil * z_r)
    return rain_mm


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def find_minimum_irrigation(
    s_t: float,
    forecast_params: Sequence[tuple[float, float]],
    etmax_schedule: Sequence[float],
    n_mc: int,
    chance_pct: float,
    n_soil: float,
    z_r: float,
    max_irrig_mm_cap: float,
    rng: np.random.Generator,
    max_iters: int = 30,
    tol: float = 1e-4,
) -> float:
    """
    Returns irrigation (mm) for day 0 only.

    Constraint: for each day in horizon, P(s_{t+i} >= S_STAR) >= chance_pct.
    """
    n_days = len(forecast_params)

    # Equation 5 cap in normalized units, then convert to mm.
    sum_etmax = float(np.sum(etmax_schedule))
    i_max_norm = min(0.15 * n_days, sum_etmax / (n_soil * z_r))
    i_max_eq5_mm = i_max_norm * (n_soil * z_r)

    max_irrig_mm = min(float(max_irrig_mm_cap), float(i_max_eq5_mm))
    max_irrig_mm = max(0.0, max_irrig_mm)

    rain_samples = np.zeros((n_days, n_mc), dtype=float)
    for d, (p_d, gamma_h_d) in enumerate(forecast_params):
        rain_samples[d] = sample_rainfall_mixed(
            p_nonzero=p_d,
            gamma_h=gamma_h_d,
            n_samples=n_mc,
            n_soil=n_soil,
            z_r=z_r,
            rng=rng,
        )

    def is_safe(irrig_mm: float) -> bool:
        s = np.full(n_mc, s_t, dtype=float)
        for d in range(n_days):
            irr_d = irrig_mm if d == 0 else 0.0
            s = soil_update(s, rain_samples[d], irr_d, float(etmax_schedule[d]))
            if float(np.mean(s >= S_STAR)) < chance_pct:
                return False
        return True

    if is_safe(0.0):
        return 0.0

    if not is_safe(max_irrig_mm):
        return max_irrig_mm

    lo, hi = 0.0, max_irrig_mm
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        if is_safe(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break

    return hi


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    dates: list = field(default_factory=list)
    soil_moisture: list = field(default_factory=list)
    irrigation_mm: list = field(default_factory=list)
    rain_mm: list = field(default_factory=list)
    ET_o: list = field(default_factory=list)
    ETmax: list = field(default_factory=list)
    Kc: list = field(default_factory=list)
    rho_vals: list = field(default_factory=list)
    cpet_stages: np.ndarray = field(default_factory=lambda: np.zeros(4))
    caet_stages: np.ndarray = field(default_factory=lambda: np.zeros(4))
    violations_s_star: int = 0
    violations_sfc: int = 0
    violations_sw: int = 0
    total_irrigation_mm: float = 0.0
    total_days: int = 0

    def calculate_relative_yield(self, ky_values: Sequence[float]) -> float:
        """Calculates overall Relative Yield (RY) using accumulated ETs."""
        numerator = 0.0
        denominator = sum(ky_values)
        
        for k in range(4):
            cpet = self.cpet_stages[k]
            caet = self.caet_stages[k]
            ky = ky_values[k]
            
            if cpet > 0.0:
                deficit_ratio = max(0.0, 1.0 - (caet / cpet))
                yl_k = ky * deficit_ratio
                numerator += yl_k

        if denominator == 0.0:
            return 1.0
            
        overall_yield_loss = numerator / denominator
        ry = 1.0 - overall_yield_loss
        return max(0.0, ry)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Date": self.dates,
                "Soil_Moisture": self.soil_moisture,
                "Irrigation_mm": self.irrigation_mm,
                "Rain_mm": self.rain_mm,
                "ET_o": self.ET_o,
                "ETmax": self.ETmax,
                "Kc": self.Kc,
                "rho": self.rho_vals,
            }
        )


def run_simulation(
    weather_df: pd.DataFrame,
    n_days_ahead: int = 7,
    chance_pct: float = 0.75,
    n_mc: int = 600,
    max_irrig_m: float = 0.06,
    eval_start: str = "2017-04-10",
    eval_end: str = "2018-04-09",
    train_start: str = "2015-01-01",
    train_end: str = "2016-12-31",
    seed: int = 42,
    verbose: bool = True,
    forecast_cols: Sequence[str] | None = None,
    decision_stride_days: int | None = None,
) -> SimulationResult:
    """
    decision_stride_days:
    - None: use block scheduling of size n_days_ahead (legacy behavior).
    - 1: receding horizon (recompute daily).
    - N: recompute every N days.
    """
    rng = np.random.default_rng(seed)
    df = weather_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    if decision_stride_days is None:
        decision_stride_days = n_days_ahead

    if decision_stride_days <= 0:
        raise ValueError("decision_stride_days must be >= 1")

    df_train = df[(df["Date"] >= train_start) & (df["Date"] <= train_end)].reset_index(drop=True)
    df_eval = df[(df["Date"] >= eval_start) & (df["Date"] <= eval_end)].reset_index(drop=True)

    if len(df_eval) == 0:
        raise ValueError(f"No evaluation data in [{eval_start}, {eval_end}]")

    resolved_fc_cols = _resolve_forecast_columns(
        columns=df.columns,
        n_leads=n_days_ahead,
        explicit_cols=forecast_cols,
    )

    if verbose:
        print(f"Training data   : {len(df_train)} days ({train_start} to {train_end})")
        print(f"Evaluation data : {len(df_eval)} days ({eval_start} to {eval_end})")
        print("Forecast columns:")
        for i, c in enumerate(resolved_fc_cols, start=1):
            print(f"  Lead {i}: {c}")

    fm = _build_forecast_model_from_hindcasts(
        df_train=df_train,
        forecast_cols=resolved_fc_cols,
        n_leads=n_days_ahead,
        bin_step=1.0,
    )

    max_irrig_mm_cap = max_irrig_m * 1000.0
    s_t = float(rng.uniform(S_STAR, SFC))

    result = SimulationResult()
    day_idx = 0
    n_eval = len(df_eval)

    while day_idx < n_eval:
        horizon_len = min(n_days_ahead, n_eval - day_idx)
        apply_span = min(decision_stride_days, n_eval - day_idx)

        decision_row = df_eval.iloc[day_idx]

        forecast_params: list[tuple[float, float]] = []
        etmax_schedule: list[float] = []

        for d in range(horizon_len):
            row_future = df_eval.iloc[day_idx + d]
            doy = row_future["Date"].timetuple().tm_yday
            kc = kc_function(doy)
            eto = calculate_ET_o(row_future)
            etmax_schedule.append(kc * eto)

            fc_val = float(max(decision_row[resolved_fc_cols[d]], 0.0))
            p_d, gamma_h_d, _ = fm.lookup(d, fc_val)
            forecast_params.append((p_d, gamma_h_d))

        irrig_mm = find_minimum_irrigation(
            s_t=s_t,
            forecast_params=forecast_params,
            etmax_schedule=etmax_schedule,
            n_mc=n_mc,
            chance_pct=chance_pct,
            n_soil=N_SOIL,
            z_r=ZR,
            max_irrig_mm_cap=max_irrig_mm_cap,
            rng=rng,
        )

        for d in range(apply_span):
            row = df_eval.iloc[day_idx + d]
            date = row["Date"]
            actual_rain = float(row["Observed Rainfall (mm)"])
            doy = date.timetuple().tm_yday
            kc = kc_function(doy)
            eto = calculate_ET_o(row)
            etmax = kc * eto
            rho_val = float(rho(s_t, etmax))

            irr_d = irrig_mm if d == 0 else 0.0
            adjusted_doy = (doy - SEASON_START_DOY) % 365
            total_growth = LINI + LDEV + LMID + LLATE

            stage_idx = -1
            if adjusted_doy <= LINI:
                stage_idx = 0
            elif adjusted_doy <= LINI + LDEV:
                stage_idx = 1
            elif adjusted_doy <= LINI + LDEV + LMID:
                stage_idx = 2
            elif adjusted_doy <= total_growth:
                stage_idx = 3

            if stage_idx != -1:
                result.cpet_stages[stage_idx] += etmax
                # Actual ET in mm is rho * n * zr
                actual_et = rho_val * (N_SOIL * ZR) 
                result.caet_stages[stage_idx] += actual_et

            result.dates.append(date)
            result.soil_moisture.append(s_t)
            result.irrigation_mm.append(irr_d)
            result.rain_mm.append(actual_rain)
            result.ET_o.append(eto)
            result.ETmax.append(etmax)
            result.Kc.append(kc)
            result.rho_vals.append(rho_val)

            s_t = float(soil_update(s_t, actual_rain, irr_d, etmax))

            if s_t < S_STAR:
                result.violations_s_star += 1
            if s_t > SFC:
                result.violations_sfc += 1
            if s_t < SW:
                result.violations_sw += 1

            result.total_irrigation_mm += irr_d
            result.total_days += 1

        day_idx += apply_span

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MC-MPC irrigation baseline with lead-specific hindcast calibration"
    )
    parser.add_argument("--n-days-ahead", type=int, default=7, choices=range(1, 8))
    parser.add_argument("--chance-pct", type=float, default=0.75)
    parser.add_argument("--n-mc", type=int, default=600)
    parser.add_argument(
        "--max-irrig",
        type=float,
        default=0.06,
        help="Action cap in meters per decision (0.06 m = 60 mm)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="daily_weather_data.csv")
    parser.add_argument("--out", type=str, default="mc_baseline_results.csv")
    parser.add_argument(
        "--forecast-cols",
        type=str,
        default="",
        help=(
            "Comma-separated lead columns (lead1,lead2,...). "
            "If omitted, script attempts automatic resolution."
        ),
    )
    parser.add_argument(
        "--decision-stride",
        type=int,
        default=0,
        help=(
            "Re-optimization interval in days. 0 means legacy block behavior "
            "(equal to n-days-ahead). Use 1 for daily receding horizon."
        ),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    fc_cols = [c.strip() for c in args.forecast_cols.split(",") if c.strip()] or None
    stride = None if args.decision_stride == 0 else args.decision_stride

    result = run_simulation(
        weather_df=df,
        n_days_ahead=args.n_days_ahead,
        chance_pct=args.chance_pct,
        n_mc=args.n_mc,
        max_irrig_m=args.max_irrig,
        seed=args.seed,
        verbose=True,
        forecast_cols=fc_cols,
        decision_stride_days=stride,
    )

    print("\n" + "=" * 60)
    print("MC-MPC Irrigation Baseline — Summary")
    print("=" * 60)
    print(f"  Horizon (days)            : {args.n_days_ahead}")
    print(f"  Chance constraint         : {args.chance_pct:.0%}")
    print(f"  MC samples                : {args.n_mc}")
    print(f"  Total simulated days      : {result.total_days}")
    print(f"  Total irrigation          : {result.total_irrigation_mm:.2f} mm")
    final_ry = result.calculate_relative_yield([KY_INI, KY_DEV, KY_MID, KY_LATE])
    print(f"  Relative Yield (RY)       : {final_ry:.4f} ({final_ry * 100:.1f}%)")
    safe_days = result.total_days - result.violations_s_star
    print(
        f"  S_STAR violations         : {result.violations_s_star} "
        f"({safe_days}/{result.total_days} safe = "
        f"{safe_days / max(result.total_days, 1):.1%})"
    )
    print(f"  SFC violations            : {result.violations_sfc}")
    print(f"  SW violations             : {result.violations_sw}")

    out_df = result.to_dataframe()
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path.resolve()}")


if __name__ == "__main__":
    main()
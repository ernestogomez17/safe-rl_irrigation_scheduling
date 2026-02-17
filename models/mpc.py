"""
Deterministic Model Predictive Control (MPC) Baseline
=====================================================

Receding-horizon MPC that maintains soil moisture at a safe reference
level using deterministic (point) weather forecasts.

Methodology
-----------
At each decision step the controller:

1. Reads the point rainfall forecast for the next *N* days.
2. Builds a predicted ET_max schedule from weather variables.
3. Solves a constrained quadratic-tracking problem::

       min  Σ_k (s_k − s_ref)² + λ Σ_k u_k²
       s.t. s_{k+1} = f(s_k, rain_k, u_k, ETmax_k)   ∀k
            s_k ≥ S* + margin                          ∀k
            0 ≤ u_k ≤ u_max                            ∀k

4. Applies the first *stride* days of irrigation (receding horizon).

The cost is dominated by the tracking term — the controller focuses on
keeping moisture close to a safe setpoint rather than minimizing water
use.  The small regularization λ provides smoothness only.

A configurable *safety_margin* tightens the soil-moisture lower bound
beyond S* to provide deterministic robustness against forecast errors,
without resorting to Monte-Carlo sampling.

Forecast modes
--------------
- ``columns``: use explicit forecast columns from the CSV (same as MC-MPC).
- ``perfect``: use observed rainfall as an oracle forecast (upper bound).
- ``zero``:    assume zero rainfall (conservative / worst-case).

Comparison with MC-MPC (mc_irrigation_baseline.py)
--------------------------------------------------
- **MC-MPC**: stochastic chance constraint, minimizes irrigation.
- **MPC**:    deterministic quadratic tracking, controls soil moisture.

Usage
-----
    $ python mpc.py --n-days-ahead 7 --s-ref 0.50
    $ python mpc.py --n-days-ahead 7 --forecast-mode zero --safety-margin 0.05
    $ python mpc.py --n-days-ahead 7 --s-ref 0.45 --safety-margin 0.03

Notes
-----
Physics and soil parameters are imported from ``env.params`` and shared
helper functions are reused from ``mc_irrigation_baseline``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

# Ensure project root & sibling imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
_MODELS_DIR = str(Path(__file__).resolve().parent)
for _p in (_PROJECT_ROOT, _MODELS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from env.params import (
    SW,
    SFC,
    S_STAR,
    N as N_SOIL,
    ZR,
    SEASON_START_DATE as SEASON_START_DOY,
    LINI,
    LDEV,
    LMID,
    LLATE,
    KY_INI,
    KY_DEV,
    KY_MID,
    KY_LATE,
)

# Reuse physics & utilities from the MC baseline (same soil model)
from models.monte_carlo import (
    kc_function,
    calculate_ET_o,
    rho,
    soil_update,
    SimulationResult,
    _resolve_forecast_columns,
    _ensure_forecast_columns,
)


# ---------------------------------------------------------------------------
# RL-env irrigation cap (linear interpolation)
# ---------------------------------------------------------------------------

def _rl_max_irrigation_mm(n_days_ahead: int) -> float:
    """Return the per-block irrigation cap (mm) matching the RL environment.

    The RL env linearly scales MAX_IRRIGATION from 10 mm (d=1) to 60 mm (d=7).
    """
    slope = (60.0 - 10.0) / (7 - 1)      # mm per day-ahead
    intercept = 10.0 - slope * 1
    return slope * n_days_ahead + intercept


# ---------------------------------------------------------------------------
# MPC solver
# ---------------------------------------------------------------------------

def mpc_solve(
    s_t: float,
    forecast_rain_mm: np.ndarray,
    etmax_schedule: np.ndarray,
    s_ref: float,
    safety_margin: float,
    max_irrig_mm: float,
    reg_weight: float = 1e-4,
    block_mode: bool = False,
) -> np.ndarray:
    """
    Solve the deterministic MPC sub-problem.

    Parameters
    ----------
    s_t : float
        Current soil moisture (normalised, 0–1).
    forecast_rain_mm : array, shape (N,)
        Point rainfall forecast for each day in the horizon.
    etmax_schedule : array, shape (N,)
        Maximum ET for each day in the horizon.
    s_ref : float
        Soil moisture setpoint to track.
    safety_margin : float
        Tightening margin added to S_STAR for constraint robustness.
    max_irrig_mm : float
        Maximum irrigation per day (mm).
    block_mode : bool
        If True, only day-0 irrigation is allowed (u[1:] = 0), matching
        the RL environment's block-scheduling semantics.
    reg_weight : float
        Quadratic regularisation weight on irrigation.

    Returns
    -------
    u_opt : ndarray, shape (N,)
        Optimal irrigation plan (mm) for each day in the horizon.
    """
    n = len(forecast_rain_mm)
    rain = np.asarray(forecast_rain_mm, dtype=float)
    etmax = np.asarray(etmax_schedule, dtype=float)
    s_lb = S_STAR + safety_margin

    # ------ forward simulator ------
    def _simulate(u: np.ndarray) -> np.ndarray:
        states = np.empty(n)
        s = s_t
        for k in range(n):
            s = float(soil_update(s, rain[k], u[k], etmax[k]))
            states[k] = s
        return states

    # Fast path: if zero irrigation already tracks s_ref well and is safe,
    # skip the solver entirely.
    states_zero = _simulate(np.zeros(n))
    if np.all(states_zero >= s_lb) and np.max(np.abs(states_zero - s_ref)) < 0.02:
        return np.zeros(n)

    # ------ objective ------
    def objective(u: np.ndarray) -> float:
        states = _simulate(u)
        tracking = float(np.sum((states - s_ref) ** 2))
        reg = reg_weight * float(np.sum(u ** 2))
        return tracking + reg

    # ------ constraints: s_k >= S_STAR + margin ------
    constraints = []
    for k in range(n):
        def _moisture_lb(u: np.ndarray, _k: int = k) -> float:
            s = s_t
            for j in range(_k + 1):
                s = float(soil_update(s, rain[j], u[j], etmax[j]))
            return s - s_lb

        constraints.append({"type": "ineq", "fun": _moisture_lb})

    if block_mode and n > 1:
        bounds = [(0.0, max_irrig_mm)] + [(0.0, 0.0)] * (n - 1)
    else:
        bounds = [(0.0, max_irrig_mm)] * n

    # ------ solve (attempt 1: cold start) ------
    u0 = np.zeros(n)
    res = minimize(
        objective,
        u0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )

    if res.success:
        return np.clip(res.x, 0.0, max_irrig_mm)

    # ------ solve (attempt 2: warm start) ------
    u_warm = np.zeros(n)
    u_warm[0] = max_irrig_mm * 0.5
    res2 = minimize(
        objective,
        u_warm,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )

    if res2.success:
        return np.clip(res2.x, 0.0, max_irrig_mm)

    # ------ fallback: maximum on day 0 ------
    u_fb = np.zeros(n)
    u_fb[0] = max_irrig_mm
    return u_fb


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    weather_df: pd.DataFrame,
    n_days_ahead: int = 7,
    s_ref: float | None = None,
    safety_margin: float = 0.02,
    max_irrig_m: float | None = None,
    eval_start: str = "2017-04-10",
    eval_end: str = "2018-04-09",
    seed: int = 42,
    verbose: bool = True,
    forecast_cols: Sequence[str] | None = None,
    forecast_mode: str = "perfect",
    decision_stride_days: int = 1,
    reg_weight: float = 1e-4,
    block_mode: bool = False,
) -> SimulationResult:
    """
    Run the deterministic MPC baseline over an evaluation period.

    Parameters
    ----------
    weather_df : DataFrame
        Must contain Date, weather variables, and forecast columns.
    n_days_ahead : int
        Prediction horizon (days).
    s_ref : float or None
        Soil moisture setpoint.  Defaults to ``(S_STAR + SFC) / 2``.
    safety_margin : float
        Constraint tightening margin added to S_STAR.
    max_irrig_m : float or None
        Irrigation cap in **metres** per block.  When *None* (default),
        uses the RL-env's linear cap (10 mm @ d=1 … 60 mm @ d=7).
    eval_start, eval_end : str
        Evaluation period (inclusive).
    seed : int
        RNG seed for initial soil moisture.
    verbose : bool
        Print progress information.
    forecast_cols : list[str] or None
        Explicit forecast column names (one per lead day).
        Only used when ``forecast_mode="columns"``.
    forecast_mode : str
        ``"perfect"``  — use observed rainfall as oracle forecast.
        ``"columns"``  — use explicit forecast columns from the CSV.
        ``"zero"``     — assume zero rainfall (worst-case / conservative).
    decision_stride_days : int
        Number of days between re-optimisations.
        Default ``1`` (full receding horizon).  Ignored when
        *block_mode* is True (overridden to ``n_days_ahead``).
    block_mode : bool
        If True, emulates the RL environment's block-scheduling:
        irrigate **only on day 0** of each block, stride = n_days_ahead,
        and the irrigation cap is scaled per the RL environment.
    reg_weight : float
        Quadratic regularisation weight on irrigation actions.

    Returns
    -------
    SimulationResult
        Same dataclass used by ``mc_irrigation_baseline``.
    """
    valid_modes = ("perfect", "columns", "zero")
    if forecast_mode not in valid_modes:
        raise ValueError(
            f"forecast_mode must be one of {valid_modes}, got '{forecast_mode}'"
        )
    # ── block mode: override stride & cap ──
    if block_mode:
        decision_stride_days = n_days_ahead
        if max_irrig_m is None:
            max_irrig_m = _rl_max_irrigation_mm(n_days_ahead) / 1000.0
    elif max_irrig_m is None:
        max_irrig_m = 0.06  # legacy default (60 mm)

    if s_ref is None:
        s_ref = (S_STAR + SFC) / 2.0

    rng = np.random.default_rng(seed)
    df = weather_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df_eval = df[
        (df["Date"] >= eval_start) & (df["Date"] <= eval_end)
    ].reset_index(drop=True)

    if len(df_eval) == 0:
        raise ValueError(f"No evaluation data in [{eval_start}, {eval_end}]")

    # Resolve forecast columns only when needed
    resolved_fc_cols: list[str] | None = None
    if forecast_mode == "columns":
        df, resolved_fc_cols = _ensure_forecast_columns(
            df, n_leads=n_days_ahead, explicit_cols=forecast_cols, verbose=verbose,
        )
        df_eval = df[
            (df["Date"] >= eval_start) & (df["Date"] <= eval_end)
        ].reset_index(drop=True)

    max_irrig_mm = max_irrig_m * 1000.0
    s_t = float(rng.uniform(S_STAR, SFC))

    if verbose:
        print(f"Evaluation data : {len(df_eval)} days ({eval_start} to {eval_end})")
        print(f"Horizon         : {n_days_ahead} days")
        print(f"Decision stride : {decision_stride_days} day(s)")
        print(f"Block mode      : {block_mode}")
        print(f"Forecast mode   : {forecast_mode}")
        print(f"Soil ref (s_ref): {s_ref:.3f}")
        print(f"Safety margin   : {safety_margin:.3f}")
        print(f"Max irrig/day   : {max_irrig_mm:.1f} mm")
        print(f"Initial s₀      : {s_t:.4f}")
        if resolved_fc_cols is not None:
            print("Forecast columns:")
            for i, c in enumerate(resolved_fc_cols, start=1):
                print(f"  Lead {i}: {c}")

    result = SimulationResult()
    day_idx = 0
    n_eval = len(df_eval)
    total_growth = LINI + LDEV + LMID + LLATE

    while day_idx < n_eval:
        horizon_len = min(n_days_ahead, n_eval - day_idx)
        apply_span = min(decision_stride_days, n_eval - day_idx, horizon_len)

        decision_row = df_eval.iloc[day_idx]

        # ---- build horizon inputs ----
        forecast_rain_mm = np.zeros(horizon_len)
        etmax_schedule = np.zeros(horizon_len)

        for d in range(horizon_len):
            row_future = df_eval.iloc[day_idx + d]
            doy = row_future["Date"].timetuple().tm_yday
            kc = kc_function(doy)
            eto = calculate_ET_o(row_future)
            etmax_schedule[d] = kc * eto

            if forecast_mode == "perfect":
                # Oracle: use observed rainfall for the future day
                forecast_rain_mm[d] = float(
                    max(row_future["Observed Rainfall (mm)"], 0.0)
                )
            elif forecast_mode == "columns":
                fc_val = float(
                    max(decision_row[resolved_fc_cols[d]], 0.0)  # type: ignore[index]
                )
                forecast_rain_mm[d] = fc_val
            # else forecast_mode == "zero" → stays 0.0

        # ---- solve MPC ----
        u_plan = mpc_solve(
            s_t=s_t,
            forecast_rain_mm=forecast_rain_mm,
            etmax_schedule=etmax_schedule,
            s_ref=s_ref,
            safety_margin=safety_margin,
            max_irrig_mm=max_irrig_mm,
            reg_weight=reg_weight,
            block_mode=block_mode,
        )

        # ---- apply first `apply_span` actions ----
        for d in range(apply_span):
            row = df_eval.iloc[day_idx + d]
            date = row["Date"]
            actual_rain = float(row["Observed Rainfall (mm)"])
            doy = date.timetuple().tm_yday
            kc = kc_function(doy)
            eto = calculate_ET_o(row)
            etmax = kc * eto
            rho_val = float(rho(s_t, etmax))

            # Block mode: irrigate only on day 0 (matching RL env)
            irr_d = float(u_plan[d]) if not block_mode or d == 0 else 0.0

            # ---- growth-stage tracking for RY ----
            adjusted_doy = (doy - SEASON_START_DOY) % 365
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
        description="Deterministic MPC irrigation baseline (quadratic tracking)"
    )
    parser.add_argument("--n-days-ahead", type=int, default=7, choices=range(1, 8))
    parser.add_argument(
        "--s-ref",
        type=float,
        default=None,
        help=(
            "Soil moisture setpoint (normalised). "
            f"Default: midpoint (S_STAR + SFC)/2 = {(S_STAR + SFC) / 2:.3f}"
        ),
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=0.02,
        help="Constraint tightening margin above S_STAR (default: 0.02)",
    )
    parser.add_argument(
        "--max-irrig",
        type=float,
        default=None,
        help=(
            "Action cap in metres per block. "
            "Default: auto-scale to match RL env (10 mm @ d=1 … 60 mm @ d=7)"
        ),
    )
    parser.add_argument(
        "--reg-weight",
        type=float,
        default=1e-4,
        help="Quadratic regularisation weight on irrigation (default: 1e-4)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data", type=str, default="daily_weather_data.csv")
    parser.add_argument("--out", type=str, default="mpc_baseline_results.csv")
    parser.add_argument(
        "--forecast-mode",
        type=str,
        default="perfect",
        choices=("perfect", "columns", "zero"),
        help=(
            "Rainfall forecast source: "
            "'perfect' = observed rain (oracle), "
            "'columns' = CSV forecast columns, "
            "'zero' = assume no rain (default: perfect)"
        ),
    )
    parser.add_argument(
        "--forecast-cols",
        type=str,
        default="",
        help=(
            "Comma-separated lead columns (lead1,lead2,...). "
            "Only used with --forecast-mode columns."
        ),
    )
    parser.add_argument(
        "--decision-stride",
        type=int,
        default=1,
        help="Re-optimisation interval in days (default: 1 = receding horizon)",
    )
    parser.add_argument(
        "--block-mode",
        action="store_true",
        default=False,
        help=(
            "Match RL env block scheduling: irrigate only on day 0, "
            "stride = n_days_ahead, auto-scale irrigation cap."
        ),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    fc_cols = (
        [c.strip() for c in args.forecast_cols.split(",") if c.strip()] or None
    )

    result = run_simulation(
        weather_df=df,
        n_days_ahead=args.n_days_ahead,
        s_ref=args.s_ref,
        safety_margin=args.safety_margin,
        max_irrig_m=args.max_irrig,
        seed=args.seed,
        verbose=True,
        forecast_cols=fc_cols,
        forecast_mode=args.forecast_mode,
        decision_stride_days=args.decision_stride,
        reg_weight=args.reg_weight,
        block_mode=args.block_mode,
    )

    # Resolve effective values for summary
    if args.block_mode:
        effective_stride = args.n_days_ahead
        effective_cap_mm = _rl_max_irrigation_mm(args.n_days_ahead)
    else:
        effective_stride = args.decision_stride
        effective_cap_mm = (args.max_irrig if args.max_irrig is not None else 0.06) * 1000.0
    s_ref_used = args.s_ref if args.s_ref is not None else (S_STAR + SFC) / 2.0

    print("\n" + "=" * 60)
    print("Deterministic MPC Baseline — Summary")
    print("=" * 60)
    print(f"  Horizon (days)            : {args.n_days_ahead}")
    print(f"  Block mode                : {args.block_mode}")
    print(f"  Forecast mode             : {args.forecast_mode}")
    print(f"  Soil reference (s_ref)    : {s_ref_used:.3f}")
    print(f"  Safety margin             : {args.safety_margin:.3f}")
    print(f"  Reg weight (λ)            : {args.reg_weight:.1e}")
    print(f"  Decision stride           : {effective_stride} day(s)")
    print(f"  Irrigation cap            : {effective_cap_mm:.1f} mm")
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

    # ── Plot (same format as RL training plots) ──
    plot_dir = str(out_path.parent)
    tag = f"det_mpc_days{args.n_days_ahead}_{args.forecast_mode}"
    from results.plots import plot_baseline_simulation
    plot_baseline_simulation(
        result=result,
        output_dir=plot_dir,
        model_name="Det-MPC",
        tag=tag,
        extra_info=(
            f"d={args.n_days_ahead}, "
            f"mode={args.forecast_mode}, "
            f"margin={args.safety_margin}"
        ),
    )


if __name__ == "__main__":
    main()

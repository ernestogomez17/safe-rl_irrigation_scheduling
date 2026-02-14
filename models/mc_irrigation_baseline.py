"""
Monte-Carlo Stochastic MPC Baseline for Irrigation Scheduling
=============================================================

A Python translation of the MATLAB chance-constrained irrigation controller
originally by Adrija.  The controller decides, once every ``n_days_ahead``
days, the minimum irrigation depth that keeps soil moisture above a
safety threshold with at least ``chance_pct`` probability across
``n_mc_samples`` Monte-Carlo rainfall realisations.

Because we do not have separate GEFS forecast NetCDF files, the controller
works with the same ``daily_weather_data.csv`` used by the RL agents:

* **Training years** (2015-2016) are used to fit a simple forecast error
  model — for each lead time *d* a linear regression maps the "forecast"
  (observed rain shifted by *d* days + noise) to the observed rain, and
  an exponential-distribution parameter ``gamma`` plus a rain-probability
  ``p`` are estimated per forecast-intensity bin.

* **Evaluation year** (2017) is simulated day-by-day: the controller looks
  up ``p`` and ``gamma`` for the current forecast value, draws MC rainfall
  samples, propagates soil moisture forward, and bisection-searches for
  the smallest irrigation satisfying the chance constraint.

The script can be run standalone:

    python mc_irrigation_baseline.py [--n-days-ahead 7] [--chance-pct 0.75]

and it will write a results CSV plus a summary to stdout, directly
comparable to the RL evaluation episodes.

Soil parameters are imported from ``water_environment.py`` so both
approaches use exactly the same physics.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Soil / crop parameters — same as WaterEnvironment
# ---------------------------------------------------------------------------
SW      = 0.3       # Wilting point
SH      = 0.2 * 0.6  # Hygroscopic point
SFC     = 0.65      # Field capacity
S_STAR  = 0.35      # Stress-onset threshold
N_SOIL  = 0.56      # Soil porosity  (n)
ZR      = 400       # Rooting depth  (mm)
KS      = 35        # Saturated hydraulic conductivity (cm/day)
BETA    = 11        # Empirical exponent

# Crop-coefficient schedule (grape)
SEASON_START_DOY = 101
LINI, LDEV, LMID, LLATE = 20, 50, 90, 20
KCINI, KCMID, KCEND = 0.4, 0.85, 0.35


# ---------------------------------------------------------------------------
# Physics helpers  (mirror WaterEnvironment methods, but pure-function style)
# ---------------------------------------------------------------------------

def kc_function(day_of_year: int) -> float:
    """FAO-56 single-crop-coefficient interpolation."""
    adjusted = (day_of_year - SEASON_START_DOY) % 365
    total_growth = LINI + LDEV + LMID + LLATE
    trans = 365 - total_growth

    if adjusted <= LINI:
        return KCINI
    elif adjusted <= LINI + LDEV:
        return KCINI + (KCMID - KCINI) * (adjusted - LINI) / LDEV
    elif adjusted <= LINI + LDEV + LMID:
        return KCMID
    elif adjusted <= total_growth:
        return KCMID - (KCMID - KCEND) * (adjusted - LINI - LDEV - LMID) / LLATE
    elif adjusted <= total_growth + trans:
        return KCEND - (KCEND - KCINI) * (adjusted - total_growth) / trans
    else:
        return KCINI


def calculate_ET_o(row: pd.Series) -> float:
    """FAO-56 Penman-Monteith reference ET (same as WaterEnvironment)."""
    Tmax = row['Daily Tmax (C)']
    Tmin = row['Daily Tmin (C)']
    DSWR = row['Daily DSWR']
    DLWR = row['Daily DLWR']
    USWR = row['Daily USWR']
    ULWR = row['Daily ULWR']
    UGRD = row['Daily UGRD']
    VGRD = row['Daily VGRD']
    Pres = row['Daily Pres (kPa)']
    date = row['Date']

    tmean = (Tmax + Tmin) / 2.0
    Rs = (DSWR + DLWR - USWR - ULWR) * 0.0864
    u2 = (np.sqrt(UGRD**2 + VGRD**2) * 4.87) / np.log(67.8 * 10 - 5.42)

    delta_val = (4098 * 0.6108 * np.exp(17.27 * tmean / (tmean + 237.3))
                 / (tmean + 237.3) ** 2)
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
    phi = np.pi * 20 / 180  # latitude 20° N
    ws = np.arccos(-np.tan(phi) * np.tan(yen))
    Ra = (24 * 60 / np.pi) * 0.082 * dr * (
        np.sin(phi) * np.sin(yen) + np.cos(phi) * np.cos(yen) * np.sin(ws))
    Rso = (0.75 + 2e-5 * 602) * Ra
    Rns = 0.77 * Rs
    Rnl = (4.903e-9
           * ((Tmax + 273.16)**4 + (Tmin + 273.16)**4) / 2
           * (0.34 - 0.14 * np.sqrt(ea))
           * (1.35 * Rs / max(Rso, 1e-6) - 0.35))
    Rn = Rns - Rnl
    Rng = 0.408 * Rn

    ET_o = DT * Rng + PT * TT * (es - ea)
    return float(ET_o)


def rho(s: float | np.ndarray, ETmax: float) -> float | np.ndarray:
    """Normalised soil-moisture loss rate (dimensionless per day).

    Accepts scalar *or* 1-D array ``s`` so we can propagate MC samples
    in a single vectorised call.
    """
    eta   = ETmax / (N_SOIL * ZR)
    eta_w = 0.15 * ETmax / (N_SOIL * ZR)
    m     = KS / (N_SOIL * ZR * (np.exp(BETA * (1 - SFC)) - 1))

    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    mask_sh  = (s > 0) & (s <= SH)
    mask_sw  = (s > SH) & (s <= SW)
    mask_ss  = (s > SW) & (s <= S_STAR)
    mask_fc  = (s > S_STAR) & (s <= SFC)
    mask_hi  = (s > SFC)

    out[mask_sh] = 0.0
    out[mask_sw] = eta_w * (s[mask_sw] - SH) / (SW - SH)
    out[mask_ss] = eta_w + (eta - eta_w) * (s[mask_ss] - SW) / (S_STAR - SW)
    out[mask_fc] = eta
    out[mask_hi] = eta + m * (np.exp(BETA * (s[mask_hi] - SFC)) - 1)

    return float(out) if out.ndim == 0 else out


def soil_update(s: float | np.ndarray, rain_mm: float | np.ndarray,
                irrig_mm: float | np.ndarray, ETmax: float) -> np.ndarray:
    """One-step soil moisture balance.  All water inputs in mm."""
    total_water = np.asarray(rain_mm, dtype=float) + np.asarray(irrig_mm, dtype=float)
    h = total_water / (N_SOIL * ZR)          # normalised water input
    ds = h - rho(s, ETmax)
    s_new = np.clip(np.asarray(s, dtype=float) + ds, 0.0, 1.0)
    return s_new


# ---------------------------------------------------------------------------
# Forecast error model  — trained on the "training" years
# ---------------------------------------------------------------------------

@dataclass
class ForecastModel:
    """Per-lead-day mapping: forecast value → (p, gamma).

    ``p``     = probability the observed rainfall is > 0.
    ``gamma`` = rate parameter of the conditional exponential distribution
                of observed rainfall given rain > 0.

    The lookup is done by binning the forecast intensity and interpolating
    the pre-computed tables, exactly replicating the MATLAB ``p_list`` /
    ``gamma_regression`` approach.
    """
    bin_edges: np.ndarray          # (n_bins,) — left edges of bins
    p_table:   np.ndarray          # (n_leads, n_bins) — prob of rain
    gamma_table: np.ndarray        # (n_leads,) — one linear regression per lead
    gamma_intercepts: np.ndarray   # (n_leads,)
    gamma_slopes: np.ndarray       # (n_leads,)

    def lookup(self, lead: int, forecast_mm: float):
        """Return (p, gamma) for a given lead day and forecast value."""
        # --- p: interpolate from the binned table ---
        idx = np.searchsorted(self.bin_edges, forecast_mm, side='right') - 1
        idx = np.clip(idx, 0, len(self.bin_edges) - 1)
        p = float(self.p_table[lead, idx])

        # --- gamma: linear model ---
        gamma = max(
            self.gamma_intercepts[lead] + self.gamma_slopes[lead] * forecast_mm,
            0.01)   # floor to avoid division-by-zero in CDF inversion
        return p, gamma


def _build_forecast_model(df_train: pd.DataFrame,
                          n_leads: int,
                          bin_step: float = 0.5,
                          noise_std: float = 2.0,
                          rng: np.random.Generator | None = None,
                          ) -> ForecastModel:
    """Fit the forecast error model from training data.

    Because we don't have real GEFS forecasts, we synthesise a "forecast"
    for lead *d* as::

        forecast_d = observed_rain[t+d] + Normal(0, noise_std)

    clipped to ≥ 0.  This gives a realistic spread of (forecast, obs) pairs
    from which we estimate the conditional distributions.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    obs = df_train['Observed Rainfall (mm)'].values.astype(float)
    n = len(obs)

    max_fc = 0.0
    pairs = []     # list of (n_leads, ...) forecast/obs arrays
    for d in range(n_leads):
        # Forecast for day t+d made at time t is the true value + noise
        fc = np.clip(obs[d:] + rng.normal(0, noise_std, n - d), 0, None)
        ob = obs[:n - d] if d > 0 else obs.copy()
        # Align: at time t the forecast is fc[t] → observation is ob[t] (= obs[t])
        # Actually for lead d: forecast made at t for day t+d = obs[t+d] + noise
        # Corresponding observation = obs[t+d]
        ob_aligned = obs[d:]
        pairs.append((fc[:len(ob_aligned)], ob_aligned))
        max_fc = max(max_fc, fc.max())

    # --- Bin edges ---
    bin_edges = np.arange(0, max_fc + bin_step, bin_step)
    n_bins = len(bin_edges)

    # --- p table (prob of rain per bin) ---
    p_table = np.zeros((n_leads, n_bins))
    for lead, (fc, ob) in enumerate(pairs):
        order = np.argsort(fc)
        fc_s, ob_s = fc[order], ob[order]
        start = 0
        for b in range(n_bins):
            edge = bin_edges[b] + bin_step
            end = np.searchsorted(fc_s, edge, side='right')
            if end <= start:
                # carry previous value
                p_table[lead, b] = p_table[lead, max(b - 1, 0)]
                continue
            seg = ob_s[start:end]
            n_pos = np.sum(seg > 0)
            p_table[lead, b] = n_pos / len(seg) if len(seg) > 0 else 0.0
            start = end

    # --- gamma (linear regression of forecast → 1/mean_obs_given_rain) ---
    gamma_intercepts = np.zeros(n_leads)
    gamma_slopes     = np.zeros(n_leads)
    for lead, (fc, ob) in enumerate(pairs):
        mask = ob > 0
        if mask.sum() < 10:
            gamma_intercepts[lead] = 1.0
            gamma_slopes[lead] = 0.0
            continue
        x, y = fc[mask], ob[mask]
        # gamma = 1 / E[rain | rain>0]  → we regress (1/y) on x?
        # The MATLAB code fits a linear model of *observed* on *forecast* for
        # rain > 0 samples, then gamma = 1 / predicted_obs.
        # Simpler and equivalent: regress obs~forecast, gamma = 1/fitted
        A = np.vstack([np.ones_like(x), x]).T
        coefs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        # gamma = n*Z / predicted_rain_mm for consistency with normalised h.
        # But the MATLAB code keeps gamma in physical rainfall space and the
        # CDF uses h = rain/(n*Z).  We'll keep gamma in 1/mm space.
        # predicted rain for a given forecast: coefs[0] + coefs[1]*fc
        # gamma = 1 / predicted_rain  (rate of the exponential).
        gamma_intercepts[lead] = coefs[0]
        gamma_slopes[lead] = coefs[1]

    return ForecastModel(
        bin_edges=bin_edges,
        p_table=p_table,
        gamma_table=np.zeros(n_leads),   # unused; kept for dataclass
        gamma_intercepts=gamma_intercepts,
        gamma_slopes=gamma_slopes,
    )


# ---------------------------------------------------------------------------
# MC sample generation
# ---------------------------------------------------------------------------

def sample_rainfall(p: float, gamma_mm_inv: float,
                    mean_obs_mm: float,
                    n_samples: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Draw ``n_samples`` rainfall realisations (mm) from the forecast CDF.

    The mixed distribution is:
        P(rain = 0)         = 1 - p
        rain | rain > 0  ~  Exponential(rate = 1/mean_obs_mm)

    Returns an array of shape ``(n_samples,)`` in **mm**.
    """
    rain = np.zeros(n_samples)
    occurs = rng.random(n_samples) < p
    n_rain = occurs.sum()
    if n_rain > 0 and mean_obs_mm > 0:
        rain[occurs] = rng.exponential(scale=mean_obs_mm, size=n_rain)
    return rain


# ---------------------------------------------------------------------------
# Bisection search for minimum irrigation
# ---------------------------------------------------------------------------

def find_minimum_irrigation(
    s_t: float,
    forecast_rain: list[tuple[float, float]],
    # list of (p_d, mean_obs_mm_d) for each of the n_days_ahead days
    etmax_schedule: list[float],
    n_mc: int,
    chance_pct: float,
    max_irrig_mm: float,
    rng: np.random.Generator,
) -> float:
    """Bisection search for the smallest irrigation (mm, applied on day 0)
    such that ≥ ``chance_pct`` fraction of MC paths keep
    s_t ≥ S_STAR on **every** day of the lookahead window.
    """
    n_days = len(forecast_rain)

    # --- Pre-sample all rainfall realisations (n_days × n_mc) ---
    rain_samples = np.zeros((n_days, n_mc))
    for d, (p_d, mean_obs_d) in enumerate(forecast_rain):
        rain_samples[d] = sample_rainfall(p_d, 0.0, mean_obs_d, n_mc, rng)

    def _fraction_safe(irrig_mm: float) -> float:
        """Fraction of MC paths where s >= S_STAR for all n_days."""
        s = np.full(n_mc, s_t)
        all_safe = np.ones(n_mc, dtype=bool)
        for d in range(n_days):
            irr_d = irrig_mm if d == 0 else 0.0
            s = soil_update(s, rain_samples[d], irr_d, etmax_schedule[d])
            all_safe &= (s >= S_STAR)
        return all_safe.mean()

    # Quick check: is zero irrigation already safe?
    if _fraction_safe(0.0) >= chance_pct:
        return 0.0

    # Quick check: is max irrigation still not enough?
    if _fraction_safe(max_irrig_mm) < chance_pct:
        return max_irrig_mm  # apply as much as we can

    # Bisection
    lo, hi = 0.0, max_irrig_mm
    for _ in range(30):          # ~30 iterations → precision < 1e-9 * max_irrig
        mid = (lo + hi) / 2.0
        if _fraction_safe(mid) >= chance_pct:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break

    return hi


# ---------------------------------------------------------------------------
# Full simulation loop
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
    violations_s_star: int = 0
    violations_sfc: int = 0
    violations_sw: int = 0
    total_irrigation_mm: float = 0.0
    total_days: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            'Date':       self.dates,
            'Soil_Moisture': self.soil_moisture,
            'Irrigation_mm': self.irrigation_mm,
            'Rain_mm':    self.rain_mm,
            'ET_o':       self.ET_o,
            'ETmax':      self.ETmax,
            'Kc':         self.Kc,
            'rho':        self.rho_vals,
        })


def run_simulation(
    weather_df: pd.DataFrame,
    n_days_ahead: int = 7,
    chance_pct: float = 0.75,
    n_mc: int = 600,
    max_irrig_m: float = 0.06,
    eval_start: str = '2017-04-10',
    eval_end:   str = '2018-04-09',
    train_start: str = '2015-01-01',
    train_end:   str = '2016-12-31',
    seed: int = 42,
    verbose: bool = True,
) -> SimulationResult:
    """Run the MC-MPC irrigation simulation and return results."""

    rng = np.random.default_rng(seed)
    df = weather_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])

    # ---- Split train / eval ----
    df_train = df[(df['Date'] >= train_start) & (df['Date'] <= train_end)].reset_index(drop=True)
    df_eval  = df[(df['Date'] >= eval_start)  & (df['Date'] <= eval_end)].reset_index(drop=True)

    if len(df_eval) == 0:
        raise ValueError(f"No evaluation data in [{eval_start}, {eval_end}]")
    if verbose:
        print(f"Training data : {len(df_train)} days  ({train_start} – {train_end})")
        print(f"Evaluation data: {len(df_eval)} days  ({eval_start} – {eval_end})")

    # ---- Fit forecast model on training data ----
    fm = _build_forecast_model(df_train, n_leads=n_days_ahead,
                               noise_std=2.0, rng=rng)

    # ---- Max irrigation per decision (mm) ----
    # WaterEnvironment action is in metres and multiplied by 1000 in
    # update_soil_moisture(), so 0.06 m = 60 mm.
    max_irrig_mm = max_irrig_m * 1000.0

    # ---- Initialise soil moisture ----
    s_t = float(rng.uniform(S_STAR, SFC))
    result = SimulationResult()

    # ---- Day-by-day simulation ----
    day_idx = 0
    n_eval = len(df_eval)

    while day_idx < n_eval:
        # How many days in this decision block?
        block_len = min(n_days_ahead, n_eval - day_idx)

        # ---- Build lookahead ----
        forecast_rain = []
        etmax_schedule = []
        for d in range(block_len):
            row = df_eval.iloc[day_idx + d]
            doy = row['Date'].timetuple().tm_yday
            kc  = kc_function(doy)
            eto = calculate_ET_o(row)
            etmax = kc * eto
            etmax_schedule.append(etmax)

            # Forecast for this day: use observed rain as the "forecast"
            fc_val = float(row['Observed Rainfall (mm)'])
            p_d, _ = fm.lookup(min(d, n_days_ahead - 1), fc_val)
            # Mean observed rain given rain > 0 — from the regression model
            pred_obs = (fm.gamma_intercepts[min(d, n_days_ahead - 1)]
                        + fm.gamma_slopes[min(d, n_days_ahead - 1)] * fc_val)
            mean_obs = max(pred_obs, 0.1)   # floor to avoid zero-scale
            forecast_rain.append((p_d, mean_obs))

        # ---- Find minimum irrigation ----
        irrig_mm = find_minimum_irrigation(
            s_t, forecast_rain, etmax_schedule,
            n_mc=n_mc, chance_pct=chance_pct,
            max_irrig_mm=max_irrig_mm, rng=rng)

        # ---- Forward-simulate the block with ACTUAL weather ----
        for d in range(block_len):
            row = df_eval.iloc[day_idx + d]
            date = row['Date']
            actual_rain = float(row['Observed Rainfall (mm)'])
            doy = date.timetuple().tm_yday
            kc  = kc_function(doy)
            eto = calculate_ET_o(row)
            etmax = kc * eto
            rho_val = float(rho(s_t, etmax))

            irr_d = irrig_mm if d == 0 else 0.0

            # Record pre-update state
            result.dates.append(date)
            result.soil_moisture.append(s_t)
            result.irrigation_mm.append(irr_d)
            result.rain_mm.append(actual_rain)
            result.ET_o.append(eto)
            result.ETmax.append(etmax)
            result.Kc.append(kc)
            result.rho_vals.append(rho_val)

            # Update soil moisture
            s_t = float(soil_update(s_t, actual_rain, irr_d, etmax))

            # Safety checks
            if s_t < S_STAR:
                result.violations_s_star += 1
            if s_t > SFC:
                result.violations_sfc += 1
            if s_t < SW:
                result.violations_sw += 1

            result.total_irrigation_mm += irr_d
            result.total_days += 1

        day_idx += block_len

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MC-MPC irrigation baseline — '
                    'comparable to RL evaluation episodes.')
    parser.add_argument('--n-days-ahead', type=int, default=7,
                        choices=range(1, 8))
    parser.add_argument('--chance-pct', type=float, default=0.75,
                        help='Fraction of MC paths that must satisfy s >= S_STAR')
    parser.add_argument('--n-mc', type=int, default=600,
                        help='Number of Monte-Carlo rainfall samples')
    parser.add_argument('--max-irrig', type=float, default=0.06,
                        help='Max irrigation per decision in metres '
                             '(matches WaterEnvironment action space; '
                             '0.06 m = 60 mm)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default='daily_weather_data.csv')
    parser.add_argument('--out', type=str, default='mc_baseline_results.csv')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    result = run_simulation(
        df,
        n_days_ahead=args.n_days_ahead,
        chance_pct=args.chance_pct,
        n_mc=args.n_mc,
        max_irrig_m=args.max_irrig,
        seed=args.seed,
        verbose=True,
    )

    # ---- Print summary ----
    print('\n' + '=' * 60)
    print('MC-MPC Irrigation Baseline — Summary')
    print('=' * 60)
    print(f'  Decision interval       : every {args.n_days_ahead} day(s)')
    print(f'  Chance constraint        : {args.chance_pct:.0%}')
    print(f'  MC samples               : {args.n_mc}')
    print(f'  Total simulated days     : {result.total_days}')
    print(f'  Total irrigation         : {result.total_irrigation_mm:.2f} mm')
    safe_days = result.total_days - result.violations_s_star
    print(f'  S_STAR violations        : {result.violations_s_star}  '
          f'({safe_days}/{result.total_days} safe = '
          f'{safe_days / max(result.total_days, 1):.1%})')
    print(f'  SFC violations           : {result.violations_sfc}')
    print(f'  SW violations            : {result.violations_sw}')

    # ---- Save CSV ----
    out_df = result.to_dataframe()
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f'\nResults saved to {out_path.resolve()}')


if __name__ == '__main__':
    main()

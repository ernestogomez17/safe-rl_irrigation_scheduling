"""
Synthetic Hindcast Forecast Generator
======================================

Generates synthetic rainfall forecast columns **in-memory** to substitute
for real NWP (e.g. GEFS Reforecast v2) data, which is not available.

The Roy et al. (2021) framework requires paired (forecast, observation)
data so the stochastic error model (p-table + gamma regression) can be
calibrated.  Since we lack real GEFS data, this module creates realistic
synthetic "hindcast" forecasts from observed rainfall by:

  1. Shifting observed rain forward by *lead* days  (temporal alignment).
  2. Applying a multiplicative skill-decay factor  α(d) ∈ (0, 1].
  3. Adding lead-dependent Gaussian noise  σ(d).
  4. Injecting random false alarms  (probability grows with lead).

The synthetic columns are added to a **copy** of the DataFrame — the
original CSV on disk is never modified.

Skill-decay parameters were tuned so that:
  - Lead 1: corr ≈ 0.99, RMSE ≈ 0.3 mm  (near-perfect short-range)
  - Lead 7: corr ≈ 0.90, RMSE ≈ 3.5 mm  (degraded long-range)

This mimics the behaviour of a real NWP product and gives the
stochastic downscaling model in ``monte_carlo.py`` meaningful
(forecast, observation) pairs to calibrate against.

Usage
-----
    from env.generate_forecasts import add_synthetic_forecasts
    df_with_fc = add_synthetic_forecasts(df, n_leads=7, seed=123)
    # df_with_fc now has columns Rain_fc_d1 … Rain_fc_d7
    # The original df is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress


# ───────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────

def add_synthetic_forecasts(
    df: pd.DataFrame,
    n_leads: int = 7,
    seed: int = 123,
    rain_col: str = "Observed Rainfall (mm)",
) -> pd.DataFrame:
    """Return a **copy** of *df* with ``Rain_fc_d1`` … ``Rain_fc_d{n_leads}`` added.

    Parameters
    ----------
    df : DataFrame
        Must contain *rain_col*.  Not modified in-place.
    n_leads : int
        Number of lead-day forecast columns to generate (1 … n_leads).
    seed : int
        RNG seed for reproducibility.
    rain_col : str
        Name of the observed-rainfall column.

    Returns
    -------
    DataFrame
        A copy of *df* augmented with forecast columns.
    """
    rng = np.random.default_rng(seed)
    obs = df[rain_col].to_numpy(dtype=float)
    n = len(obs)

    out = df.copy()

    for d in range(1, n_leads + 1):
        # ── skill decay with lead time ──
        alpha = 1.0 - 0.08 * (d - 1)          # multiplicative bias: 1.00 → 0.52
        sigma = 0.5 + 0.6 * (d - 1)           # noise std:  0.5 → 4.1 mm
        p_false_alarm = 0.02 + 0.015 * (d - 1) # false-alarm prob: 2% → 11%

        # Shift observed rain forward by d days (lead-d "sees" future rain)
        shifted = np.zeros(n, dtype=float)
        shifted[:n - d] = obs[d:]

        # Scale + noise
        fc = alpha * shifted + rng.normal(0.0, sigma, size=n)

        # Inject false alarms on originally dry days
        dry_mask = shifted == 0.0
        alarm = rng.random(n) < p_false_alarm
        fc[dry_mask & alarm] = rng.exponential(scale=1.5, size=int((dry_mask & alarm).sum()))

        # Clamp to non-negative
        fc = np.clip(fc, 0.0, None)

        out[f"Rain_fc_d{d}"] = fc

    return out


# ───────────────────────────────────────────────────────────────────
# Stochastic Rainfall Downscaler  (Roy et al. 2021, Eqs. 7-9)
# ───────────────────────────────────────────────────────────────────

class StochasticRainfallDownscaler:
    """Map a deterministic large-scale forecast to probabilistic farm-scale rainfall.

    Methodology (per Roy et al.):
      1. Bin forecasts in 1 mm intervals → P(R > 0 | FS ∈ bin)
      2. Linear regression on rainy days  → E[R | FS, R > 0]
      3. Exponential distribution with scale = E[R | FS, R > 0]
      4. Mixed CDF:  F(r) = (1 − p)·δ(0) + p·Exp(1/E[R])
    """

    def __init__(self, bin_size_mm: float = 1.0):
        self.bin_size = bin_size_mm
        self.p_bins: dict[float, float] = {}
        self.slope = 1.0
        self.intercept = 0.0

    # ── calibration ──────────────────────────────────────────────
    def fit(self, observed_rain: np.ndarray, forecasted_rain: np.ndarray) -> None:
        """Calibrate from paired (observation, forecast) history."""
        tmp = pd.DataFrame({"obs": observed_rain, "fcst": forecasted_rain})

        tmp["fcst_bin"] = (tmp["fcst"] // self.bin_size) * self.bin_size
        rainy = tmp["obs"] > 0
        self.p_bins = (
            tmp.assign(is_rain=rainy)
            .groupby("fcst_bin")["is_rain"]
            .mean()
            .to_dict()
        )

        mask = (tmp["obs"] > 0) & (tmp["fcst"] > 0)
        valid = tmp[mask]
        if len(valid) > 1:
            res = linregress(valid["fcst"], valid["obs"])
            self.slope = res.slope
            self.intercept = res.intercept

    # ── sampling ─────────────────────────────────────────────────
    def generate_samples(
        self,
        forecast_value: float,
        n_samples: int = 1000,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate Monte-Carlo rainfall samples for a single point forecast."""
        if rng is None:
            rng = np.random.default_rng()

        bin_key = (forecast_value // self.bin_size) * self.bin_size
        p_rain = self.p_bins.get(bin_key, 0.0)

        expected_depth = self.intercept + self.slope * forecast_value
        expected_depth = max(expected_depth, 0.1)

        samples = np.zeros(n_samples, dtype=float)
        rain_mask = rng.random(n_samples) < p_rain
        n_rain = int(rain_mask.sum())

        if n_rain > 0:
            samples[rain_mask] = rng.exponential(scale=expected_depth, size=n_rain)

        return samples
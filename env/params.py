"""
Shared soil and crop hyper-parameters.
======================================

All modules that need these constants should import from here so that a
single edit propagates everywhere.

Usage
-----
    from data.params import *          # quick & dirty
    from data.params import SW, SFC    # explicit

These values correspond to the default scenario (sandy-loam soil, grape
crop).  To experiment with different soils or crops, change the values
here and re-run the pipeline.
"""

# ---------------------------------------------------------------------------
# Soil parameters
# ---------------------------------------------------------------------------
SW   = 0.3          # Wilting point                         (volumetric, –)
SH   = 0.2 * 0.6   # Hygroscopic point                     (volumetric, –)
SFC  = 0.65         # Field capacity                        (volumetric, –)
S_STAR = 0.35       # Stress-onset threshold                (volumetric, –)

N    = 0.56         # Soil porosity                         (–)
ZR   = 400          # Rooting-zone depth                    (mm)
KS   = 35           # Saturated hydraulic conductivity      (cm day⁻¹)
BETA = 11           # Empirical deep-percolation exponent   (–)

# ---------------------------------------------------------------------------
# Crop parameters  (grape — FAO-56 single-crop-coefficient method)
# ---------------------------------------------------------------------------
SEASON_START_DATE = 101     # Day-of-year when the growing season begins
                            # (April 10 in a non-leap year)

LINI  = 20          # Initial growth stage         (days)
LDEV  = 50          # Development stage            (days)
LMID  = 90          # Mid-season stage             (days)
LLATE = 20          # Late-season stage            (days)

KCINI = 0.4         # Crop coefficient — initial   (–)
KCMID = 0.85        # Crop coefficient — mid       (–)
KCEND = 0.35        # Crop coefficient — end       (–)

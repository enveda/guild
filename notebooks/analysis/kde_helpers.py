"""
Shared KDE helpers for the Figure 2 / Figure 3 notebooks.

Both figures draw a density curve for a genuinely unbounded raw docking score
(``kde_curve``) and a density curve for a score bounded to [0, 1] by
construction, such as a rank percentile (``kde_curve_bounded``). Kept in one
place, imported by both notebooks, so the boundary-correction fix for Figure
2's b2 overshoot (a plain KDE puts mass outside [0, 1]) is the same function
Figure 3 uses, not a second copy that could drift.
"""

import numpy as np
from scipy.stats import gaussian_kde


def kde_curve(values, x_grid, bw="scott"):
    """Plain KDE -- correct for a genuinely unbounded score (e.g. a raw
    docking score)."""
    values = np.asarray(values).astype(float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.zeros_like(x_grid)
    return gaussian_kde(values, bw_method=bw)(x_grid)


def kde_curve_bounded(values, x_grid, lo=0.0, hi=1.0, bw="scott"):
    """KDE on a bounded support, corrected by reflection at both boundaries.

    A plain gaussian_kde puts mass outside [lo, hi], which is what Reviewer 4
    saw in Figure 2's right column. Reflecting the sample about each boundary
    and summing the three components keeps the estimate inside the support and
    integrating to 1.
    """
    v = np.asarray(values)
    v = v[~np.isnan(v.astype(float))]
    if len(v) < 2:
        return np.zeros_like(x_grid)
    kde = gaussian_kde(np.concatenate([v, lo - (v - lo), hi + (hi - v)]), bw_method=bw)
    return kde(x_grid) * 3.0

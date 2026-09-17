"""Shared KDE helpers for the Figure 2 / Figure 3 notebooks."""

import numpy as np
from scipy.stats import gaussian_kde


def kde_curve(values, x_grid, bw="scott"):
    """Plain KDE, for a genuinely unbounded score."""
    values = np.asarray(values).astype(float)
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.zeros_like(x_grid)
    return gaussian_kde(values, bw_method=bw)(x_grid)


def kde_curve_bounded(values, x_grid, lo=0.0, hi=1.0, bw="scott"):
    """KDE on [lo, hi], corrected by reflection at both boundaries so mass
    doesn't leak outside the support (a plain gaussian_kde would)."""
    v = np.asarray(values)
    v = v[~np.isnan(v.astype(float))]
    if len(v) < 2:
        return np.zeros_like(x_grid)
    kde = gaussian_kde(np.concatenate([v, lo - (v - lo), hi + (hi - v)]), bw_method=bw)
    return kde(x_grid) * 3.0

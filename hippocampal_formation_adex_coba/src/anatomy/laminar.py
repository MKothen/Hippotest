
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np


@dataclass(frozen=True)
class LaminarAxes:
    # Unit vectors in voxel-index space
    longitudinal: np.ndarray  # largest-variance axis
    transverse: np.ndarray    # middle
    radial: np.ndarray        # smallest-variance (thickness / laminar) axis
    centroid: np.ndarray      # in voxel coords


def pca_axes(points_ijk: np.ndarray) -> LaminarAxes:
    """
    PCA-based axes for a parcel, in voxel index coordinates.

    We treat:
      - longitudinal axis: largest variance component
      - radial axis: smallest variance component (thickness; used for laminar coordinate)
    """
    pts = points_ijk.astype(float)
    centroid = pts.mean(axis=0)
    X = pts - centroid
    # SVD of covariance
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    # Vt rows are principal axes
    axes = Vt
    longitudinal = axes[0] / (np.linalg.norm(axes[0]) + 1e-12)
    transverse = axes[1] / (np.linalg.norm(axes[1]) + 1e-12)
    radial = axes[2] / (np.linalg.norm(axes[2]) + 1e-12)
    return LaminarAxes(longitudinal=longitudinal, transverse=transverse, radial=radial, centroid=centroid)


def laminar_coordinate(points_ijk: np.ndarray, axes: LaminarAxes) -> np.ndarray:
    """
    Returns a normalized laminar coordinate in [0,1] given parcel PCA axes.

    The coordinate is the projection onto `axes.radial`, min-max normalized over the provided points.
    """
    X = points_ijk.astype(float) - axes.centroid[None, :]
    proj = X @ axes.radial
    lo, hi = float(proj.min()), float(proj.max())
    if hi - lo < 1e-9:
        return np.zeros_like(proj)
    return (proj - lo) / (hi - lo)


def longitudinal_coordinate(points_ijk: np.ndarray, axes: LaminarAxes) -> np.ndarray:
    X = points_ijk.astype(float) - axes.centroid[None, :]
    proj = X @ axes.longitudinal
    lo, hi = float(proj.min()), float(proj.max())
    if hi - lo < 1e-9:
        return np.zeros_like(proj)
    return (proj - lo) / (hi - lo)

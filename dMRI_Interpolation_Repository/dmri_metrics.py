"""Error metrics translated from the MATLAB helper functions."""

from __future__ import annotations

import numpy as np


def absolute_relative_error(estimate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Elementwise absolute relative error with safe reference-zero handling."""

    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != reference.shape:
        raise ValueError("Estimate and reference arrays must have equal shapes")
    error = np.zeros_like(reference)
    nonzero = np.abs(reference) > np.finfo(np.float64).eps
    error[nonzero] = np.abs(reference[nonzero] - estimate[nonzero]) / np.abs(
        reference[nonzero]
    )
    error[~np.isfinite(error)] = 0.0
    return error


def checkerboard_selection(shape: tuple[int, int]) -> np.ndarray:
    """Sampling mask used by the original MATLAB error evaluation."""

    rows, cols = shape
    selection = np.zeros((rows, cols), dtype=bool)
    selection[1::2, :] = True
    selection[0::2, 1::2] = True
    return selection


def frobenius_map(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Voxelwise Frobenius distance for tensor fields ending in 3-by-3."""

    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("Tensor fields must have equal shapes")
    if reference.ndim != 4 or reference.shape[-2:] != (3, 3):
        raise ValueError("Inputs must have shape (rows, columns, 3, 3)")
    error_map = np.linalg.norm(reference - estimate, axis=(-2, -1))
    return error_map, error_map[checkerboard_selection(error_map.shape)]


def tensor_coefficient_distance(
    reference: np.ndarray, estimate: np.ndarray, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Distance for channel-last symmetric tensor coefficient fields."""

    expected = {2: 6, 4: 15, 6: 28}
    if order not in expected:
        raise ValueError("Supported tensor orders are 2, 4, and 6")
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError("Tensor coefficient fields must have equal shapes")
    if reference.ndim != 3 or reference.shape[-1] != expected[order]:
        raise ValueError(
            f"Order {order} requires shape (rows, columns, {expected[order]})"
        )
    error_map = np.linalg.norm(reference - estimate, axis=-1)
    return error_map, error_map[checkerboard_selection(error_map.shape)]


def summarize_error(error_values: np.ndarray) -> dict[str, float]:
    """Return finite mean, standard deviation, RMSE, and maximum."""

    values = np.asarray(error_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "std": np.nan, "rmse": np.nan, "max": np.nan}
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(values**2))),
        "max": float(values.max()),
    }

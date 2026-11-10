"""Utilities for symmetric tensor fields stored in the MATLAB coefficient order."""

from __future__ import annotations

import math
from itertools import combinations_with_replacement, permutations
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SUPPORTED_ORDERS = (2, 4, 6)


def coefficient_count(order: int) -> int:
    """Number of independent coefficients of a symmetric 3-D order-K tensor."""

    _validate_order(order)
    return math.comb(order + 2, 2)


def symmetric_indices(order: int) -> tuple[tuple[int, ...], ...]:
    """Return the coefficient index order used by the original MATLAB demos.

    Indices run from ``(2, ..., 2)`` (MATLAB component 3) down to
    ``(0, ..., 0)`` (MATLAB component 1). This yields 6, 15, and 28 entries
    for tensor orders 2, 4, and 6, respectively.
    """

    _validate_order(order)
    return tuple(reversed(tuple(combinations_with_replacement(range(3), order))))


def load_tensor_field(path: str | Path, variable: str = "training_data") -> np.ndarray:
    """Load a channel-first tensor field from a MATLAB ``.mat`` file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = loadmat(path)
    if variable not in payload:
        available = sorted(key for key in payload if not key.startswith("__"))
        raise KeyError(f"{variable!r} not found in {path.name}; available={available}")
    field = np.asarray(payload[variable], dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("tensor data must have shape (coefficients, rows, columns)")
    return field


def infer_order(coefficients: np.ndarray) -> int:
    """Infer tensor order from a channel-first coefficient field."""

    count = np.asarray(coefficients).shape[0]
    mapping = {coefficient_count(order): order for order in SUPPORTED_ORDERS}
    if count not in mapping:
        raise ValueError(f"unsupported coefficient count: {count}")
    return mapping[count]


def coefficients_to_full(coefficients: np.ndarray, order: int | None = None) -> np.ndarray:
    """Expand independent symmetric coefficients into full tensor arrays.

    Input layout is ``(coefficients, rows, columns)``. Output layout is
    ``(rows, columns, 3, ..., 3)`` with ``order`` trailing tensor modes.
    """

    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 3:
        raise ValueError("coefficients must have shape (channels, rows, columns)")
    order = infer_order(coefficients) if order is None else int(order)
    expected = coefficient_count(order)
    if coefficients.shape[0] != expected:
        raise ValueError(f"order {order} requires {expected} coefficients")

    rows, cols = coefficients.shape[1:]
    full = np.zeros((rows, cols) + (3,) * order, dtype=np.float64)
    for channel, index in enumerate(symmetric_indices(order)):
        for permuted in set(permutations(index)):
            full[(slice(None), slice(None)) + permuted] = coefficients[channel]
    return full


def full_to_coefficients(tensors: np.ndarray, order: int) -> np.ndarray:
    """Compress full symmetric tensors to the original channel-first order."""

    tensors = np.asarray(tensors, dtype=np.float64)
    expected_shape = (3,) * order
    if tensors.ndim != order + 2 or tensors.shape[-order:] != expected_shape:
        raise ValueError(
            f"tensors must have shape (rows, columns, {', '.join('3' for _ in range(order))})"
        )
    channels = [
        tensors[(slice(None), slice(None)) + index]
        for index in symmetric_indices(order)
    ]
    return np.stack(channels, axis=0)


def project_spd(matrices: np.ndarray, minimum_eigenvalue: float = 1e-8) -> np.ndarray:
    """Project symmetric 3-by-3 matrices to the positive-definite cone."""

    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.shape[-2:] != (3, 3):
        raise ValueError("matrices must end in shape (3, 3)")
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    eigenvalues = np.maximum(eigenvalues, float(minimum_eigenvalue))
    return np.einsum("...ik,...k,...jk->...ij", eigenvectors, eigenvalues, eigenvectors)


def frobenius_error(
    estimate: np.ndarray,
    reference: np.ndarray,
    order: int | None = None,
    mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Compute voxelwise Frobenius distance and summary statistics."""

    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape")
    order = infer_order(reference) if order is None else int(order)
    estimate_full = coefficients_to_full(estimate, order)
    reference_full = coefficients_to_full(reference, order)
    axes = tuple(range(2, 2 + order))
    error_map = np.sqrt(np.sum((estimate_full - reference_full) ** 2, axis=axes))
    values = error_map.ravel() if mask is None else error_map[np.asarray(mask, dtype=bool)]
    return {
        "map": error_map,
        "values": values,
        "mean": float(np.mean(values)) if values.size else float("nan"),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    }


def unsampled_mask(shape: tuple[int, int], factor: int = 2) -> np.ndarray:
    """Mask target positions that were not retained during regular subsampling."""

    mask = np.ones(shape, dtype=bool)
    mask[::factor, ::factor] = False
    return mask


def _validate_order(order: int) -> None:
    if int(order) not in SUPPORTED_ORDERS:
        raise ValueError(f"order must be one of {SUPPORTED_ORDERS}")


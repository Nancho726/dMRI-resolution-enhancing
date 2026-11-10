"""Second-order DTI and FA estimation translated from MATLAB."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


COEFFICIENT_ORDER = ("Dxx", "Dxy", "Dxz", "Dyy", "Dyz", "Dzz")


def dti_design_matrix(bvals: np.ndarray, bvecs: np.ndarray) -> np.ndarray:
    """Build the linearized Stejskal-Tanner design matrix."""

    bvals = np.asarray(bvals, dtype=np.float64).reshape(-1)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    if bvecs.shape != (bvals.size, 3):
        raise ValueError("bvals and bvecs have incompatible dimensions")
    gx, gy, gz = bvecs.T
    design = np.column_stack(
        (
            np.ones_like(bvals),
            -bvals * gx**2,
            -2.0 * bvals * gx * gy,
            -2.0 * bvals * gx * gz,
            -bvals * gy**2,
            -2.0 * bvals * gy * gz,
            -bvals * gz**2,
        )
    )
    if np.linalg.matrix_rank(design) < 7:
        raise ValueError("The gradient table does not provide a full-rank DTI design")
    return design


def fit_dti(
    data: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    slice_indices: Iterable[int] | None = None,
    signal_floor: float = 1.0,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    """Fit second-order diffusion tensors for selected zero-based slices.

    Coefficients use the channel-last layout ``(x, y, selected_z, 6)``.
    """

    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError("data must have shape (x, y, z, gradients)")
    if signal_floor <= 0:
        raise ValueError("signal_floor must be positive")
    if slice_indices is None:
        selected = np.arange(data.shape[2], dtype=int)
    else:
        selected = np.asarray(list(slice_indices), dtype=int)
    if selected.size == 0 or np.any(selected < 0) or np.any(selected >= data.shape[2]):
        raise IndexError("slice_indices contains an index outside the dMRI volume")

    design = dti_design_matrix(bvals, bvecs)
    pseudo_inverse = np.linalg.pinv(design)
    x_size, y_size = data.shape[:2]
    coefficients = np.zeros((x_size, y_size, selected.size, 6), dtype=np.float32)
    s0 = np.zeros((x_size, y_size, selected.size), dtype=np.float32)

    for output_index, source_slice in enumerate(selected):
        signal = np.asarray(data[:, :, source_slice, :], dtype=np.float64)
        flat_signal = signal.reshape(-1, data.shape[3]).T
        valid = np.any(flat_signal > 0, axis=0)
        fitted = pseudo_inverse @ np.log(np.maximum(flat_signal, signal_floor))
        fitted[~np.isfinite(fitted)] = 0.0
        fitted[:, ~valid] = 0.0
        s0[:, :, output_index] = np.exp(fitted[0]).reshape(x_size, y_size)
        coefficients[:, :, output_index, :] = fitted[1:].T.reshape(
            x_size, y_size, 6
        )

    return {
        "coefficients": coefficients,
        "s0": s0,
        "slice_indices": selected,
        "coefficient_order": COEFFICIENT_ORDER,
    }


def coefficients_to_tensors(coefficients: np.ndarray) -> np.ndarray:
    """Convert channel-last DTI coefficients into symmetric 3-by-3 matrices."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape[-1] != 6:
        raise ValueError("The final coefficient dimension must have length 6")
    tensors = np.zeros(coefficients.shape[:-1] + (3, 3), dtype=np.float64)
    dxx, dxy, dxz, dyy, dyz, dzz = np.moveaxis(coefficients, -1, 0)
    tensors[..., 0, 0] = dxx
    tensors[..., 0, 1] = tensors[..., 1, 0] = dxy
    tensors[..., 0, 2] = tensors[..., 2, 0] = dxz
    tensors[..., 1, 1] = dyy
    tensors[..., 1, 2] = tensors[..., 2, 1] = dyz
    tensors[..., 2, 2] = dzz
    return tensors


def compute_fa(coefficients: np.ndarray) -> dict[str, np.ndarray]:
    """Compute FA, mean diffusivity, eigenvalues, and principal directions."""

    tensors = coefficients_to_tensors(coefficients)
    tensors = 0.5 * (tensors + np.swapaxes(tensors, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(tensors)
    eigenvalues = np.maximum(eigenvalues, np.finfo(np.float64).eps)
    mean_diffusivity = eigenvalues.mean(axis=-1)
    numerator = 1.5 * np.sum(
        (eigenvalues - mean_diffusivity[..., None]) ** 2, axis=-1
    )
    denominator = np.sum(eigenvalues**2, axis=-1)
    fa = np.sqrt(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0))
    fa = np.clip(fa, 0.0, 1.0).astype(np.float32)
    principal_direction = eigenvectors[..., :, -1].astype(np.float32)

    invalid = ~np.any(np.asarray(coefficients) != 0, axis=-1)
    fa[invalid] = 0.0
    mean_diffusivity = mean_diffusivity.astype(np.float32)
    mean_diffusivity[invalid] = 0.0
    principal_direction[invalid] = 0.0

    return {
        "fa": fa,
        "mean_diffusivity": mean_diffusivity,
        "eigenvalues": eigenvalues.astype(np.float32),
        "principal_direction": principal_direction,
        "tensors": tensors.astype(np.float32),
    }

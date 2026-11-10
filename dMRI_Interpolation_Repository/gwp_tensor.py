"""Generalized Wishart process interpolation for second-order tensor fields.

The field is represented as

    T(x) = sum_i L u_i(x) u_i(x)^T L^T,

where every component of ``u_i`` follows a shared RBF Gaussian process. The
implementation uses the conditional GP mean, preserves observed tensors, and
projects numerical round-off back to the SPD cone.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from tensor_utils import (
    coefficients_to_full,
    full_to_coefficients,
    project_spd,
)


def rbf_covariance(
    first: np.ndarray,
    second: np.ndarray,
    length_scale: float = 2.0,
    variance: float = 1.0,
) -> np.ndarray:
    """Squared-exponential covariance between two coordinate sets."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != second.shape[1]:
        raise ValueError("coordinate arrays must be two-dimensional and compatible")
    if length_scale <= 0 or variance <= 0:
        raise ValueError("length_scale and variance must be positive")
    squared = np.sum((first[:, None, :] - second[None, :, :]) ** 2, axis=-1)
    return variance * np.exp(-0.5 * squared / (length_scale**2))


def _grid_coordinates(shape: tuple[int, int], spacing: int = 1) -> np.ndarray:
    rows = np.arange(shape[0], dtype=np.float64) * spacing
    cols = np.arange(shape[1], dtype=np.float64) * spacing
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.column_stack((rr.ravel(), cc.ravel()))


def _latent_factors(
    tensors: np.ndarray,
    scale_cholesky: np.ndarray,
    degrees_of_freedom: int,
) -> np.ndarray:
    """Construct GWP-compatible latent factors from observed SPD tensors."""

    tensors = project_spd(tensors)
    flat = tensors.reshape(-1, 3, 3)
    latent = np.zeros((flat.shape[0], degrees_of_freedom, 3), dtype=np.float64)
    for index, tensor in enumerate(flat):
        tensor_cholesky = np.linalg.cholesky(tensor)
        factors = np.linalg.solve(scale_cholesky, tensor_cholesky)
        used = min(3, degrees_of_freedom)
        latent[index, :used, :] = factors[:, :used].T
    return latent


def _reconstruct(
    latent: np.ndarray,
    scale_cholesky: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    transformed = np.einsum("ab,nkb->nka", scale_cholesky, latent)
    tensors = np.einsum("nki,nkj->nij", transformed, transformed)
    return project_spd(tensors).reshape(target_shape + (3, 3))


def gwp_interpolate(
    low_resolution_coefficients: np.ndarray,
    target_shape: tuple[int, int],
    factor: int = 2,
    degrees_of_freedom: int = 4,
    length_scale: float = 2.0,
    kernel_variance: float = 1.0,
    noise_variance: float = 1e-6,
    scale_matrix: np.ndarray | None = None,
    preserve_samples: bool = True,
) -> dict[str, np.ndarray]:
    """Interpolate a low-resolution second-order tensor field with a GWP.

    Parameters use the tensor-domain coefficient layout ``(6, rows, columns)``
    and the MATLAB component order ``T33, T23, T22, T13, T12, T11``.
    """

    coefficients = np.asarray(low_resolution_coefficients, dtype=np.float64)
    if coefficients.ndim != 3 or coefficients.shape[0] != 6:
        raise ValueError("low_resolution_coefficients must have shape (6, rows, columns)")
    if degrees_of_freedom < 3:
        raise ValueError("degrees_of_freedom must be at least 3")
    if noise_variance <= 0:
        raise ValueError("noise_variance must be positive")

    observed = project_spd(coefficients_to_full(coefficients, order=2))
    if scale_matrix is None:
        scale_matrix = np.mean(observed.reshape(-1, 3, 3), axis=0) / degrees_of_freedom
    scale_matrix = project_spd(np.asarray(scale_matrix, dtype=np.float64))
    scale_cholesky = np.linalg.cholesky(scale_matrix)

    x_train = _grid_coordinates(observed.shape[:2], spacing=factor)
    x_target = _grid_coordinates(tuple(map(int, target_shape)), spacing=1)
    latent_train = _latent_factors(observed, scale_cholesky, degrees_of_freedom)
    responses = latent_train.reshape(x_train.shape[0], -1)
    response_mean = responses.mean(axis=0, keepdims=True)

    covariance = rbf_covariance(
        x_train, x_train, length_scale=length_scale, variance=kernel_variance
    )
    covariance.flat[:: covariance.shape[0] + 1] += noise_variance
    factorized = cho_factor(covariance, lower=True, check_finite=False)
    alpha = cho_solve(factorized, responses - response_mean, check_finite=False)
    cross_covariance = rbf_covariance(
        x_target, x_train, length_scale=length_scale, variance=kernel_variance
    )
    latent_target = cross_covariance @ alpha + response_mean
    latent_target = latent_target.reshape(-1, degrees_of_freedom, 3)
    reconstructed = _reconstruct(latent_target, scale_cholesky, tuple(target_shape))

    output_coefficients = full_to_coefficients(reconstructed, order=2)
    if preserve_samples:
        source_rows = min(observed.shape[0], (target_shape[0] - 1) // factor + 1)
        source_cols = min(observed.shape[1], (target_shape[1] - 1) // factor + 1)
        output_coefficients[
            :,
            0 : source_rows * factor : factor,
            0 : source_cols * factor : factor,
        ] = coefficients[:, :source_rows, :source_cols]
        reconstructed = coefficients_to_full(output_coefficients, order=2)

    return {
        "coefficients": output_coefficients,
        "tensors": reconstructed,
        "latent_train": latent_train,
        "latent_target": latent_target.reshape(tuple(target_shape) + (degrees_of_freedom, 3)),
        "scale_matrix": scale_matrix,
        "training_coordinates": x_train,
        "target_coordinates": x_target,
    }


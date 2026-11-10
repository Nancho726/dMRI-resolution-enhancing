"""Canonical decomposition process interpolation for tensor fields.

The implementation estimates a shared canonical-polyadic (CP/CANDECOMP)
basis and models its spatially varying component weights with RBF Gaussian
processes. It accepts second-, fourth-, and sixth-order tensor fields directly;
raw dMRI is intentionally not accepted by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from gwp_tensor import rbf_covariance
from tensor_utils import coefficients_to_full, full_to_coefficients, infer_order


FAST_RANKS = {2: 3, 4: 6, 6: 8}
MATLAB_LEGACY_RANKS = {2: 10, 4: 12, 6: 16}


@dataclass(frozen=True)
class CPDecomposition:
    weights: np.ndarray
    factors: tuple[np.ndarray, ...]
    relative_error: float
    iterations: int


def _component_tensor(factors: list[np.ndarray], component: int) -> np.ndarray:
    result = factors[0][:, component]
    for factor in factors[1:]:
        result = np.multiply.outer(result, factor[:, component])
    return result


def cp_to_tensor(weights: np.ndarray, factors: tuple[np.ndarray, ...] | list[np.ndarray]) -> np.ndarray:
    """Reconstruct a dense tensor from CP weights and factor matrices."""

    weights = np.asarray(weights, dtype=np.float64)
    factors = [np.asarray(factor, dtype=np.float64) for factor in factors]
    shape = tuple(factor.shape[0] for factor in factors)
    output = np.zeros(shape, dtype=np.float64)
    for component, weight in enumerate(weights):
        output += weight * _component_tensor(factors, component)
    return output


def _design_for_mode(
    factors: list[np.ndarray],
    weights: np.ndarray,
    mode: int,
) -> np.ndarray:
    other_modes = [index for index in range(len(factors)) if index != mode]
    other_shape = tuple(factors[index].shape[0] for index in other_modes)
    design = np.empty((int(np.prod(other_shape)), weights.size), dtype=np.float64)
    for row, multi_index in enumerate(np.ndindex(other_shape)):
        values = weights.copy()
        for position, factor_index in enumerate(other_modes):
            values *= factors[factor_index][multi_index[position], :]
        design[row] = values
    return design


def cp_als(
    tensor: np.ndarray,
    rank: int,
    max_iterations: int = 100,
    tolerance: float = 1e-7,
    regularization: float = 1e-8,
    random_state: int = 1,
) -> CPDecomposition:
    """Small dense CP-ALS solver specialized for 3-by-...-by-3 tensors."""

    tensor = np.asarray(tensor, dtype=np.float64)
    if tensor.ndim < 2 or any(size != 3 for size in tensor.shape):
        raise ValueError("tensor must have shape (3,)*order")
    if rank < 1:
        raise ValueError("rank must be positive")
    rng = np.random.default_rng(random_state)
    factors = [rng.normal(size=(size, rank)) for size in tensor.shape]
    weights = np.ones(rank, dtype=np.float64)
    for mode in range(tensor.ndim):
        norms = np.linalg.norm(factors[mode], axis=0)
        norms[norms == 0] = 1.0
        factors[mode] /= norms
        weights *= norms

    tensor_norm = max(np.linalg.norm(tensor), np.finfo(np.float64).eps)
    previous_error = np.inf
    relative_error = previous_error
    for iteration in range(1, max_iterations + 1):
        for mode in range(tensor.ndim):
            design = _design_for_mode(factors, weights, mode)
            gram = design.T @ design + regularization * np.eye(rank)
            right_side = design.T
            solved_rows = []
            for coordinate in range(tensor.shape[mode]):
                target_slice = np.take(tensor, coordinate, axis=mode).reshape(-1)
                solved_rows.append(np.linalg.solve(gram, right_side @ target_slice))
            updated = np.vstack(solved_rows)
            norms = np.linalg.norm(updated, axis=0)
            norms[norms < np.finfo(np.float64).eps] = 1.0
            factors[mode] = updated / norms
            weights *= norms

        approximation = cp_to_tensor(weights, factors)
        relative_error = float(np.linalg.norm(tensor - approximation) / tensor_norm)
        if np.isfinite(previous_error) and abs(previous_error - relative_error) <= (
            tolerance * max(1.0, previous_error)
        ):
            break
        previous_error = relative_error

    return CPDecomposition(
        weights=weights,
        factors=tuple(factors),
        relative_error=relative_error,
        iterations=iteration,
    )


def _basis_matrix(decomposition: CPDecomposition) -> np.ndarray:
    atoms = []
    for component, weight in enumerate(decomposition.weights):
        atoms.append((weight * _component_tensor(list(decomposition.factors), component)).reshape(-1))
    return np.column_stack(atoms)


def _grid_coordinates(shape: tuple[int, int], spacing: int) -> np.ndarray:
    rows = np.arange(shape[0], dtype=np.float64) * spacing
    cols = np.arange(shape[1], dtype=np.float64) * spacing
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.column_stack((rr.ravel(), cc.ravel()))


def cdp_interpolate(
    low_resolution_coefficients: np.ndarray,
    target_shape: tuple[int, int],
    order: int | None = None,
    factor: int = 2,
    rank: int | None = None,
    length_scale: float = 2.0,
    kernel_variance: float = 1.0,
    noise_variance: float = 1e-6,
    cp_iterations: int = 80,
    cp_tolerance: float = 1e-7,
    ridge: float = 1e-7,
    random_state: int = 1,
    preserve_samples: bool = True,
) -> dict[str, np.ndarray | float | int | CPDecomposition]:
    """Interpolate a tensor field with a canonical decomposition process."""

    coefficients = np.asarray(low_resolution_coefficients, dtype=np.float64)
    if coefficients.ndim != 3:
        raise ValueError("coefficients must have shape (channels, rows, columns)")
    order = infer_order(coefficients) if order is None else int(order)
    rank = FAST_RANKS[order] if rank is None else int(rank)
    if rank < 1:
        raise ValueError("rank must be positive")

    observed_full = coefficients_to_full(coefficients, order=order)
    mean_tensor = observed_full.reshape((-1,) + (3,) * order).mean(axis=0)
    decomposition = cp_als(
        mean_tensor,
        rank=rank,
        max_iterations=cp_iterations,
        tolerance=cp_tolerance,
        random_state=random_state,
    )
    basis = _basis_matrix(decomposition)
    flattened_observations = observed_full.reshape(-1, 3**order)
    gram = basis.T @ basis + ridge * np.eye(rank)
    spatial_weights = np.linalg.solve(
        gram, basis.T @ flattened_observations.T
    ).T

    x_train = _grid_coordinates(observed_full.shape[:2], spacing=factor)
    x_target = _grid_coordinates(tuple(map(int, target_shape)), spacing=1)
    weight_mean = spatial_weights.mean(axis=0, keepdims=True)
    covariance = rbf_covariance(
        x_train, x_train, length_scale=length_scale, variance=kernel_variance
    )
    covariance.flat[:: covariance.shape[0] + 1] += noise_variance
    factorized = cho_factor(covariance, lower=True, check_finite=False)
    alpha = cho_solve(factorized, spatial_weights - weight_mean, check_finite=False)
    cross_covariance = rbf_covariance(
        x_target, x_train, length_scale=length_scale, variance=kernel_variance
    )
    target_weights = cross_covariance @ alpha + weight_mean
    target_full = (target_weights @ basis.T).reshape(tuple(target_shape) + (3,) * order)
    target_coefficients = full_to_coefficients(target_full, order=order)

    if preserve_samples:
        source_rows = min(coefficients.shape[1], (target_shape[0] - 1) // factor + 1)
        source_cols = min(coefficients.shape[2], (target_shape[1] - 1) // factor + 1)
        target_coefficients[
            :,
            0 : source_rows * factor : factor,
            0 : source_cols * factor : factor,
        ] = coefficients[:, :source_rows, :source_cols]
        target_full = coefficients_to_full(target_coefficients, order=order)

    return {
        "coefficients": target_coefficients,
        "tensors": target_full,
        "spatial_weights_train": spatial_weights,
        "spatial_weights_target": target_weights.reshape(tuple(target_shape) + (rank,)),
        "decomposition": decomposition,
        "cp_relative_error": decomposition.relative_error,
        "cp_iterations": decomposition.iterations,
        "rank": rank,
        "order": order,
        "training_coordinates": x_train,
        "target_coordinates": x_target,
    }


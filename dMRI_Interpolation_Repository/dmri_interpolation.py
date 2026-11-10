"""Nearest-neighbor and bilinear interpolation for raw dMRI volumes.

These functions operate on the diffusion-weighted signal itself. Tensor-domain
methods (GWP and CDP) are implemented separately in ``gwp_tensor.py`` and
``cdp_tensor.py``.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import map_coordinates


SUPPORTED_METHODS = {"nearest": 0, "bilinear": 1}


def downsample_in_plane(
    data: np.ndarray,
    factor: int = 2,
    offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Subsample the first two spatial axes of a dMRI array.

    The input may be a 2-D image, a 3-D stack, or a 4-D dMRI volume with
    layout ``(x, y, z, gradients)``. No averaging is performed: this mirrors
    the MATLAB expression ``image(1:factor:end, 1:factor:end)``.
    """

    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError("data must have at least two spatial dimensions")
    if not isinstance(factor, (int, np.integer)) or factor < 2:
        raise ValueError("factor must be an integer greater than or equal to 2")
    row_offset, col_offset = map(int, offset)
    if not 0 <= row_offset < factor or not 0 <= col_offset < factor:
        raise ValueError("each offset must satisfy 0 <= offset < factor")
    return array[row_offset::factor, col_offset::factor, ...]


def _interpolate_plane(
    low_resolution: np.ndarray,
    target_shape: tuple[int, int],
    factor: int,
    order: int,
) -> np.ndarray:
    """Interpolate one 2-D plane on the original sample-aligned grid."""

    low_resolution = np.asarray(low_resolution, dtype=np.float64)
    if low_resolution.ndim != 2:
        raise ValueError("low_resolution must be a two-dimensional image")
    rows, cols = map(int, target_shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("target_shape entries must be positive")

    # Coordinates are expressed in low-resolution index units. Therefore,
    # output pixels at multiples of ``factor`` coincide exactly with samples.
    row_coordinates = np.arange(rows, dtype=np.float64) / factor
    col_coordinates = np.arange(cols, dtype=np.float64) / factor
    rr, cc = np.meshgrid(row_coordinates, col_coordinates, indexing="ij")
    result = map_coordinates(
        low_resolution,
        (rr, cc),
        order=order,
        mode="nearest",
        prefilter=False,
    )
    return result


def interpolate_in_plane(
    low_resolution: np.ndarray,
    target_shape: tuple[int, int],
    factor: int = 2,
    method: str = "bilinear",
    preserve_samples: bool = True,
) -> np.ndarray:
    """Interpolate the first two axes of an image or dMRI array.

    Parameters
    ----------
    low_resolution:
        Array with layout ``(x_low, y_low, ...)``.
    target_shape:
        Requested high-resolution in-plane size ``(x_high, y_high)``.
    factor:
        Integer spacing between retained samples on the target grid.
    method:
        ``"nearest"`` or ``"bilinear"``.
    preserve_samples:
        If true, place every observed low-resolution value back at its exact
        target-grid coordinate after interpolation.
    """

    array = np.asarray(low_resolution)
    if array.ndim < 2:
        raise ValueError("low_resolution must have at least two dimensions")
    method = method.lower().strip()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)}")
    if factor < 2:
        raise ValueError("factor must be at least 2")

    trailing_shape = array.shape[2:]
    flat = array.reshape(array.shape[0], array.shape[1], -1)
    output = np.empty(tuple(target_shape) + (flat.shape[2],), dtype=np.float64)
    for channel in range(flat.shape[2]):
        output[..., channel] = _interpolate_plane(
            flat[..., channel], target_shape, factor, SUPPORTED_METHODS[method]
        )

    if preserve_samples:
        source_rows = min(array.shape[0], (target_shape[0] - 1) // factor + 1)
        source_cols = min(array.shape[1], (target_shape[1] - 1) // factor + 1)
        output[
            0 : source_rows * factor : factor,
            0 : source_cols * factor : factor,
            :,
        ] = flat[:source_rows, :source_cols, :]

    output = output.reshape(tuple(target_shape) + trailing_shape)
    if np.issubdtype(array.dtype, np.floating):
        return output.astype(array.dtype, copy=False)
    return output.astype(np.float32)


def reconstruct_dmri(
    data: np.ndarray,
    factor: int = 2,
    methods: Iterable[str] = ("nearest", "bilinear"),
    slice_indices: Iterable[int] | None = None,
    gradient_indices: Iterable[int] | None = None,
) -> dict[str, np.ndarray]:
    """Downsample and reconstruct selected raw-dMRI slices and gradients.

    The input must have layout ``(x, y, z, gradients)``. Returned arrays have
    layout ``(x, y, selected_z, selected_gradients)``.
    """

    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError("data must have shape (x, y, z, gradients)")
    slices = (
        np.arange(data.shape[2], dtype=int)
        if slice_indices is None
        else np.asarray(list(slice_indices), dtype=int)
    )
    gradients = (
        np.arange(data.shape[3], dtype=int)
        if gradient_indices is None
        else np.asarray(list(gradient_indices), dtype=int)
    )
    if slices.size == 0 or np.any((slices < 0) | (slices >= data.shape[2])):
        raise IndexError("slice_indices contains an invalid index")
    if gradients.size == 0 or np.any((gradients < 0) | (gradients >= data.shape[3])):
        raise IndexError("gradient_indices contains an invalid index")

    selected = data[:, :, slices, :][:, :, :, gradients]
    low_resolution = downsample_in_plane(selected, factor=factor)
    result: dict[str, np.ndarray] = {
        "reference": selected,
        "low_resolution": low_resolution,
        "slice_indices": slices,
        "gradient_indices": gradients,
    }
    for method in methods:
        normalized = method.lower().strip()
        result[normalized] = interpolate_in_plane(
            low_resolution,
            data.shape[:2],
            factor=factor,
            method=normalized,
        )
    return result


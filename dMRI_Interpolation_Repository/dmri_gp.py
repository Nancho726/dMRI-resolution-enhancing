"""Patch-based Gaussian-process reconstruction for raw dMRI."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
try:
    from tqdm.auto import tqdm
except ImportError:  # Progress bars are optional outside Colab.
    def tqdm(iterable, **_kwargs):
        return iterable


def patch_starts(image_length: int, patch_size: int, step: int) -> list[int]:
    """Return zero-based patch starts and force inclusion of the final patch."""

    if patch_size > image_length:
        raise ValueError("patch_size exceeds the image dimension")
    starts = list(range(0, image_length - patch_size + 1, step))
    last = image_length - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _resize_baseline(image: np.ndarray, target_shape: tuple[int, int], order: int) -> np.ndarray:
    from dmri_interpolation import interpolate_in_plane

    factor = max(2, int(round(target_shape[0] / image.shape[0])))
    method = "nearest" if order == 0 else "bilinear"
    return interpolate_in_plane(
        image,
        target_shape=target_shape,
        factor=factor,
        method=method,
        preserve_samples=True,
    )


def patch_gp_upsample(
    low_resolution: np.ndarray,
    target_shape: tuple[int, int],
    patch_size: int = 10,
    factor: int = 2,
    optimize_kernel: bool = True,
    show_progress: bool = True,
) -> np.ndarray:
    """Reconstruct one image with overlapping local exact GP models.

    The squared-exponential covariance and local maximum-likelihood fitting
    replace the legacy MATLAB GP toolbox. Overlapping predictions are averaged.
    """

    low_resolution = np.asarray(low_resolution, dtype=np.float64)
    if low_resolution.ndim != 2:
        raise ValueError("low_resolution must be a two-dimensional image")
    if factor < 2 or patch_size < 2:
        raise ValueError("factor and patch_size must be at least 2")

    step = patch_size - 1
    row_starts = patch_starts(low_resolution.shape[0], patch_size, step)
    col_starts = patch_starts(low_resolution.shape[1], patch_size, step)
    output_span = factor * (patch_size - 1) + 1

    train_axis = np.arange(0, factor * patch_size, factor, dtype=np.float64)
    train_row, train_col = np.meshgrid(train_axis, train_axis, indexing="ij")
    x_train = np.column_stack((train_row.ravel(), train_col.ravel()))
    test_axis = np.arange(output_span, dtype=np.float64)
    test_row, test_col = np.meshgrid(test_axis, test_axis, indexing="ij")
    x_test = np.column_stack((test_row.ravel(), test_col.ravel()))

    accumulator = np.zeros(target_shape, dtype=np.float64)
    weights = np.zeros(target_shape, dtype=np.float64)
    locations = [(r, c) for r in row_starts for c in col_starts]
    iterator = tqdm(locations, desc="GP patches", disable=not show_progress)

    for row_start, col_start in iterator:
        patch = low_resolution[
            row_start : row_start + patch_size,
            col_start : col_start + patch_size,
        ]
        y_train = patch.ravel()
        scale = float(np.max(np.abs(y_train)))
        if scale <= np.finfo(np.float64).eps:
            prediction = np.zeros((output_span, output_span), dtype=np.float64)
        else:
            kernel = (
                ConstantKernel(1.0, (1e-3, 1e3))
                * RBF(length_scale=[2.0, 2.0], length_scale_bounds=(1e-2, 1e3))
                + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1.0))
            )
            model = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-8,
                normalize_y=True,
                optimizer="fmin_l_bfgs_b" if optimize_kernel else None,
                n_restarts_optimizer=0,
                random_state=1,
            )
            model.fit(x_train, y_train / scale)
            prediction = model.predict(x_test).reshape(output_span, output_span) * scale
            prediction = np.maximum(np.nan_to_num(prediction), 0.0)

        output_rows = factor * row_start + np.arange(output_span)
        output_cols = factor * col_start + np.arange(output_span)
        output_rows = output_rows[output_rows < target_shape[0]]
        output_cols = output_cols[output_cols < target_shape[1]]
        prediction = prediction[: output_rows.size, : output_cols.size]
        index = np.ix_(output_rows, output_cols)
        accumulator[index] += prediction
        weights[index] += 1.0

    reconstructed = np.zeros(target_shape, dtype=np.float64)
    covered = weights > 0
    reconstructed[covered] = accumulator[covered] / weights[covered]
    if not np.all(covered):
        baseline = _resize_baseline(low_resolution, target_shape, order=1)
        reconstructed[~covered] = baseline[~covered]
    return reconstructed.astype(np.float32)


def super_resolution_raw_dmri(
    data: np.ndarray,
    slice_indices: Iterable[int] = (19,),
    gradient_indices: Iterable[int] | None = None,
    patch_size: int = 10,
    factor: int = 2,
    optimize_kernel: bool = True,
    show_progress: bool = True,
) -> dict[str, np.ndarray]:
    """Reconstruct selected zero-based dMRI slices and gradients."""

    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError("data must have shape (x, y, z, gradients)")
    slices = np.asarray(list(slice_indices), dtype=int)
    gradients = (
        np.arange(data.shape[3], dtype=int)
        if gradient_indices is None
        else np.asarray(list(gradient_indices), dtype=int)
    )
    if slices.size == 0 or np.any(slices < 0) or np.any(slices >= data.shape[2]):
        raise IndexError("slice_indices contains an invalid index")
    if gradients.size == 0 or np.any(gradients < 0) or np.any(gradients >= data.shape[3]):
        raise IndexError("gradient_indices contains an invalid index")

    output_shape = data.shape[:2]
    result_shape = output_shape + (slices.size, gradients.size)
    gp = np.zeros(result_shape, dtype=np.float32)
    nearest = np.zeros_like(gp)
    bilinear = np.zeros_like(gp)

    total = slices.size * gradients.size
    task_index = 0
    for slice_output, source_slice in enumerate(slices):
        for gradient_output, source_gradient in enumerate(gradients):
            task_index += 1
            if show_progress:
                print(
                    f"Reconstruction {task_index}/{total}: "
                    f"slice={source_slice}, gradient={source_gradient}"
                )
            high = np.asarray(data[:, :, source_slice, source_gradient], dtype=np.float64)
            low = high[::factor, ::factor]
            gp[:, :, slice_output, gradient_output] = patch_gp_upsample(
                low,
                output_shape,
                patch_size=patch_size,
                factor=factor,
                optimize_kernel=optimize_kernel,
                show_progress=show_progress,
            )
            nearest[:, :, slice_output, gradient_output] = _resize_baseline(
                low, output_shape, order=0
            )
            bilinear[:, :, slice_output, gradient_output] = _resize_baseline(
                low, output_shape, order=1
            )

    return {
        "gp": gp,
        "nearest": nearest,
        "bilinear": bilinear,
        "slice_indices": slices,
        "gradient_indices": gradients,
    }

"""TensorFlow/Keras translation of dMRI_CNN_interpolation.m/.py."""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
try:
    from tqdm.auto import tqdm
except ImportError:  # Progress bars are optional outside Colab.
    def tqdm(iterable, **_kwargs):
        return iterable


def build_coordinate_cnn(patch_size: int = 15, hidden_units: int = 128) -> keras.Model:
    """Build the original coordinate-based Conv1D regression architecture."""

    model = keras.Sequential(
        [
            keras.Input(shape=(2, 1), name="coordinates"),
            layers.Conv1D(patch_size * patch_size, 2, activation="relu", name="conv1d"),
            layers.MaxPooling1D(pool_size=1, name="pool"),
            layers.Flatten(name="flatten"),
            layers.Dense(hidden_units, activation="relu", name="fc_hidden"),
            layers.Dense(1, name="prediction"),
        ],
        name="dmri_coordinate_cnn",
    )
    return model


def interpolate_cnn_patch(
    data: np.ndarray,
    slice_index: int = 19,
    gradient_index: int = 15,
    low_resolution_start: tuple[int, int] = (40, 50),
    patch_size: int = 15,
    factor: int = 2,
    epochs: int = 2000,
    learning_rate: float = 5e-4,
    verbose: int = 0,
) -> dict[str, object]:
    """Train and evaluate the CNN on one patch using zero-based indices."""

    data = np.asarray(data)
    if data.ndim != 4:
        raise ValueError("data must have shape (x, y, z, gradients)")
    if not 0 <= slice_index < data.shape[2]:
        raise IndexError("slice_index is outside the dMRI volume")
    if not 0 <= gradient_index < data.shape[3]:
        raise IndexError("gradient_index is outside the dMRI volume")

    high = np.asarray(data[:, :, slice_index, gradient_index], dtype=np.float32)
    low = high[::factor, ::factor]
    row_start, col_start = map(int, low_resolution_start)
    low_patch = low[
        row_start : row_start + patch_size,
        col_start : col_start + patch_size,
    ]
    if low_patch.shape != (patch_size, patch_size):
        raise ValueError("The requested low-resolution patch is outside the image")

    output_span = factor * patch_size - 1
    high_row_start = factor * row_start
    high_col_start = factor * col_start
    reference = high[
        high_row_start : high_row_start + output_span,
        high_col_start : high_col_start + output_span,
    ]
    if reference.shape != (output_span, output_span):
        raise ValueError("The corresponding high-resolution ROI is outside the image")

    train_axis = np.arange(0, factor * patch_size, factor, dtype=np.float32)
    train_row, train_col = np.meshgrid(train_axis, train_axis, indexing="ij")
    x_train = np.column_stack((train_row.ravel(), train_col.ravel())).reshape(-1, 2, 1)
    test_axis = np.arange(output_span, dtype=np.float32)
    test_row, test_col = np.meshgrid(test_axis, test_axis, indexing="ij")
    x_test = np.column_stack((test_row.ravel(), test_col.ravel())).reshape(-1, 2, 1)

    y_train = low_patch.ravel().astype(np.float32)
    y_reference = reference.ravel().astype(np.float32)
    scale = float(np.max(np.abs(y_train)))
    if scale > np.finfo(np.float32).eps:
        y_train_normalized = y_train / scale
        y_reference_normalized = y_reference / scale
    else:
        y_train_normalized = y_train
        y_reference_normalized = y_reference

    tf.keras.utils.set_random_seed(1)
    model = build_coordinate_cnn(patch_size)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )
    start = time.perf_counter()
    history = model.fit(
        x_train,
        y_train_normalized,
        batch_size=max(1, int(0.05 * patch_size * patch_size)),
        epochs=epochs,
        verbose=verbose,
    )
    prediction_normalized = model.predict(x_test, verbose=0).reshape(-1)
    elapsed_seconds = time.perf_counter() - start
    normalized_mse = float(np.mean((y_reference_normalized - prediction_normalized) ** 2))
    predicted_patch = (prediction_normalized * scale).reshape(output_span, output_span)

    return {
        "predicted_patch": predicted_patch,
        "reference_patch": reference,
        "low_resolution_patch": low_patch,
        "normalized_mse": normalized_mse,
        "elapsed_seconds": elapsed_seconds,
        "model": model,
        "history": history.history,
        "slice_index": slice_index,
        "gradient_index": gradient_index,
    }


def cnn_reconstruct_slice(
    high_resolution: np.ndarray,
    patch_size: int = 10,
    factor: int = 2,
    epochs_per_patch: int = 2000,
    learning_rate: float = 5e-4,
    show_progress: bool = True,
) -> dict[str, object]:
    """Translate the optimized MATLAB/Python patchwise CNN reconstruction.

    One network is trained sequentially across all overlapping patches, as in
    the original optimized script. Overlapping predictions are averaged and
    uncovered boundary pixels are filled with a bilinear baseline.
    """

    from scipy.ndimage import zoom

    high_resolution = np.asarray(high_resolution, dtype=np.float32)
    if high_resolution.ndim != 2:
        raise ValueError("high_resolution must be a two-dimensional image")
    low_resolution = high_resolution[::factor, ::factor]
    if patch_size > min(low_resolution.shape):
        raise ValueError("patch_size exceeds the low-resolution image dimensions")

    step = patch_size - 1

    def starts(length: int) -> list[int]:
        values = list(range(0, length - patch_size + 1, step))
        last = length - patch_size
        if values[-1] != last:
            values.append(last)
        return values

    row_starts = starts(low_resolution.shape[0])
    col_starts = starts(low_resolution.shape[1])
    output_span = factor * (patch_size - 1) + 1
    train_axis = np.arange(0, factor * patch_size, factor, dtype=np.float32)
    train_row, train_col = np.meshgrid(train_axis, train_axis, indexing="ij")
    x_train = np.column_stack((train_row.ravel(), train_col.ravel())).reshape(-1, 2, 1)
    test_axis = np.arange(output_span, dtype=np.float32)
    test_row, test_col = np.meshgrid(test_axis, test_axis, indexing="ij")
    x_test = np.column_stack((test_row.ravel(), test_col.ravel())).reshape(-1, 2, 1)

    tf.keras.utils.set_random_seed(1)
    model = build_coordinate_cnn(patch_size, hidden_units=256)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )
    accumulator = np.zeros_like(high_resolution, dtype=np.float64)
    weights = np.zeros_like(high_resolution, dtype=np.float64)
    locations = [(r, c) for r in row_starts for c in col_starts]

    for row_start, col_start in tqdm(
        locations, desc="CNN patches", disable=not show_progress
    ):
        patch = low_resolution[
            row_start : row_start + patch_size,
            col_start : col_start + patch_size,
        ]
        y_train = patch.ravel()
        scale = float(np.max(np.abs(y_train)))
        if scale <= np.finfo(np.float32).eps:
            prediction = np.zeros((output_span, output_span), dtype=np.float32)
        else:
            model.fit(
                x_train,
                y_train / scale,
                batch_size=max(1, int(0.15 * patch_size * patch_size)),
                epochs=epochs_per_patch,
                verbose=0,
            )
            prediction = (
                model.predict(x_test, verbose=0).reshape(output_span, output_span)
                * scale
            )
            prediction = np.maximum(np.nan_to_num(prediction), 0.0)

        output_rows = factor * row_start + np.arange(output_span)
        output_cols = factor * col_start + np.arange(output_span)
        output_rows = output_rows[output_rows < high_resolution.shape[0]]
        output_cols = output_cols[output_cols < high_resolution.shape[1]]
        prediction = prediction[: output_rows.size, : output_cols.size]
        index = np.ix_(output_rows, output_cols)
        accumulator[index] += prediction
        weights[index] += 1

    reconstructed = np.zeros_like(high_resolution, dtype=np.float64)
    covered = weights > 0
    reconstructed[covered] = accumulator[covered] / weights[covered]
    if not np.all(covered):
        factors = (
            high_resolution.shape[0] / low_resolution.shape[0],
            high_resolution.shape[1] / low_resolution.shape[1],
        )
        baseline = zoom(low_resolution, factors, order=1, mode="nearest")
        baseline = baseline[: high_resolution.shape[0], : high_resolution.shape[1]]
        reconstructed[~covered] = baseline[~covered]

    return {
        "reconstructed": reconstructed.astype(np.float32),
        "low_resolution": low_resolution,
        "network": model,
    }

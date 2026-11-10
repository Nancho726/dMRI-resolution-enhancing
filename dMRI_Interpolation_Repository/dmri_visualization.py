"""Visualizations replacing the MATLAB plotting scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def show_image_pair(
    reference: np.ndarray,
    estimate: np.ndarray,
    titles: tuple[str, str] = ("Reference", "Estimate"),
    cmap: str = "gray",
):
    """Display two images with identical intensity limits."""

    reference = np.asarray(reference)
    estimate = np.asarray(estimate)
    vmin = float(min(np.nanmin(reference), np.nanmin(estimate)))
    vmax = float(max(np.nanmax(reference), np.nanmax(estimate)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, image, title in zip(axes, (reference, estimate), titles):
        axis.imshow(image.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
    return fig, axes


def show_fa_map(fa: np.ndarray, slice_position: int = 0, title: str = "FA map"):
    """Display a selected FA map from a 2-D or 3-D FA array."""

    fa = np.asarray(fa)
    image = fa if fa.ndim == 2 else fa[:, :, slice_position]
    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    display = axis.imshow(image.T, origin="lower", cmap="summer", vmin=0, vmax=1)
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(display, ax=axis, label="FA")
    return fig, axis


def show_error_map(error_map: np.ndarray, title: str = "Error map", cmap: str = "viridis"):
    """Display a scalar error map with a colorbar."""

    fig, axis = plt.subplots(figsize=(6, 5), constrained_layout=True)
    display = axis.imshow(np.asarray(error_map).T, origin="lower", cmap=cmap)
    axis.set_title(title)
    axis.axis("off")
    fig.colorbar(display, ax=axis)
    return fig, axis


def plot_tensor_glyphs(
    tensor_field: np.ndarray,
    stride: int = 4,
    scale: float = 2e4,
    color_by_fa: np.ndarray | None = None,
):
    """Plot a 2-D field of axial tensor ellipses.

    ``tensor_field`` must have shape ``(rows, columns, 3, 3)``. This replaces
    the MATLAB ``DTIplot`` display while remaining lightweight for Colab.
    """

    tensor_field = np.asarray(tensor_field, dtype=np.float64)
    if tensor_field.ndim != 4 or tensor_field.shape[-2:] != (3, 3):
        raise ValueError("tensor_field must have shape (rows, columns, 3, 3)")
    fig, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.set_aspect("equal")

    rows, cols = tensor_field.shape[:2]
    for row in range(0, rows, stride):
        for col in range(0, cols, stride):
            tensor_2d = tensor_field[row, col, :2, :2]
            if not np.all(np.isfinite(tensor_2d)) or not np.any(tensor_2d):
                continue
            values, vectors = np.linalg.eigh(0.5 * (tensor_2d + tensor_2d.T))
            values = np.maximum(values, 0)
            order = np.argsort(values)[::-1]
            values, vectors = values[order], vectors[:, order]
            angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
            color = (
                plt.cm.summer(float(np.clip(color_by_fa[row, col], 0, 1)))
                if color_by_fa is not None
                else "tab:blue"
            )
            glyph = Ellipse(
                (col, row),
                width=max(values[0] * scale, 0.05),
                height=max(values[1] * scale, 0.05),
                angle=angle,
                facecolor="none",
                edgecolor=color,
                linewidth=0.7,
            )
            axis.add_patch(glyph)
    axis.set_xlim(-1, cols)
    axis.set_ylim(-1, rows)
    axis.invert_yaxis()
    axis.set_title("Diffusion tensor glyphs")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    return fig, axis


def plot_streamlines(streamlines, max_streamlines: int = 1000):
    """Display DIPY streamlines with Matplotlib in Colab."""

    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    axis = fig.add_subplot(111, projection="3d")
    for streamline in list(streamlines)[:max_streamlines]:
        streamline = np.asarray(streamline)
        axis.plot(streamline[:, 0], streamline[:, 1], streamline[:, 2], linewidth=0.4)
    axis.set_title(f"Deterministic tractography (up to {max_streamlines} streamlines)")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    return fig, axis

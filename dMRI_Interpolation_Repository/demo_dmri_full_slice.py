"""Full-slice patch reconstruction demo for raw dMRI.

The selected diffusion-weighted image is downsampled, divided into overlapping
low-resolution patches, reconstructed patch by patch, and joined into one
complete high-resolution slice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dmri_gp import patch_gp_upsample
from dmri_io import load_dmri_study


def _unsampled_metrics(
    reference: np.ndarray, estimate: np.ndarray, factor: int
) -> tuple[float, float, np.ndarray]:
    mask = np.ones(reference.shape, dtype=bool)
    mask[::factor, ::factor] = False
    error = np.abs(estimate - reference)
    rmse = float(np.sqrt(np.mean((estimate[mask] - reference[mask]) ** 2)))
    data_range = float(np.max(reference) - np.min(reference))
    nrmse = rmse / data_range if data_range > 0 else 0.0
    return rmse, nrmse, error


def _save_result(
    output: Path,
    method: str,
    reference: np.ndarray,
    low_resolution: np.ndarray,
    reconstruction: np.ndarray,
    factor: int,
    patch_count: int,
    weight_map: np.ndarray,
) -> None:
    rmse, nrmse, error = _unsampled_metrics(reference, reconstruction, factor)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"{method}_full_slice_results.npz",
        reference=reference,
        low_resolution=low_resolution,
        reconstruction=reconstruction,
        absolute_error=error,
        weight_map=weight_map,
        rmse_unsampled=rmse,
        nrmse_unsampled=nrmse,
        patch_count=patch_count,
    )
    figure, axes = plt.subplots(1, 4, figsize=(17, 4))
    panels = (
        (reference, "Reference", "gray"),
        (low_resolution, "Low resolution", "gray"),
        (reconstruction, f"{method.upper()} full slice", "gray"),
        (error, "Absolute error", "magma"),
    )
    for axis, (image, title, cmap) in zip(axes, panels, strict=True):
        shown = axis.imshow(image.T, cmap=cmap, origin="lower")
        axis.set_title(title)
        axis.axis("off")
        if title == "Absolute error":
            figure.colorbar(shown, ax=axis, fraction=0.046)
    figure.suptitle(
        f"{method.upper()}: {patch_count} overlapping patches; "
        f"unsampled RMSE={rmse:.6g}, NRMSE={nrmse:.6g}"
    )
    figure.tight_layout()
    figure.savefig(output / f"{method}_full_slice_comparison.png", dpi=200)
    plt.close(figure)
    print(
        f"{method.upper()} complete slice: {patch_count} patches, "
        f"RMSE={rmse:.6g}, NRMSE={nrmse:.6g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--slice", type=int, default=19, dest="slice_index")
    parser.add_argument("--gradient", type=int, default=15, dest="gradient_index")
    parser.add_argument("--method", choices=("gp", "cnn", "both"), default="gp")
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--cnn-epochs", type=int, default=5)
    parser.add_argument("--optimize-gp-kernel", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/full_slice_reconstruction")
    )
    args = parser.parse_args()

    study = load_dmri_study(args.root)
    reference = np.asarray(
        study.data[:, :, args.slice_index, args.gradient_index], dtype=np.float32
    )
    low_resolution = reference[:: args.factor, :: args.factor]

    if args.method in ("gp", "both"):
        gp = patch_gp_upsample(
            low_resolution,
            reference.shape,
            patch_size=args.patch_size,
            factor=args.factor,
            stride=args.stride,
            blending="hann",
            preserve_samples=True,
            optimize_kernel=args.optimize_gp_kernel,
            show_progress=True,
            return_details=True,
        )
        _save_result(
            args.output,
            "gp",
            reference,
            gp["low_resolution"],
            gp["reconstructed"],
            args.factor,
            gp["patch_count"],
            gp["weight_map"],
        )

    if args.method in ("cnn", "both"):
        # TensorFlow is imported only when CNN execution is requested.
        from dmri_cnn import cnn_reconstruct_slice

        cnn = cnn_reconstruct_slice(
            reference,
            patch_size=args.patch_size,
            factor=args.factor,
            stride=args.stride,
            blending="hann",
            preserve_samples=True,
            epochs_per_patch=args.cnn_epochs,
            show_progress=True,
        )
        _save_result(
            args.output,
            "cnn",
            reference,
            cnn["low_resolution"],
            cnn["reconstructed"],
            args.factor,
            cnn["patch_count"],
            cnn["weight_map"],
        )


if __name__ == "__main__":
    main()


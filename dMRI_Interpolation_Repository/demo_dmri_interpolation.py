"""Command-line demo for nearest-neighbor and bilinear raw-dMRI interpolation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dmri_interpolation import reconstruct_dmri
from dmri_io import load_dmri_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--slice", type=int, default=19, dest="slice_index")
    parser.add_argument("--gradient", type=int, default=15, dest="gradient_index")
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/dmri_interpolation"))
    args = parser.parse_args()

    study = load_dmri_study(args.root)
    result = reconstruct_dmri(
        study.data,
        factor=args.factor,
        slice_indices=[args.slice_index],
        gradient_indices=[args.gradient_index],
    )
    reference = result["reference"][:, :, 0, 0]
    nearest = result["nearest"][:, :, 0, 0]
    bilinear = result["bilinear"][:, :, 0, 0]
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "dmri_interpolation_results.npz",
        reference=reference,
        low_resolution=result["low_resolution"][:, :, 0, 0],
        nearest=nearest,
        bilinear=bilinear,
    )

    figure, axes = plt.subplots(1, 4, figsize=(15, 4))
    images = [
        reference,
        result["low_resolution"][:, :, 0, 0],
        nearest,
        bilinear,
    ]
    titles = ["Reference", "Low resolution", "Nearest neighbor", "Bilinear"]
    for axis, image, title in zip(axes, images, titles, strict=True):
        axis.imshow(image.T, cmap="gray", origin="lower")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(args.output / "dmri_interpolation_comparison.png", dpi=200)
    plt.close(figure)
    for name, estimate in (("nearest", nearest), ("bilinear", bilinear)):
        rmse = float(np.sqrt(np.mean((estimate - reference) ** 2)))
        print(f"{name} RMSE: {rmse:.6g}")


if __name__ == "__main__":
    main()


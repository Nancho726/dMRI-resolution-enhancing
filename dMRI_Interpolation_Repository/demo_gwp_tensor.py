"""Command-line demo for GWP interpolation of a second-order tensor field."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gwp_tensor import gwp_interpolate
from tensor_utils import frobenius_error, load_tensor_field, unsampled_mask


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=root / "data" / "tensors" / "training_data2.mat",
    )
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--length-scale", type=float, default=2.0)
    parser.add_argument("--crop", type=int, default=0, help="0 uses the complete field")
    parser.add_argument("--output", type=Path, default=Path("results/gwp"))
    args = parser.parse_args()

    reference = load_tensor_field(args.data)
    if args.crop:
        reference = reference[:, : args.crop, : args.crop]
    low_resolution = reference[:, :: args.factor, :: args.factor]
    result = gwp_interpolate(
        low_resolution,
        target_shape=reference.shape[1:],
        factor=args.factor,
        length_scale=args.length_scale,
    )
    metrics = frobenius_error(
        result["coefficients"],
        reference,
        order=2,
        mask=unsampled_mask(reference.shape[1:], args.factor),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "gwp_results.npz",
        reference=reference,
        low_resolution=low_resolution,
        reconstruction=result["coefficients"],
        error_map=metrics["map"],
        scale_matrix=result["scale_matrix"],
    )
    print(f"GWP Frobenius distance: {metrics['mean']:.6g} +/- {metrics['std']:.6g}")


if __name__ == "__main__":
    main()


"""Command-line demo for CDP interpolation of order-2, -4, or -6 tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cdp_tensor import FAST_RANKS, MATLAB_LEGACY_RANKS, cdp_interpolate
from tensor_utils import frobenius_error, load_tensor_field, unsampled_mask


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=(2, 4, 6), default=4)
    parser.add_argument("--real", action="store_true", help="Use the real tensor field")
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--legacy-rank", action="store_true")
    parser.add_argument("--crop", type=int, default=0, help="0 uses the complete field")
    parser.add_argument("--output", type=Path, default=Path("results/cdp"))
    args = parser.parse_args()

    suffix = "_real" if args.real else ""
    data_path = root / "data" / "tensors" / f"training_data{args.order}{suffix}.mat"
    reference = load_tensor_field(data_path)
    if args.crop:
        reference = reference[:, : args.crop, : args.crop]
    rank = args.rank
    if rank is None:
        rank = MATLAB_LEGACY_RANKS[args.order] if args.legacy_rank else FAST_RANKS[args.order]
    low_resolution = reference[:, :: args.factor, :: args.factor]
    result = cdp_interpolate(
        low_resolution,
        target_shape=reference.shape[1:],
        order=args.order,
        factor=args.factor,
        rank=rank,
    )
    metrics = frobenius_error(
        result["coefficients"],
        reference,
        order=args.order,
        mask=unsampled_mask(reference.shape[1:], args.factor),
    )
    output = args.output / f"order_{args.order}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "cdp_results.npz",
        reference=reference,
        low_resolution=low_resolution,
        reconstruction=result["coefficients"],
        error_map=metrics["map"],
        weights_train=result["spatial_weights_train"],
        weights_target=result["spatial_weights_target"],
    )
    print(
        f"CDP order {args.order}, rank {rank}: Frobenius distance "
        f"{metrics['mean']:.6g} +/- {metrics['std']:.6g}; "
        f"CP basis error={result['cp_relative_error']:.6g}"
    )


if __name__ == "__main__":
    main()


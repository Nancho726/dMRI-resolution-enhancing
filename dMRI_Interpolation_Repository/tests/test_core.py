"""Fast dependency-light tests for direct dMRI, GWP, and CDP code."""

from __future__ import annotations

import unittest

import numpy as np

from cdp_tensor import cdp_interpolate, cp_to_tensor
from dmri_gp import patch_gp_upsample
from dmri_interpolation import downsample_in_plane, interpolate_in_plane
from gwp_tensor import gwp_interpolate
from tensor_utils import coefficients_to_full, full_to_coefficients


class CoreTests(unittest.TestCase):
    def test_raw_dmri_interpolation_preserves_samples(self) -> None:
        data = np.arange(8 * 10 * 3, dtype=np.float32).reshape(8, 10, 3)
        low = downsample_in_plane(data, factor=2)
        for method in ("nearest", "bilinear"):
            reconstructed = interpolate_in_plane(low, (8, 10), factor=2, method=method)
            self.assertEqual(reconstructed.shape, data.shape)
            np.testing.assert_allclose(reconstructed[::2, ::2], low)

    def test_patch_gp_joins_a_complete_image(self) -> None:
        rows, cols = np.meshgrid(
            np.linspace(0, 1, 8), np.linspace(0, 1, 10), indexing="ij"
        )
        reference = (rows + 0.5 * cols).astype(np.float32)
        low = reference[::2, ::2]
        result = patch_gp_upsample(
            low,
            reference.shape,
            patch_size=3,
            factor=2,
            stride=2,
            optimize_kernel=False,
            show_progress=False,
            return_details=True,
        )
        self.assertEqual(result["reconstructed"].shape, reference.shape)
        self.assertGreater(result["patch_count"], 1)
        self.assertTrue(np.all(result["weight_map"] > 0))
        np.testing.assert_allclose(result["reconstructed"][::2, ::2], low)

    def test_symmetric_tensor_round_trip(self) -> None:
        rng = np.random.default_rng(2)
        for order, channels in ((2, 6), (4, 15), (6, 28)):
            coefficients = rng.normal(size=(channels, 4, 5))
            recovered = full_to_coefficients(
                coefficients_to_full(coefficients, order), order
            )
            np.testing.assert_allclose(recovered, coefficients)

    def test_gwp_returns_spd_predictions(self) -> None:
        field = np.zeros((6, 8, 8), dtype=np.float64)
        for row in range(8):
            for col in range(8):
                tensor = np.diag([1.0 + 0.02 * row, 0.8 + 0.01 * col, 0.6])
                field[:, row, col] = tensor[2, 2], 0, tensor[1, 1], 0, 0, tensor[0, 0]
        low = field[:, ::2, ::2]
        result = gwp_interpolate(low, (8, 8), factor=2)
        self.assertEqual(result["coefficients"].shape, field.shape)
        np.testing.assert_allclose(result["coefficients"][:, ::2, ::2], low)
        unsampled = result["tensors"][1::2, 1::2]
        self.assertGreater(float(np.min(np.linalg.eigvalsh(unsampled))), 0.0)

    def test_cdp_handles_orders_2_4_6(self) -> None:
        rng = np.random.default_rng(4)
        for order in (2, 4, 6):
            factors = rng.normal(size=(3, 2))
            base = cp_to_tensor(np.array([1.0, 0.4]), [factors] * order)
            field = np.empty((6, 6) + (3,) * order)
            for row in range(6):
                for col in range(6):
                    field[row, col] = (1 + 0.04 * row + 0.02 * col) * base
            coefficients = full_to_coefficients(field, order)
            result = cdp_interpolate(
                coefficients[:, ::2, ::2],
                (6, 6),
                order=order,
                factor=2,
                rank=2,
                cp_iterations=80,
            )
            self.assertEqual(result["coefficients"].shape, coefficients.shape)
            np.testing.assert_allclose(
                result["coefficients"][:, ::2, ::2], coefficients[:, ::2, ::2]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)


"""Fast Colab-compatible validation of the converted project."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dmri_dti import compute_fa, dti_design_matrix, fit_dti
from dmri_io import load_dmri_study
from dmri_metrics import absolute_relative_error, frobenius_map, tensor_coefficient_distance


def run_smoke_tests(project_root: str | Path = ".") -> None:
    study = load_dmri_study(project_root)
    assert study.data.shape == (256, 256, 33, 26)
    assert study.bvals.shape == (26,)
    assert study.bvecs.shape == (26, 3)
    design = dti_design_matrix(study.bvals, study.bvecs)
    assert design.shape == (26, 7) and np.linalg.matrix_rank(design) == 7

    relative = absolute_relative_error(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    assert np.all(np.isfinite(relative))
    matrices = np.zeros((2, 2, 3, 3))
    matrices_estimate = matrices.copy()
    matrices_estimate[..., 0, 0] = 1
    matrix_map, matrix_values = frobenius_map(matrices, matrices_estimate)
    assert matrix_map.shape == (2, 2) and np.all(matrix_values == 1)
    coefficients = np.zeros((2, 2, 6))
    coefficient_map, _ = tensor_coefficient_distance(
        coefficients, np.ones_like(coefficients), 2
    )
    assert coefficient_map.shape == (2, 2)

    # One real source slice validates the complete DTI/FA numerical pipeline.
    fitted = fit_dti(
        study.data,
        study.bvals,
        study.bvecs,
        slice_indices=[19],
    )
    fa_result = compute_fa(fitted["coefficients"])
    assert fa_result["fa"].shape == (256, 256, 1)
    assert np.all(np.isfinite(fa_result["fa"]))
    print("All Python/Colab smoke tests passed.")


if __name__ == "__main__":
    run_smoke_tests(Path(__file__).resolve().parent)

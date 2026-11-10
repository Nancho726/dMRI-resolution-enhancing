"""Optional DIPY replacement for the legacy MATLAB tractography scripts."""

from __future__ import annotations

import numpy as np


def deterministic_tractography(
    data: np.ndarray,
    bvals: np.ndarray,
    bvecs: np.ndarray,
    affine: np.ndarray,
    mask: np.ndarray | None = None,
    fa_threshold: float = 0.2,
    seed_density: int = 1,
    step_size: float = 0.5,
    max_angle: float = 30.0,
):
    """Generate deterministic streamlines from a fitted DTI model.

    DIPY imports are intentionally local so the rest of the project can run
    without installing tractography dependencies.
    """

    from dipy.core.gradients import gradient_table
    from dipy.data import default_sphere
    from dipy.direction import peaks_from_model
    from dipy.reconst.dti import TensorModel, fractional_anisotropy
    from dipy.tracking.local_tracking import LocalTracking
    from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
    from dipy.tracking.utils import seeds_from_mask

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 4:
        raise ValueError("data must have shape (x, y, z, gradients)")
    if mask is None:
        b0_indices = np.asarray(bvals) < 50
        baseline = data[..., b0_indices].mean(axis=-1)
        mask = baseline > 0.1 * np.max(baseline)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != data.shape[:3]:
        raise ValueError("mask must match the spatial data dimensions")

    gradient_table_object = gradient_table(bvals, bvecs=bvecs)
    model = TensorModel(gradient_table_object)
    fit = model.fit(data, mask=mask)
    fa = np.nan_to_num(fractional_anisotropy(fit.evals))
    stopping = ThresholdStoppingCriterion(fa, fa_threshold)
    direction_getter = peaks_from_model(
        model,
        data,
        sphere=default_sphere,
        relative_peak_threshold=0.5,
        min_separation_angle=max_angle,
        mask=mask,
        return_sh=False,
        parallel=False,
    )
    seeds = seeds_from_mask(mask, affine=affine, density=seed_density)
    streamlines = LocalTracking(
        direction_getter,
        stopping,
        seeds,
        affine=affine,
        step_size=step_size,
        return_all=False,
    )
    return list(streamlines), fa, fit

"""Input/output helpers for the dMRI PatchGP Colab project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class DmriStudy:
    """Validated diffusion-MRI study and gradient information."""

    data: np.ndarray
    bvals: np.ndarray
    bvecs: np.ndarray
    affine: np.ndarray
    header: nib.nifti1.Nifti1Header
    image_path: Path


def _resolve_image(project_root: Path) -> Path:
    candidates = (
        project_root / "TENSOR_2x2x2.nii.gz",
        project_root / "TENSOR_2x2x2.nii",
        project_root / "data" / "dmri" / "TENSOR_2x2x2.nii.gz",
        project_root / "data" / "dmri" / "TENSOR_2x2x2.nii",
        project_root / "dMRI_Data" / "TENSOR_2x2x2.nii.gz",
        project_root / "dMRI_Data" / "TENSOR_2x2x2.nii",
        project_root.parent / "dMRI_Data" / "TENSOR_2x2x2.nii.gz",
        project_root.parent / "dMRI_Data" / "TENSOR_2x2x2.nii",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "The external dMRI_Data folder was not found. Place dMRI_Data next "
        f"to the repository folder. Project root: {project_root}"
    )


def load_dmri_study(project_root: str | Path = ".", dtype=np.float32) -> DmriStudy:
    """Load the NIfTI study, b-values, and normalized b-vectors.

    Both 3-by-N and N-by-3 b-vector layouts are accepted. The returned data
    layout is ``(x, y, z, gradients)``.
    """

    project_root = Path(project_root).expanduser().resolve()
    image_path = _resolve_image(project_root)
    gradient_root = image_path.parent
    bval_path = gradient_root / "TENSOR_2x2x2.bval"
    bvec_path = gradient_root / "TENSOR_2x2x2.bvec"
    for path in (bval_path, bvec_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required gradient file not found: {path}")

    image = nib.load(str(image_path))
    data = image.get_fdata(dtype=dtype)
    if data.ndim != 4:
        raise ValueError(f"Expected a 4-D dMRI volume, received shape {data.shape}")

    bvals = np.asarray(np.loadtxt(bval_path), dtype=np.float64).reshape(-1)
    bvecs = np.asarray(np.loadtxt(bvec_path), dtype=np.float64)
    if bvecs.ndim != 2:
        raise ValueError("The b-vector file must be a two-dimensional matrix")
    if bvecs.shape[0] == 3:
        bvecs = bvecs.T
    elif bvecs.shape[1] != 3:
        raise ValueError("The b-vector file must have shape 3-by-N or N-by-3")

    n_gradients = data.shape[3]
    if bvals.size != n_gradients or bvecs.shape != (n_gradients, 3):
        raise ValueError(
            "Gradient count mismatch: "
            f"NIfTI={n_gradients}, bvals={bvals.size}, bvecs={bvecs.shape[0]}"
        )

    norms = np.linalg.norm(bvecs, axis=1)
    nonzero = norms > np.sqrt(np.finfo(np.float64).eps)
    bvecs = bvecs.copy()
    bvecs[nonzero] /= norms[nonzero, None]
    bvecs[~nonzero] = 0.0

    return DmriStudy(
        data=data,
        bvals=bvals,
        bvecs=bvecs,
        affine=np.asarray(image.affine, dtype=np.float64),
        header=image.header.copy(),
        image_path=image_path,
    )


def save_nifti(data: np.ndarray, affine: np.ndarray, output_path: str | Path) -> Path:
    """Save a NumPy array as a compressed NIfTI file."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), affine), str(output_path))
    return output_path

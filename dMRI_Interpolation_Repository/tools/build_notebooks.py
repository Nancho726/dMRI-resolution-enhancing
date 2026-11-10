"""Build all Google Colab notebooks with the Python standard library."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


BOOTSTRAP = code(
    """
    from pathlib import Path
    import subprocess
    import sys
    import zipfile

    PROJECT_ROOT = Path('/content/dMRI_Interpolation_Repository')
    DATA_ROOT = Path('/content/dMRI_Data')
    if not PROJECT_ROOT.is_dir():
        from google.colab import files
        print('Upload dMRI_Interpolation_Repository.zip')
        uploaded = files.upload()
        archives = [name for name in uploaded if name.lower().endswith('.zip')]
        if not archives:
            raise FileNotFoundError('No ZIP archive was uploaded.')
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall('/content')
    if not PROJECT_ROOT.is_dir():
        raise FileNotFoundError(f'Project directory not found: {PROJECT_ROOT}')
    if not (DATA_ROOT / 'TENSOR_2x2x2.nii.gz').is_file():
        from google.colab import files
        print('Upload dMRI_Data.zip')
        uploaded = files.upload()
        archives = [name for name in uploaded if name.lower().endswith('.zip')]
        if not archives:
            raise FileNotFoundError('No dMRI data ZIP archive was uploaded.')
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall('/content')
    if not (DATA_ROOT / 'TENSOR_2x2x2.nii.gz').is_file():
        raise FileNotFoundError(f'dMRI data folder not found: {DATA_ROOT}')
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q', '-r',
        str(PROJECT_ROOT / 'requirements_colab.txt')
    ])
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    print('Project ready:', PROJECT_ROOT)
    print('External dMRI data ready:', DATA_ROOT)
    """
)


def notebook(title: str, description: str, body: list[dict]) -> dict:
    return {
        "cells": [
            markdown(f"# {title}\n\n{description}"),
            markdown("## 1. Prepare the Colab environment"),
            BOOTSTRAP,
            *body,
        ],
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


notebooks: dict[str, dict] = {}

notebooks["01_Nearest_Bilinear_dMRI.ipynb"] = notebook(
    "Nearest-neighbor and bilinear interpolation of raw dMRI",
    "This notebook operates directly on diffusion-weighted images. It does not use tensor fields.",
    [
        markdown("## 2. Load the dMRI study"),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from dmri_io import load_dmri_study
            from dmri_interpolation import reconstruct_dmri

            study = load_dmri_study(PROJECT_ROOT)
            SLICE_INDEX = 19
            GRADIENT_INDEX = 15
            result = reconstruct_dmri(
                study.data,
                factor=2,
                slice_indices=[SLICE_INDEX],
                gradient_indices=[GRADIENT_INDEX],
            )
            print('dMRI shape:', study.data.shape)
            """
        ),
        markdown("## 3. Compare reference, downsampled image, and reconstructions"),
        code(
            """
            reference = result['reference'][:, :, 0, 0]
            low = result['low_resolution'][:, :, 0, 0]
            nearest = result['nearest'][:, :, 0, 0]
            bilinear = result['bilinear'][:, :, 0, 0]
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            for ax, image, title in zip(
                axes,
                (reference, low, nearest, bilinear),
                ('Reference', 'Low resolution', 'Nearest neighbor', 'Bilinear'),
            ):
                ax.imshow(image.T, cmap='gray', origin='lower')
                ax.set_title(title)
                ax.axis('off')
            plt.tight_layout()
            for name, image in [('nearest', nearest), ('bilinear', bilinear)]:
                rmse = np.sqrt(np.mean((image - reference) ** 2))
                print(f'{name} RMSE: {rmse:.6g}')
            """
        ),
    ],
)

notebooks["02_GP_CNN_dMRI.ipynb"] = notebook(
    "Patch GP and CNN reconstruction of raw dMRI",
    "Both methods receive raw diffusion-weighted intensities. The short examples use one selected image patch.",
    [
        markdown("## 2. Load one diffusion-weighted image"),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from dmri_io import load_dmri_study
            from dmri_gp import patch_gp_upsample
            from dmri_cnn import interpolate_cnn_patch

            study = load_dmri_study(PROJECT_ROOT)
            SLICE_INDEX, GRADIENT_INDEX = 19, 15
            high = study.data[:, :, SLICE_INDEX, GRADIENT_INDEX]
            low = high[::2, ::2]
            """
        ),
        markdown("## 3. Patch-based Gaussian process"),
        code(
            """
            low_roi = low[36:64, 46:74]
            reference_roi = high[72:128, 92:148]
            gp_roi = patch_gp_upsample(
                low_roi, reference_roi.shape, patch_size=10, factor=2,
                optimize_kernel=False, show_progress=True,
            )
            print('GP RMSE:', float(np.sqrt(np.mean((gp_roi-reference_roi)**2))))
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(reference_roi.T, cmap='gray', origin='lower'); axes[0].set_title('Reference')
            axes[1].imshow(gp_roi.T, cmap='gray', origin='lower'); axes[1].set_title('Patch GP')
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            """
        ),
        markdown("## 4. Coordinate CNN translated from the original Python/MATLAB demo"),
        code(
            """
            CNN_EPOCHS = 20  # Change to 2000 for the original long configuration.
            cnn = interpolate_cnn_patch(
                study.data, slice_index=SLICE_INDEX, gradient_index=GRADIENT_INDEX,
                low_resolution_start=(40, 50), patch_size=15, factor=2,
                epochs=CNN_EPOCHS, verbose=0,
            )
            print('CNN normalized MSE:', cnn['normalized_mse'])
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(cnn['reference_patch'].T, cmap='gray', origin='lower'); axes[0].set_title('Reference')
            axes[1].imshow(cnn['predicted_patch'].T, cmap='gray', origin='lower'); axes[1].set_title('CNN')
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            """
        ),
    ],
)

notebooks["03_GWP_Tensor.ipynb"] = notebook(
    "Generalized Wishart process for second-order tensors",
    "GWP receives a second-order tensor coefficient field, not raw dMRI.",
    [
        markdown("## 2. Load and downsample the tensor field"),
        code(
            """
            import matplotlib.pyplot as plt
            from gwp_tensor import gwp_interpolate
            from tensor_utils import frobenius_error, load_tensor_field, unsampled_mask

            path = PROJECT_ROOT / 'data' / 'tensors' / 'training_data2.mat'
            reference = load_tensor_field(path)
            low = reference[:, ::2, ::2]
            print('Reference:', reference.shape, 'Low resolution:', low.shape)
            """
        ),
        markdown("## 3. GWP interpolation and tensor-domain evaluation"),
        code(
            """
            gwp = gwp_interpolate(low, reference.shape[1:], factor=2, length_scale=2.0)
            metrics = frobenius_error(
                gwp['coefficients'], reference, order=2,
                mask=unsampled_mask(reference.shape[1:], 2),
            )
            print(f"Frobenius distance: {metrics['mean']:.6g} +/- {metrics['std']:.6g}")
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(reference[5], cmap='viridis'); axes[0].set_title('Reference T11')
            axes[1].imshow(gwp['coefficients'][5], cmap='viridis'); axes[1].set_title('GWP T11')
            axes[2].imshow(metrics['map'], cmap='magma'); axes[2].set_title('Frobenius error')
            for ax in axes: ax.axis('off')
            plt.tight_layout()
            """
        ),
    ],
)

notebooks["04_CDP_Tensors.ipynb"] = notebook(
    "Canonical decomposition processes for order-2, -4, and -6 tensors",
    "CDP receives tensor coefficient fields. The notebook never feeds raw dMRI to CDP.",
    [
        markdown("## 2. Run CDP for each supported tensor order"),
        code(
            """
            import matplotlib.pyplot as plt
            from cdp_tensor import FAST_RANKS, cdp_interpolate
            from tensor_utils import frobenius_error, load_tensor_field, unsampled_mask

            results = {}
            for order in (2, 4, 6):
                path = PROJECT_ROOT / 'data' / 'tensors' / f'training_data{order}.mat'
                reference = load_tensor_field(path)
                low = reference[:, ::2, ::2]
                cdp = cdp_interpolate(
                    low, reference.shape[1:], order=order, factor=2,
                    rank=FAST_RANKS[order], cp_iterations=80,
                )
                metrics = frobenius_error(
                    cdp['coefficients'], reference, order=order,
                    mask=unsampled_mask(reference.shape[1:], 2),
                )
                results[order] = (reference, cdp, metrics)
                print(
                    f"Order {order}, rank {cdp['rank']}: "
                    f"{metrics['mean']:.6g} +/- {metrics['std']:.6g}; "
                    f"basis error={cdp['cp_relative_error']:.6g}"
                )
            """
        ),
        markdown("## 3. Error maps"),
        code(
            """
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            for axis, order in zip(axes, (2, 4, 6)):
                image = axis.imshow(results[order][2]['map'], cmap='magma')
                axis.set_title(f'CDP order {order}')
                axis.axis('off')
                fig.colorbar(image, ax=axis, fraction=0.046)
            plt.tight_layout()
            """
        ),
    ],
)

notebooks["00_All_Methods_Colab.ipynb"] = notebook(
    "All dMRI and tensor-domain interpolation demos",
    "Consolidated entry point. Raw-signal methods and tensor-domain methods remain explicitly separated.",
    [
        markdown("## 2. Validate core numerical functions"),
        code(
            """
            import subprocess, sys
            subprocess.check_call([sys.executable, str(PROJECT_ROOT / 'run_core_tests.py')])
            """
        ),
        markdown("## 3. Raw dMRI: nearest neighbor, bilinear, and patch GP"),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            from dmri_io import load_dmri_study
            from dmri_interpolation import reconstruct_dmri
            from dmri_gp import patch_gp_upsample

            study = load_dmri_study(PROJECT_ROOT)
            raw = reconstruct_dmri(study.data, factor=2, slice_indices=[19], gradient_indices=[15])
            reference = raw['reference'][:, :, 0, 0]
            for method in ('nearest', 'bilinear'):
                estimate = raw[method][:, :, 0, 0]
                print(method, 'RMSE =', float(np.sqrt(np.mean((estimate-reference)**2))))
            low_roi = raw['low_resolution'][36:64, 46:74, 0, 0]
            ref_roi = reference[72:128, 92:148]
            gp_roi = patch_gp_upsample(low_roi, ref_roi.shape, patch_size=10, factor=2,
                                       optimize_kernel=False, show_progress=False)
            print('patch GP RMSE =', float(np.sqrt(np.mean((gp_roi-ref_roi)**2))))
            """
        ),
        markdown("## 4. Raw dMRI: CNN patch demo"),
        code(
            """
            from dmri_cnn import interpolate_cnn_patch
            cnn = interpolate_cnn_patch(
                study.data, slice_index=19, gradient_index=15,
                low_resolution_start=(40, 50), patch_size=15,
                factor=2, epochs=10, verbose=0,
            )
            print('CNN normalized MSE =', cnn['normalized_mse'])
            """
        ),
        markdown("## 5. Tensor domain: GWP for order 2"),
        code(
            """
            from gwp_tensor import gwp_interpolate
            from tensor_utils import frobenius_error, load_tensor_field, unsampled_mask
            tensor_root = PROJECT_ROOT / 'data' / 'tensors'
            tensor2 = load_tensor_field(tensor_root / 'training_data2.mat')
            gwp = gwp_interpolate(tensor2[:, ::2, ::2], tensor2.shape[1:], factor=2)
            gwp_error = frobenius_error(
                gwp['coefficients'], tensor2, 2, unsampled_mask(tensor2.shape[1:], 2)
            )
            print('GWP:', gwp_error['mean'], '+/-', gwp_error['std'])
            """
        ),
        markdown("## 6. Tensor domain: CDP for orders 2, 4, and 6"),
        code(
            """
            from cdp_tensor import FAST_RANKS, cdp_interpolate
            for order in (2, 4, 6):
                field = load_tensor_field(tensor_root / f'training_data{order}.mat')
                cdp = cdp_interpolate(
                    field[:, ::2, ::2], field.shape[1:], order=order, factor=2,
                    rank=FAST_RANKS[order], cp_iterations=80,
                )
                error = frobenius_error(
                    cdp['coefficients'], field, order, unsampled_mask(field.shape[1:], 2)
                )
                print(f"CDP order {order}: {error['mean']} +/- {error['std']}")
            """
        ),
        markdown("## 7. DTI and FA from the raw dMRI study"),
        code(
            """
            from dmri_dti import compute_fa, fit_dti
            dti = fit_dti(study.data, study.bvals, study.bvecs, slice_indices=[19])
            fa = compute_fa(dti['coefficients'])['fa'][:, :, 0]
            print('FA range:', float(fa.min()), float(fa.max()))
            plt.figure(figsize=(5, 5)); plt.imshow(fa.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
            plt.colorbar(label='FA'); plt.title('Reference FA'); plt.axis('off');
            """
        ),
    ],
)


NOTEBOOKS.mkdir(parents=True, exist_ok=True)
for filename, payload in notebooks.items():
    destination = NOTEBOOKS / filename
    destination.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


# dMRI interpolation and tensor-process methods

Version 1.0.0

Reproducible Python and Google Colab implementation accompanying the research
article on patch-based Gaussian-process reconstruction of diffusion MRI. The
repository implements raw-signal and tensor-domain interpolation methods and
contains second-, fourth-, and sixth-order tensor fields, reusable modules,
command-line demonstrations, automated tests, and Colab notebooks. The raw
dMRI study and gradient files are distributed separately in `dMRI_Data.zip`.

Authors: Hernán Darío Vargas-Cardona and Cristian Torres. Author identifiers
and contributions are provided in `AUTHORS.md` and `CITATION.cff`.

## Important domain separation

| Method | Input domain | Python module |
|---|---|---|
| Nearest neighbor | Raw dMRI signal | `dmri_interpolation.py` |
| Bilinear | Raw dMRI signal | `dmri_interpolation.py` |
| Patch GP | Raw dMRI signal | `dmri_gp.py` |
| CNN | Raw dMRI signal | `dmri_cnn.py` |
| GWP | Second-order tensor field | `gwp_tensor.py` |
| CDP | Order-2, order-4, or order-6 tensor field | `cdp_tensor.py` |

GWP and CDP do **not** receive the raw dMRI volume. Their inputs are the
tensor coefficient fields in `data/tensors/`.

## Repository layout

```text
dMRI_Interpolation_Repository/
├── data/
│   ├── dmri/       # Instructions for the external dMRI package
│   └── tensors/    # MATLAB tensor coefficient fields for GWP/CDP
├── notebooks/      # Google Colab demonstrations
├── tests/          # Fast numerical tests
├── dmri_interpolation.py
├── dmri_gp.py
├── dmri_cnn.py
├── dmri_dti.py
├── gwp_tensor.py
├── cdp_tensor.py
├── tensor_utils.py
└── requirements_colab.txt

dMRI_Data/                         # External sibling folder
├── TENSOR_2x2x2.nii.gz
├── TENSOR_2x2x2.bval
└── TENSOR_2x2x2.bvec
```

## Google Colab

1. Open one of the notebooks in `notebooks/` in Google Colab.
2. Upload `dMRI_Interpolation_Repository.zip` when prompted.
3. Upload `dMRI_Data.zip` when prompted.
4. Run the cells in order. Each notebook installs its dependencies and finds
   both the external dMRI data and the included tensor fields automatically.

Recommended starting point: `00_All_Methods_Colab.ipynb`.

## Local quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_colab.txt
python run_core_tests.py
```

Windows users can activate the environment with `.venv\Scripts\activate`.
For local execution, place `dMRI_Data` next to
`dMRI_Interpolation_Repository`, not inside it.

## Demos

- `01_Nearest_Bilinear_dMRI.ipynb`: direct nearest-neighbor and bilinear
  interpolation on a selected dMRI slice and gradient.
- `02_GP_CNN_dMRI.ipynb`: patch GP and CNN workflows on raw dMRI.
- `03_GWP_Tensor.ipynb`: GWP reconstruction of a second-order tensor field.
- `04_CDP_Tensors.ipynb`: CDP reconstruction for tensor orders 2, 4, and 6.
- `00_All_Methods_Colab.ipynb`: consolidated execution entry point.

Equivalent command-line entry points are `demo_dmri_interpolation.py`,
`demo_gwp_tensor.py`, and `demo_cdp_tensor.py`.

## Mathematical correspondence

The GWP module uses

```text
T(x) = sum_i L u_i(x) u_i(x)^T L^T,
```

with RBF Gaussian processes for the latent components. This construction
produces positive-definite predictions. Original observed tensors are restored
at sampled locations, matching the replacement step in the MATLAB demo.

The CDP module estimates a shared CP/CANDECOMP basis and interpolates its
spatially varying component weights with RBF Gaussian processes. It supports
the coefficient counts used by the MATLAB project: 6 (order 2), 15 (order 4),
and 28 (order 6). `FAST_RANKS` provides practical Colab defaults;
`MATLAB_LEGACY_RANKS` retains the original rank settings 10, 12, and 16.

## Tests

In Colab or a local environment with the requirements installed:

```bash
python run_core_tests.py
python run_smoke_tests.py
```

The first command validates interpolation, tensor ordering, GWP, and CDP on
synthetic data. The second loads the included NIfTI study and validates the
gradient table, DTI fit, and FA calculation.

## Data conventions

- Raw dMRI: `(x, y, z, gradients)`.
- Tensor coefficients: `(coefficients, rows, columns)`.
- MATLAB tensor coefficient order is preserved.
- Python indices are zero based. MATLAB slice 20 and gradient 16 correspond to
  Python indices 19 and 15.

## Reproducibility note

The Python implementation preserves the mathematical constructions and data
flow of the MATLAB demos. It uses conditional GP means and a NumPy CP-ALS
solver so that the notebooks run without the discontinued MATLAB GP and tensor
toolboxes. Random seeds are fixed in the CDP decomposition and CNN examples.

Detailed execution settings, full-versus-short configurations, and expected
test outcomes are documented in `REPRODUCIBILITY.md`. Dataset provenance,
dimensions, file roles, and coefficient conventions are documented in
`DATA.md`.

## Citation

Use the **Cite this repository** function on GitHub or import
`CITATION.cff` into a reference manager. When the associated article receives
its final bibliographic information, cite both the article and the archived
software version.

Suggested software citation before publication of the article:

> Vargas-Cardona, H. D., and Torres, C. (2026). *dMRI interpolation and
> tensor-process methods* (Version 1.0.0) [Computer software].

## Licensing

- Source code and documentation: MIT License; see `LICENSE`.
- Files under `data/`: Creative Commons Attribution 4.0 International;
  see `DATA_LICENSE.md`.

## Responsible data use

The separately distributed study is fully anonymized. Users must not attempt participant
re-identification. Any redistributed or modified copy must retain the data
license, attribution, and provenance information.


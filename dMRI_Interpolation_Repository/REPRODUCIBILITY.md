# Reproducibility guide

## Google Colab

1. Open `notebooks/00_All_Methods_Colab.ipynb` in Google Colab.
2. Run the environment-preparation cell.
3. Upload `dMRI_Interpolation_Repository.zip` when prompted.
4. Upload `dMRI_Data.zip` when prompted.
5. Execute the remaining cells in order.

The consolidated notebook validates the core functions before running the
demonstrations. Individual notebooks are available for each methodological
family.

## Local execution

Python 3.10 or later is recommended.

Extract the two packages so that their folders are siblings:

```text
working-directory/
├── dMRI_Interpolation_Repository/
└── dMRI_Data/
```

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_colab.txt
python run_core_tests.py
python run_smoke_tests.py
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Demonstration commands

```bash
python demo_dmri_interpolation.py
python demo_dmri_full_slice.py --method gp
python demo_dmri_full_slice.py --method cnn --cnn-epochs 5
python demo_gwp_tensor.py
python demo_cdp_tensor.py --order 2
python demo_cdp_tensor.py --order 4
python demo_cdp_tensor.py --order 6
```

## Full and short configurations

- The GP and CNN notebooks reconstruct a complete 256-by-256 slice from
  overlapping low-resolution patches. The Colab defaults use 16-by-16 patches,
  a stride of 14, Hann-weighted overlap-add blending, and exact restoration of
  the observed samples.
- The CNN demo uses five epochs per patch as a short runnable configuration.
  Increase `CNN_EPOCHS_PER_PATCH` for longer experiments.
- Use `MATLAB_LEGACY_RANKS` in `cdp_tensor.py` to select the original CDP ranks
  10, 12, and 16. `FAST_RANKS` provides the default Colab configuration.
- Random seeds are fixed where stochastic initialization is used.

## Expected validation

`run_core_tests.py` must report five passing tests covering:

1. sample-aligned nearest-neighbor and bilinear interpolation;
2. complete-image GP patch joining and exact preservation of observed samples;
3. reversible coefficient mappings for tensor orders 2, 4, and 6;
4. positive-definite GWP predictions at interpolated locations; and
5. CDP execution for all supported tensor orders.

`run_smoke_tests.py` additionally verifies the NIfTI dimensions, gradient-table
consistency, DTI design matrix, DTI fitting, and FA calculation.


# External raw dMRI data

The raw diffusion MRI study is intentionally stored outside this GitHub
repository so that all repository files can be uploaded through the GitHub web
interface.

Download and extract `dMRI_Data.zip`, then place the resulting `dMRI_Data`
folder next to `dMRI_Interpolation_Repository`:

```text
working-directory/
├── dMRI_Interpolation_Repository/
└── dMRI_Data/
    ├── TENSOR_2x2x2.nii.gz
    ├── TENSOR_2x2x2.bval
    └── TENSOR_2x2x2.bvec
```

Google Colab notebooks request both ZIP files and arrange them automatically.
The dMRI files are licensed under CC BY 4.0; see `DATA_LICENSE.md`.


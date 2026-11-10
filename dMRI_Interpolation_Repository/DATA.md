# Dataset documentation

## External raw dMRI study

The raw dMRI study is distributed separately as `dMRI_Data.zip`. After
extraction, the following sibling-folder arrangement is expected:

```text
dMRI_Interpolation_Repository/
dMRI_Data/
```

The external `dMRI_Data/` directory contains:

- `TENSOR_2x2x2.nii.gz`: anonymized diffusion MRI study with array dimensions
  `256 × 256 × 33 × 26`;
- `TENSOR_2x2x2.bval`: 26 b-values;
- `TENSOR_2x2x2.bvec`: 26 diffusion-gradient vectors.

The study was acquired at Hospital de Fuenlabrada, Spain, and was previously
used in the doctoral thesis of Hernán Darío Vargas-Cardona. It corresponds to
the reuse of fully anonymized clinical data. The participant authorized the use
of the study at Hospital de Fuenlabrada.

## Tensor fields

The repository directory `data/tensors/` contains the second-, fourth-, and sixth-order
tensor coefficient fields required by the GWP and CDP demonstrations.

| Files | Order | Coefficients per voxel | Intended method |
|---|---:|---:|---|
| `training_data2*.mat` | 2 | 6 | GWP and CDP |
| `training_data4*.mat` | 4 | 15 | CDP |
| `training_data6*.mat` | 6 | 28 | CDP |
| `HOT2_CDP_Real_X4.mat` | 2 | 6 | Reference CDP result |

The tensor files use the MATLAB-compatible channel-first layout
`(coefficients, rows, columns)`. The mapping is implemented and tested in
`tensor_utils.py`.

## Method-domain boundary

- Nearest neighbor, bilinear interpolation, patch GP, and CNN operate on raw
  dMRI intensities.
- GWP operates on second-order tensor fields.
- CDP operates on second-, fourth-, or sixth-order tensor fields.

Raw dMRI must not be passed directly to the GWP or CDP modules.

## License and integrity

The data are distributed under CC BY 4.0; see `DATA_LICENSE.md`. The repository
and external dMRI package each provide their own `SHA256SUMS.txt` file.


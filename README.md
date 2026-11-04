# ThermoRDF final models

This repository contains the final models used for the reported ThermoRDF
results. The release is intentionally restricted to one thermodynamic-state
selection:

- geometry: `range`
- selection: `designed_maximin`
- repetition: `0` (deterministic design)
- selected states: `N=36`
- subset identifier: `d073f2c16c2ae18b2177`

Exactly one final artifact is included for each physical reconstruction:

- `EOS`: density surrogate;
- `IDW`: inverse-distance RDF baseline;
- `Ridge`: regularised linear RDF reconstruction; and
- `MLP`: compact joint reconstruction of the four RDF channels.

The MLP artifact stores the best accepted parameter state used to generate the
reported predictions, not an arbitrary last training epoch. No intermediate
checkpoints, control repetitions, higher-budget models or profiling figures are
included.

## Data

The reference NetCDF dataset is available from Zenodo:

https://doi.org/10.5281/zenodo.23960981

The file `thermordf_master.nc` is not duplicated in this code repository. Its
SHA-256 and size are recorded in `release_manifest.json`; predictions are
refused if a different NetCDF file is supplied.

## Contents

- `code/`: model, selection, inference and evaluation implementation used in
  the completed study;
- `models/`: the four final validated model artifacts and their metadata;
- `sampling/model_jobs.csv`: the single state selection associated with the
  released models;
- `results/`: evaluation rows produced from those same artifacts;
- `provenance/`: the sanitised study configuration, sampling summary and
  completion records;
- `thermordf_predict_final.py`: density and RDF prediction without retraining;
- `thermordf_verify_release.py`: checksum and scope verification.

## Environment

Create or activate a Python environment and install:

```bash
python -m pip install -r requirements.txt
```

## Verify the repository

```bash
python thermordf_verify_release.py
```

Successful verification reports four model artifacts and the subset identifier
shown above.

## Predict one thermodynamic state

Place or reference the Zenodo NetCDF file and run, for example:

```bash
python thermordf_predict_final.py thermordf_master.nc   --temperature 250   --pressure 500   --device cpu   --output prediction_T250_P500.csv
```

Use `--device cuda` for MLP inference on an available CUDA device. The command
creates a CSV containing the radial coordinate and the four RDF channels from
IDW, Ridge and MLP, together with a JSON file containing the predicted density
and provenance. If the requested state is present in the NetCDF grid, the
reference density and RDF columns are added for comparison.

The released models support interpolation only within 200--350 K and
1--3000 bar.

<!-- THERMORDF-PUBLICATION-METADATA:START -->
## Citation and licensing

Citation metadata for the software is provided in `CITATION.cff`. The reference
dataset is available from [Zenodo](https://doi.org/10.5281/zenodo.23960981).

The executable source code is licensed under `LGPL-3.0-or-later`. Trained models,
result tables, provenance records and documentation are licensed under
`CC-BY-4.0`. See `LICENSE_SCOPE.md` for the exact file-level scope.
<!-- THERMORDF-PUBLICATION-METADATA:END -->

### lp_utils
Utilities for BAO Linear Point analyses on RayGal simulation catalogs: loading and filtering catalogs, applying redshift/angular cuts, parsing RayGal catalog filenames (`catalogs.parse_catalog_filename`) and Corrfunc pair-count filenames (`utils.parse_corrfunc_filename`), Basic cosmology computations such as P(k)->xi and functions to construct the infrastructure for LP analysis.

#### Features
- RayGal catalog parsing & metadata extraction
- Corrfunc filename parameter decoding (DD/DR/RR smu outputs)
- Cosmology parameter bundle (h, Omegas, sigma8, w) for reproducibility
- Power spectrum to correlation function utilities
- Simple helpers for constructing analysis inputs

#### Installation
```
python3 -m pip install --upgrade build
python3 -m pip install -e .
```
### cosmo_utils
The repository contains utilities to work with catalogs and Utilities for basic cosmology computations such as distances, volumes, growth factor, growth rate, P(k)-xi etc...

Utilities for cosmological computations and working with catalogs (mostly [Raygal lightcones](https://cosmo.obspm.fr/public-datasets/raygalgroupsims-relativistic-halo-catalogs/))

- Catalog loading/filtering and filename metadata parsing
- Corrfunc pair-count filename decoding and retrieval 
- Cosmology helpers (growth, volume, P(k) I/O)
- Filters and special functions

## Package layout

- Core
  - [`cosmo_utils.utils`](src/cosmo_utils/utils.py) — I/O helpers (JSON, PK), path utilities, Corrfunc filename parsing, result writers 
  - [`cosmo_utils.cosmology`](src/cosmo_utils/cosmology.py) — cosmology calculations (see [`cosmology.Cosmology`](src/cosmo_utils/cosmology.py))
  - [`cosmo_utils.filters_et_functions`](src/cosmo_utils/filters_et_functions.py) — Special functions and useful filters
  - [`cosmo_utils.catalogs`](src/cosmo_utils/catalogs.py) — catalog retrieval, filtering, sampling, and small plotting helpers
  
Configuration files (JSON) are expected under:
- [`src/lp_utils/config`](src/lp_utils/config

## Installation

Using uv:
```bash
# From repository root
uv pip install -e .
```

Using pip:
```bash
python -m pip install -e

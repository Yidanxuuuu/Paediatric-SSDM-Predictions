## Overview

Statistical shape and density model (SSDM)-based prediction of paediatric femoral and tibial shape and density for FE modelling under no imaging or limited imaging.


## Methods Implemented

- `sd_fitting`: shape + density fitting (full/partial input)
- `so_fitting`: shape-only fitting (full/partial input)
- `plsr`: no-imaging prediction using demographics only or demographics + linear bone measurements

## Environment

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Minimal Project Layout

Required runtime files/folders:

- `demo/run_all_methods.py`
- `src/` (all submodules)
- `models/` (PLSR + PCA artifacts)
- `demo/` input example files

Notes:

- `so_fitting` and `sd_fitting` default to `models/<bone>_pc30.npz`
- If you use a custom model path, pass `--ssdm-path`

## Quick Start

Run from repository root.

### 1. Shape-only fitting (`so_fitting`)

```bash
python demo/run_all_methods.py --method so_fitting --bone femur --target-xyzd demo/femur_example1_shape_only.txt
```

Current default behavior for `so_fitting`:

- `fit_comps = 0..29`
- `fit_scale = True`
- `auto_align = True`
- default surface nodes by bone: `femur=5116`, `tibia=4644`

### 2. Shape+density fitting (`sd_fitting`)

```bash
python demo/run_all_methods.py --method sd_fitting --bone tibia --target-xyzd demo/tibia_example1_shape_density.txt
```

### 3. PLSR prediction

Print required feature names first:

```bash
python demo/run_all_methods.py --method plsr --bone femur --strategy demo_plus_bone --print-features
```

If `Sex` is required, use coding: `Female=1, Male=2`.

Batch prediction from CSV:

```bash
python demo/run_all_methods.py --method plsr --bone femur --strategy demo_plus_bone --input-csv demo/femur_example1_demoplusbone_input.csv --output-dir outputs
```

## Input Format

- `so_fitting`: target file must contain at least 3 columns; first three are used as `x,y,z`
- `sd_fitting`: target file must contain exactly 4 columns `x,y,z,d`
- `plsr`: CSV must include all required feature columns for selected `bone + strategy`

## Outputs

Fitting outputs:

- `outputs/recon/<bone>_so_fitting_<target_stem>.txt`
- `outputs/recon/<bone>_sd_fitting_<target_stem>.txt`

PLSR reconstruction outputs:

- written under the selected `--output-dir`

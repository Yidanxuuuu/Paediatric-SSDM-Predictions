# Paediatric-SSDM-Predictions
Statistical shape and density model (SSDM)-based prediction of paediatric femoral and tibial shape and density for FE modelling under no imaging or limited imaging

# Paediatric SSDM Prediction Framework

This repository provides a unified framework for predicting paediatric bone shape and density using Statistical Shape and Density Models (SSDMs) under different input scenarios.

## Methods implemented

- SD-fitting (shape + density, full/partial input)
- SO-fitting (shape only, full/partial input)
- PLSR-based (no imaging, Demographic only or plus linear bone measurements)

## Quick Start

```bash
pip install -r requirements.txt
python demo/run_all_methods.py
# Paediatric SSDM Prediction

Inference pipeline for paediatric femur/tibia prediction and fitting.

Supported methods in `demo/run_all_methods.py`:
- `plsr`: predict PC weights from tabular features, then reconstruct shape+density
- `sd_fitting`: fit shape+density target (x,y,z,d) to SSDM
- `so_fitting`: fit shape-only target (x,y,z) to SSDM (surface fitting)

## 1. Environment

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Minimal Project Layout

Required runtime files/folders:

- `demo/run_all_methods.py`
- `src/` (all submodules)
- `models/` (PLSR + PCA artifacts for `plsr` mode)
- `demo/` input example files

Notes:
- `so_fitting` and `sd_fitting` default to `models/<bone>_pc30.npz`
- If you use custom model paths, pass `--ssdm-path`

## 3. Quick Start

Run from repository root.

### 3.1 Shape-only fitting (`so_fitting`)

```bash
python demo/run_all_methods.py --method so_fitting --bone femur --target-xyzd demo/femur_example1_shape_only.txt 
```

Current default behavior for `so_fitting`:
- `fit_comps = 0..29`
- `fit_scale = True`
- `auto_align = True`
- default surface nodes by bone: `femur=5116`, `tibia=4644`

### 3.2 Shape+density fitting (`sd_fitting`)

```bash
python demo/run_all_methods.py --method sd_fitting --bone tibia --target-xyzd demo/tibia_example1_shape_density.txt 
```

### 3.3 PLSR prediction

Print required feature names first:

```bash
python demo/run_all_methods.py --method plsr --bone femur --strategy demo_plus_bone --print-features
```

If `Sex` is required, use coding: `Female=1, Male=2`.

Batch prediction from CSV:

```bash
python demo/run_all_methods.py --method plsr --bone femur --strategy demo_plus_bone --input-csv demo/femur_example1_demoplusbone_input.csv --output-dir outputs
```

## 4. Input Format

- `so_fitting`: target file must contain at least 3 columns; first three are used as `x,y,z`
- `sd_fitting`: target file must contain exactly 4 columns `x,y,z,d`
- `plsr`: CSV must include all required feature columns for selected `bone + strategy`

## 5. Outputs

Fitting outputs:
- `outputs/recon/<bone>_so_fitting_<target_stem>.txt`
- `outputs/recon/<bone>_sd_fitting_<target_stem>.txt`

PLSR reconstruction outputs:
- written under the selected `--output-dir`

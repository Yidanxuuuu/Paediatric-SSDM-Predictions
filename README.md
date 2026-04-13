# Paediatric-SSDM-Predictions
Statistical shape and density model (SSDM)-based prediction of paediatric femoral and tibial shape and density for FE modelling under no imaging or limited imaging

# Paediatric SSDM Prediction Framework

This repository provides a unified framework for predicting paediatric bone shape and density using Statistical Shape and Density Models (SSDMs) under different input scenarios.

## Methods implemented

- SD-Full (shape + density, full input)
- SD-Partial (shape + density, partial input)
- SO-Full (shape only, full input)
- SO-Partial (shape only, partial input)
- PLSR-based (no imaging, Demographic+bone measurements)
- PC-Reconstruction (upper-bound reference)

## Quick Start

```bash
pip install -r requirements.txt
python demo/run_all_methods.py

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable when running as:
# python demo/run_all_methods.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pca_model import load_ssdm_model
from src.core.reconstruction import (
    reconstruct_vector,
    save_shape_density_txt,
    vector_to_shape_density,
)
from src.methods.plsr import (
    find_bundle_path,
    load_plsr_bundle,
    predict_patient_weights,
    predict_with_manual_input,
)
from src.methods.sd_fitting import run_sd_fitting
from src.methods.so_fitting import run_so_fitting


def _parse_fit_comps(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prediction/fitting methods.")
    parser.add_argument(
        "--method",
        default="plsr",
        choices=["plsr", "sd_fitting", "so_fitting"],
        help="Method to run.",
    )
    parser.add_argument("--bone", required=True, choices=["femur", "tibia"])
    parser.add_argument("--models-dir", default="models")
    parser.add_argument(
        "--strategy",
        default="demo_plus_bone",
        choices=["demo_only", "demo_plus_bone"],
        help="PLSR feature strategy.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="CSV with one or multiple patient rows containing required feature columns.",
    )
    parser.add_argument(
        "--manual-input",
        action="store_true",
        help="Prompt user to type patient features in terminal.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where shape_recon/ and density_recon/ are written.",
    )
    parser.add_argument(
        "--case-id",
        default="predicted_case",
        help="Output file stem for manual-input mode.",
    )
    parser.add_argument(
        "--id-column",
        default="Case_name",
        help="Identifier column in input CSV used for output filenames.",
    )
    parser.add_argument(
        "--weights-output-csv",
        default=None,
        help="Optional path to save predicted PC weights for all CSV rows.",
    )
    parser.add_argument(
        "--print-features",
        action="store_true",
        help="Print required features for selected bone/strategy and exit.",
    )
    parser.add_argument(
        "--target-xyzd",
        default=None,
        help="Path to target file for fitting. sd_fitting uses x,y,z,d; so_fitting uses x,y,z (or first 3 columns).",
    )
    parser.add_argument(
        "--ssdm-path",
        default=None,
        help="Optional model .npz path for sd_fitting/so_fitting. Defaults to models/<bone>_pc30.npz",
    )
    parser.add_argument(
        "--fit-comps",
        default=None,
        help="Comma-separated component indices for sd_fitting/so_fitting (e.g. 0,1,2,...,29).",
    )
    parser.add_argument(
        "--fit-mode",
        default="ts",
        choices=["st", "ts", "2way", "corr"],
        help="Fitting mode for sd_fitting/so_fitting.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Optional sample count for sd_fitting/so_fitting.",
    )
    parser.add_argument(
        "--sd-verbose",
        action="store_true",
        help="Verbose optimisation logging for sd_fitting/so_fitting.",
    )
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument("--pop-size", type=int, default=200)
    parser.add_argument("--n-gen", type=int, default=400)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--mweight",
        type=float,
        default=0.1,
        help="Mahalanobis weight for so_fitting. Defaults to 0.1 (legacy shape-only behavior).",
    )
    parser.add_argument(
        "--surface-nodes",
        type=int,
        default=None,
        help="Optional number of surface nodes for so_fitting. Defaults by bone: femur=5116, tibia=4644.",
    )
    args = parser.parse_args()

    if args.method == "sd_fitting":
        if args.target_xyzd is None:
            raise ValueError("For --method sd_fitting, please provide --target-xyzd.")
        fit_comps = _parse_fit_comps(args.fit_comps)
        result = run_sd_fitting(
            bone=args.bone,
            target_xyzd_path=args.target_xyzd,
            output_dir=args.output_dir,
            ssdm_path=args.ssdm_path,
            fit_comps=fit_comps,
            fit_mode=args.fit_mode,
            fit_scale=True,
            auto_align=True,
            sample=args.sample,
            verbose=args.sd_verbose,
            workers=args.workers,
            pop_size=args.pop_size,
            n_gen=args.n_gen,
            random_seed=args.random_seed,
        )
        print(f"Saved sd_fitting reconstruction: {result.recon_xyzd_path}")
        print(f"RMS: {result.reg_rms:.6f}")
        print(f"Error(shape,density): {result.opt_error.tolist()}")
        return
    if args.method == "so_fitting":
        if args.target_xyzd is None:
            raise ValueError("For --method so_fitting, please provide --target-xyzd.")
        fit_comps = _parse_fit_comps(args.fit_comps)
        result = run_so_fitting(
            bone=args.bone,
            target_xyz_path=args.target_xyzd,
            output_dir=args.output_dir,
            ssm_path=args.ssdm_path,
            fit_comps=fit_comps,
            fit_mode=args.fit_mode,
            fit_scale=True,
            auto_align=True,
            sample=args.sample,
            verbose=args.sd_verbose,
            workers=args.workers,
            mweight=args.mweight,
            surface_nodes=args.surface_nodes,
        )
        print(f"Saved so_fitting reconstruction: {result.recon_xyzd_path}")
        print(f"RMS: {result.reg_rms:.6f}")
        print(f"Mahalanobis: {result.mahalanobis_dist:.6f}")
        return

    models_dir = Path(args.models_dir)
    bundle_path = find_bundle_path(models_dir, args.bone, args.strategy)
    bundle = load_plsr_bundle(bundle_path)

    if args.print_features:
        print("Required features:")
        for feature in bundle["feature_names"]:
            print(f"- {feature}")
        if "Sex" in bundle["feature_names"]:
            print("Sex coding: Female=1, Male=2")
        return

    if args.manual_input:
        pred_weights = predict_with_manual_input(bundle)
        print("Predicted PC weights:")
        print(",".join([f"{w:.8f}" for w in pred_weights]))

        pca_model = load_ssdm_model(models_dir, args.bone)
        recon_vec = reconstruct_vector(pca_model, pred_weights)
        shape_xyz, density = vector_to_shape_density(recon_vec)
        recon_path = save_shape_density_txt(
            shape_xyz,
            density,
            output_dir=args.output_dir,
            stem=f"{args.bone}_{args.strategy}_{args.case_id}",
        )
        print(f"Saved reconstruction file (x,y,z,d): {recon_path}")
        return

    if args.input_csv is None:
        raise ValueError("Provide --input-csv or use --manual-input.")

    x_df = pd.read_csv(args.input_csv)
    feature_names = bundle["feature_names"]
    missing = [c for c in feature_names if c not in x_df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns in CSV: {missing}")

    pca_model = load_ssdm_model(models_dir, args.bone)
    pred_rows: list[dict] = []
    for i, (_, row) in enumerate(x_df.iterrows()):
        case_id = (
            str(row[args.id_column])
            if args.id_column in x_df.columns and pd.notna(row[args.id_column])
            else f"row_{i+1}"
        )
        patient_info = row[feature_names].to_dict()
        pred_weights = predict_patient_weights(bundle, patient_info)

        recon_vec = reconstruct_vector(pca_model, pred_weights)
        shape_xyz, density = vector_to_shape_density(recon_vec)
        recon_path = save_shape_density_txt(
            shape_xyz,
            density,
            output_dir=args.output_dir,
            stem=f"{args.bone}_{args.strategy}_{case_id}",
        )
        print(f"[{i+1}/{len(x_df)}] {case_id}: {recon_path}")

        row_out = {"Case_ID": case_id}
        for j, w in enumerate(pred_weights, start=1):
            row_out[f"PC{j}"] = float(w)
        pred_rows.append(row_out)

    if args.weights_output_csv:
        out_csv = Path(args.weights_output_csv)
    else:
        in_path = Path(args.input_csv)
        out_csv = in_path.with_name(f"{in_path.stem}_predicted_weights.csv")
    pd.DataFrame(pred_rows).to_csv(out_csv, index=False)
    print(f"Saved predicted weights: {out_csv}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

EXPECTED_UNITS = {
    "Age": "years",
    "Height": "cm",
    "Mass": "kg",
    "Sex": "coded integer (e.g., F=1, M=2)",
    "Epicondylar_width": "cm",
    "Femoral_length": "cm",
    "Condylar_width": "cm",
    "Malleolar_width": "cm",
    "Tibial_length": "cm",
}


def load_plsr_bundle(bundle_path: str | Path) -> dict:
    with Path(bundle_path).open("rb") as f:
        return pickle.load(f)


def _vector_from_patient_info(
    bundle: Mapping,
    patient_info: Mapping[str, float] | Sequence[float],
) -> np.ndarray:
    feature_names = bundle["feature_names"]

    if isinstance(patient_info, Mapping):
        missing = [name for name in feature_names if name not in patient_info]
        if missing:
            raise KeyError(f"Missing patient features: {missing}")
        values = [float(patient_info[name]) for name in feature_names]
    else:
        values = [float(v) for v in patient_info]
        if len(values) != len(feature_names):
            raise ValueError(f"Expected {len(feature_names)} features, got {len(values)}.")

    return np.asarray(values, dtype=float).reshape(1, -1)


def predict_patient_weights(
    bundle: Mapping,
    patient_info: Mapping[str, float] | Sequence[float],
) -> np.ndarray:
    x_new = _vector_from_patient_info(bundle, patient_info)
    return bundle["model"].predict(x_new).reshape(-1)


def prompt_patient_info(feature_names: Sequence[str]) -> dict[str, float]:
    patient_info: dict[str, float] = {}
    for name in feature_names:
        while True:
            raw = input(f"Enter {name}: ").strip()
            try:
                patient_info[name] = float(raw)
                break
            except ValueError:
                print("Invalid number. Please enter a numeric value.")
    return patient_info


def predict_with_manual_input(bundle: Mapping) -> np.ndarray:
    patient_info = prompt_patient_info(bundle["feature_names"])
    return predict_patient_weights(bundle, patient_info)


def find_bundle_path(models_dir: str | Path, bone: str, strategy: str) -> Path:
    models_path = Path(models_dir)
    candidates = sorted(models_path.glob(f"plsr_{bone.lower()}_{strategy}_pc*.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No bundle found for bone={bone}, strategy={strategy} in {models_path}"
        )
    return candidates[-1]


def _apply_optional_unit_conversion(
    x_df: pd.DataFrame,
    feature_names: Sequence[str],
    auto_unit_convert: bool,
) -> pd.DataFrame:
    if not auto_unit_convert:
        return x_df

    converted = x_df.copy()
    # Height/lengths: expected cm. If values look like meters, convert to cm.
    for col in ["Height", "Epicondylar_width", "Femoral_length", "Condylar_width", "Malleolar_width", "Tibial_length"]:
        if col in feature_names and col in converted.columns:
            median_val = float(converted[col].median())
            if 0 < median_val < 10:
                converted[col] = converted[col] * 100.0
            elif median_val > 50:
                converted[col] = converted[col] / 10.0

    # Mass: expected kg. If values look like grams, convert to kg.
    if "Mass" in feature_names and "Mass" in converted.columns:
        median_mass = float(converted["Mass"].median())
        if median_mass > 500:
            converted["Mass"] = converted["Mass"] / 1000.0

    return converted


def _predict_from_csv(
    bundle: Mapping,
    input_csv: str | Path,
    id_column: str,
    auto_unit_convert: bool,
) -> pd.DataFrame:
    x_df = pd.read_csv(input_csv)
    feature_names = bundle["feature_names"]
    missing = [c for c in feature_names if c not in x_df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns in CSV: {missing}")

    x_features = x_df[feature_names].apply(pd.to_numeric, errors="raise")
    x_features = _apply_optional_unit_conversion(
        x_features,
        feature_names=feature_names,
        auto_unit_convert=auto_unit_convert,
    )
    pred = bundle["model"].predict(x_features.values)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)

    if id_column in x_df.columns:
        case_ids = x_df[id_column].astype(str).tolist()
    else:
        case_ids = [f"row_{i+1}" for i in range(len(x_df))]

    pred_cols = [f"PC{i+1}" for i in range(pred.shape[1])]
    out_df = pd.DataFrame(pred, columns=pred_cols)
    out_df.insert(0, "Case_ID", case_ids)
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description="PLSR inference from saved .pkl bundles.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--bone", required=True, choices=["femur", "tibia"])
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["demo_only", "demo_plus_bone"],
        help="Feature strategy used by the trained bundle.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="CSV containing one or multiple patients with required feature columns.",
    )
    parser.add_argument(
        "--manual-input",
        action="store_true",
        help="Prompt feature values one-by-one in terminal.",
    )
    parser.add_argument(
        "--print-features",
        action="store_true",
        help="Print required feature names and exit.",
    )
    parser.add_argument(
        "--id-column",
        default="Case_name",
        help="Optional case identifier column in input CSV.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path for batch prediction. Default: <input>_predicted_weights.csv",
    )
    parser.add_argument(
        "--auto-unit-convert",
        action="store_true",
        help="Auto-convert common units (e.g., m->cm, mm->cm, g->kg) if detected.",
    )
    args = parser.parse_args()

    bundle_path = find_bundle_path(args.models_dir, args.bone, args.strategy)
    bundle = load_plsr_bundle(bundle_path)
    feature_names = bundle["feature_names"]

    if args.print_features:
        print("Required features:")
        for f in feature_names:
            unit = EXPECTED_UNITS.get(f, "check training-unit consistency")
            print(f"- {f} [{unit}]")
        return

    if args.manual_input:
        pred_weights = predict_with_manual_input(bundle)
        print("Predicted PC weights:")
        print(",".join([f"{w:.8f}" for w in pred_weights]))
        return

    if args.input_csv is None:
        raise ValueError("Provide --input-csv or use --manual-input.")

    pred_df = _predict_from_csv(
        bundle=bundle,
        input_csv=args.input_csv,
        id_column=args.id_column,
        auto_unit_convert=args.auto_unit_convert,
    )
    output_csv = Path(args.output_csv) if args.output_csv else Path(args.input_csv).with_name(
        f"{Path(args.input_csv).stem}_predicted_weights.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_csv, index=False)
    print(f"Predicted {len(pred_df)} cases.")
    print(f"Saved: {output_csv}")
    print(pred_df.head(min(5, len(pred_df))).to_string(index=False))


if __name__ == "__main__":
    main()

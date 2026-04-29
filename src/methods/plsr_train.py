from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.cross_decomposition import PLSRegression


DEFAULT_MODEL_DIR_BY_BONE = {
    "femur": "SSDM_femur",
    "tibia": "SSDM_tibia",
}

DEFAULT_FEATURES_CSV_BY_BONE = {
    "femur": [
        "features/femur_all_measurements_and_factors.csv",
        "features/Femur_all_measurements_and_factors.csv",
        "features/Femur_demographics.csv",
    ],
    "tibia": [
        "features/tibia_all_measurements_and_factors.csv",
        "features/Tibia_all_measurements_and_factors.csv",
        "features/Tibia_demographics.csv",
    ],
}

DEFAULT_DEMOGRAPHIC_COLUMNS = ("Age", "Height", "Mass", "Sex")


def _resolve_features_csv_path(
    root_dir: str | Path,
    bone: str,
    features_csv: Optional[str | Path] = None,
) -> Path:
    if features_csv is not None:
        return Path(features_csv)

    root_path = Path(root_dir)
    bone_key = bone.lower()
    candidates = DEFAULT_FEATURES_CSV_BY_BONE.get(bone_key, [])
    for rel_path in candidates:
        candidate = root_path / rel_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find features CSV for bone={bone}. Tried: {candidates}"
    )


def _resolve_model_mat_path(
    root_dir: str | Path,
    bone: str,
    model_mat_path: Optional[str | Path] = None,
) -> Path:
    if model_mat_path is not None:
        return Path(model_mat_path)

    bone_key = bone.lower()
    if bone_key not in DEFAULT_MODEL_DIR_BY_BONE:
        raise ValueError(f"Unknown bone '{bone}'. Please provide model_mat_path explicitly.")

    return Path(root_dir) / DEFAULT_MODEL_DIR_BY_BONE[bone_key] / f"{bone_key}.mat"


def _load_feature_matrix(
    features_csv: str | Path,
    case_name_column: str = "Case_name",
) -> tuple[pd.DataFrame, list[str]]:
    features_df = pd.read_csv(features_csv)
    drop_cols = [c for c in [case_name_column, "Filename"] if c in features_df.columns]
    x_df = features_df.drop(columns=drop_cols) if drop_cols else features_df.copy()
    x_df = x_df.apply(pd.to_numeric, errors="coerce")
    x_df = x_df.dropna(axis=1, how="all")
    if x_df.isna().any().any():
        bad_cols = x_df.columns[x_df.isna().any()].tolist()
        raise ValueError(f"Found non-numeric or missing values in feature columns: {bad_cols}")
    return x_df, x_df.columns.tolist()


def _load_projected_weights(
    model_mat_path: str | Path,
    n_samples: int,
    pc_count: int,
) -> np.ndarray:
    mat = sio.loadmat(model_mat_path)
    if "projectedWeights" not in mat:
        raise KeyError(f"'projectedWeights' not found in {model_mat_path}")

    projected_weights = np.asarray(mat["projectedWeights"])
    if projected_weights.ndim != 2:
        raise ValueError(f"projectedWeights must be 2D, got shape {projected_weights.shape}")

    # Keep consistent with legacy LOO pipeline:
    # Y = mat['projectedWeights'].T[:, :PC]
    y = projected_weights.T
    if y.shape[0] != n_samples:
        # Fallback for unexpected files where projectedWeights is already [samples, modes]
        if projected_weights.shape[0] == n_samples:
            y = projected_weights
        else:
            raise ValueError(
                "Unable to align projectedWeights with features: "
                f"projectedWeights shape={projected_weights.shape}, n_samples={n_samples}"
            )

    if y.shape[1] < pc_count:
        raise ValueError(
            f"Requested pc_count={pc_count} but only {y.shape[1]} modes are available."
        )
    return y[:, :pc_count]


def fit_plsr(
    bone: str,
    root_dir: str | Path,
    features_csv: Optional[str | Path] = None,
    model_mat_path: Optional[str | Path] = None,
    pc_count: int = 20,
    n_components: int = 6,
    strategy: str = "demo_plus_bone",
    demographic_columns: Sequence[str] = DEFAULT_DEMOGRAPHIC_COLUMNS,
) -> dict:
    root_path = Path(root_dir)
    bone_key = bone.lower()

    features_csv = _resolve_features_csv_path(root_path, bone_key, features_csv)

    model_mat_path = _resolve_model_mat_path(root_path, bone_key, model_mat_path)

    x_df, feature_names = _load_feature_matrix(features_csv)
    if strategy == "demo_only":
        selected = [c for c in demographic_columns if c in x_df.columns]
        if not selected:
            raise ValueError(
                f"None of demographic columns found in {features_csv}: {list(demographic_columns)}"
            )
        x_df = x_df[selected]
        feature_names = selected
    elif strategy != "demo_plus_bone":
        raise ValueError(f"Unknown strategy '{strategy}'. Use demo_only or demo_plus_bone.")

    y = _load_projected_weights(model_mat_path, n_samples=len(x_df), pc_count=int(pc_count))

    max_components = min(x_df.shape[0] - 1, x_df.shape[1], y.shape[1])
    if max_components < 1:
        raise ValueError("Invalid data dimensions for PLSR.")
    n_components = min(int(n_components), max_components)

    model = PLSRegression(n_components=n_components, scale=True)
    model.fit(x_df.values, y)

    return {
        "bone": bone_key,
        "strategy": strategy,
        "feature_names": feature_names,
        "pc_count": int(pc_count),
        "n_components": int(n_components),
        "features_csv": str(features_csv),
        "model_mat_path": str(model_mat_path),
        "model": model,
    }


def save_plsr_bundle(bundle: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(bundle, f)
    return output_path


def export_public_ssdm_pc(
    bone: str,
    root_dir: str | Path,
    output_dir: str | Path = "models",
    pc_count: int = 30,
    model_mat_path: Optional[str | Path] = None,
) -> dict[str, Path]:
    bone_key = bone.lower()
    root_path = Path(root_dir)
    mat_path = _resolve_model_mat_path(root_path, bone_key, model_mat_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = sio.loadmat(mat_path)
    required_keys = ("mean", "modes", "SD", "weights")
    missing = [k for k in required_keys if k not in mat]
    if missing:
        raise KeyError(f"Missing keys in {mat_path}: {missing}")

    mean = np.asarray(mat["mean"], dtype=np.float64).reshape(-1)
    modes = np.asarray(mat["modes"], dtype=np.float64)
    sd = np.asarray(mat["SD"], dtype=np.float64).reshape(-1)
    weights = np.asarray(mat["weights"], dtype=np.float64).reshape(-1)

    if modes.ndim != 2:
        raise ValueError(f"modes must be 2D, got shape {modes.shape}")
    if modes.shape[0] != mean.shape[0]:
        raise ValueError(f"mean/modes mismatch: mean={mean.shape}, modes={modes.shape}")

    k = min(int(pc_count), modes.shape[1], weights.shape[0])
    modes_k = modes[:, :k]
    weights_k = weights[:k]

    npz_path = out_dir / f"{bone_key}_pc{k}.npz"
    mean_path = out_dir / f"{bone_key}_mean_pc{k}.npy"

    np.savez(
        npz_path,
        mean=mean,
        weights=weights_k,
        modes=modes_k,
        SD=sd,
        pc_count=np.int32(k),
    )
    np.save(mean_path, mean)

    return {"npz": npz_path, "mean_npy": mean_path}


def fit_and_save_public_plsr(
    bone: str,
    root_dir: str | Path,
    output_dir: str | Path = "models",
    features_csv: Optional[str | Path] = None,
    model_mat_path: Optional[str | Path] = None,
    pc_count: int = 30,
    n_components: int = 6,
    strategy: str = "demo_plus_bone",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = fit_plsr(
        bone=bone,
        root_dir=root_dir,
        features_csv=features_csv,
        model_mat_path=model_mat_path,
        pc_count=pc_count,
        n_components=n_components,
        strategy=strategy,
    )
    return save_plsr_bundle(
        bundle,
        out_dir / f"plsr_{bone.lower()}_{strategy}_pc{bundle['pc_count']}.pkl",
    )

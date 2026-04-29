from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PCAModel:
    mean: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues: np.ndarray
    sd: np.ndarray | None = None

    @property
    def pc_count(self) -> int:
        return int(self.eigenvectors.shape[1])


def _load_from_npz(npz_path: Path) -> PCAModel:
    data = np.load(npz_path, allow_pickle=True)
    mean = np.asarray(data["mean"], dtype=np.float64).reshape(-1)

    if "eigenvectors" in data.files:
        eigenvectors = np.asarray(data["eigenvectors"], dtype=np.float64)
    else:
        eigenvectors = np.asarray(data["modes"], dtype=np.float64)

    if "eigenvalues" in data.files:
        eigenvalues = np.asarray(data["eigenvalues"], dtype=np.float64).reshape(-1)
    else:
        eigenvalues = np.asarray(data["weights"], dtype=np.float64).reshape(-1)

    sd = None
    if "SD" in data.files:
        sd = np.asarray(data["SD"], dtype=np.float64).reshape(-1)

    return PCAModel(
        mean=mean,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        sd=sd,
    )


def load_ssdm_model(models_dir: str | Path, bone: str) -> PCAModel:
    """
    Load SSDM PCA basis for a bone.

    Supported formats:
    1) models/{bone}_ssdm/mean.npy + eigenvectors.npy + eigenvalues.npy
    2) models/{bone}_pc*.npz (legacy/public export format)
    """
    models_path = Path(models_dir)
    bone_key = bone.lower()

    folder_path = models_path / f"{bone_key}_ssdm"
    mean_npy = folder_path / "mean.npy"
    eigvec_npy = folder_path / "eigenvectors.npy"
    eigval_npy = folder_path / "eigenvalues.npy"
    if mean_npy.exists() and eigvec_npy.exists() and eigval_npy.exists():
        sd_path = folder_path / "sd.npy"
        sd = np.load(sd_path) if sd_path.exists() else None
        return PCAModel(
            mean=np.load(mean_npy),
            eigenvectors=np.load(eigvec_npy),
            eigenvalues=np.load(eigval_npy),
            sd=sd,
        )

    candidates = sorted(models_path.glob(f"{bone_key}_pc*.npz"))
    if not candidates:
        raise FileNotFoundError(
            f"No SSDM basis found for bone={bone_key} in {models_path}"
        )
    return _load_from_npz(candidates[-1])

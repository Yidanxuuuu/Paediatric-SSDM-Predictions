from __future__ import annotations

from pathlib import Path

import numpy as np

from src.core.pca_model import PCAModel


def reconstruct_vector(model: PCAModel, pc_weights: np.ndarray) -> np.ndarray:
    w = np.asarray(pc_weights, dtype=np.float64).reshape(-1)
    k = min(len(w), model.eigenvectors.shape[1])
    f_reduced = model.eigenvectors[:, :k] @ w[:k]
    if model.sd is None:
        raise ValueError("SD is missing in PCA model. SD-normalized reconstruction is required.")
    if model.sd.shape[0] != model.mean.shape[0]:
        raise ValueError(
            f"SD/mean length mismatch: {model.sd.shape[0]} vs {model.mean.shape[0]}"
        )
    return model.mean + f_reduced * model.sd


def vector_to_shape_density(
    reconstructed_vector: np.ndarray,
    density_floor: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(reconstructed_vector, dtype=np.float64).reshape(-1)
    if v.shape[0] % 4 != 0:
        raise ValueError(
            f"Reconstructed vector length must be divisible by 4, got {v.shape[0]}"
        )
    reshaped = v.reshape(-1, 4)
    shape_xyz = reshaped[:, :3]
    density = np.maximum(reshaped[:, 3], density_floor)
    return shape_xyz, density


def save_shape_density_txt(
    shape_xyz: np.ndarray,
    density: np.ndarray,
    output_dir: str | Path,
    stem: str,
) -> Path:
    out_dir = Path(output_dir)
    recon_dir = out_dir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)

    if shape_xyz.shape[0] != density.shape[0]:
        raise ValueError(
            f"Shape/density row mismatch: {shape_xyz.shape[0]} vs {density.shape[0]}"
        )

    xyzd = np.column_stack([shape_xyz, density])
    recon_path = recon_dir / f"{stem}.txt"
    np.savetxt(recon_path, xyzd, delimiter=",")
    return recon_path

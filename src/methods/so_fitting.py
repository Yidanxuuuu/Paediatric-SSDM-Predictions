from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.fitting.legacy_so_adapter import fit_ssm_to_target


def default_ssm_path(project_root: str | Path, bone: str) -> Path:
    bone_key = bone.lower()
    if bone_key not in {"femur", "tibia"}:
        raise ValueError(f"Unsupported bone: {bone}")
    return Path(project_root) / "models" / f"{bone_key}_pc30.npz"


def default_surface_nodes(bone: str) -> int:
    bone_key = bone.lower()
    if bone_key == "femur":
        return 5116
    if bone_key == "tibia":
        return 4644
    raise ValueError(f"Unsupported bone: {bone}")


def load_target_xyz(target_path: str | Path) -> np.ndarray:
    arr = np.loadtxt(target_path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 3:
        raise ValueError(
            f"Target file must have at least 3 columns (x,y,z). Got shape: {arr.shape}"
        )
    return arr[:, :3].astype(np.float64)


@dataclass
class SOOptimisationConfig:
    fit_comps: Sequence[int]
    fit_mode: str = "ts"  # st, ts, 2way, corr
    fit_scale: bool = True
    auto_align: bool = True
    sample: Optional[int] = None
    verbose: bool = False
    workers: int = 1
    mweight: float = 0.1
    surface_nodes: Optional[int] = None


@dataclass
class SOOptimisationResult:
    recon_xyzd: np.ndarray
    reg_rms: float
    opt_params: np.ndarray
    opt_error: np.ndarray
    init_transform: np.ndarray
    mahalanobis_dist: float


@dataclass
class SOFittingResult:
    recon_xyzd_path: Path
    reg_rms: float
    opt_error: np.ndarray
    opt_params: np.ndarray
    mahalanobis_dist: float


def run_so_optimisation(
    target_xyz: np.ndarray,
    ssm_path: str | Path,
    config: SOOptimisationConfig,
) -> SOOptimisationResult:
    if target_xyz.ndim != 2 or target_xyz.shape[1] != 3:
        raise ValueError(f"target_xyz must be Nx3, got {target_xyz.shape}")

    surface_nodes = config.surface_nodes if config.surface_nodes is not None else int(target_xyz.shape[0])

    legacy_result = fit_ssm_to_target(
        target_xyz=target_xyz,
        ssm_path=ssm_path,
        fit_comps=config.fit_comps,
        fit_mode=config.fit_mode,
        fit_scale=config.fit_scale,
        auto_align=config.auto_align,
        sample=config.sample,
        verbose=config.verbose,
        workers=config.workers,
        mweight=config.mweight,
        surface_nodes=surface_nodes,
    )

    return SOOptimisationResult(
        recon_xyzd=legacy_result.recon_xyzd,
        reg_rms=legacy_result.reg_rms,
        opt_params=legacy_result.opt_params,
        opt_error=legacy_result.opt_error,
        init_transform=legacy_result.init_transform,
        mahalanobis_dist=legacy_result.mahalanobis_dist,
    )


def run_so_fitting(
    bone: str,
    target_xyz_path: str | Path,
    output_dir: str | Path,
    ssm_path: Optional[str | Path] = None,
    fit_comps: Optional[Sequence[int]] = None,
    fit_mode: str = "ts",
    fit_scale: bool = True,
    auto_align: bool = True,
    sample: Optional[int] = None,
    verbose: bool = False,
    workers: int = 1,
    mweight: float = 0.1,
    surface_nodes: Optional[int] = None,
) -> SOFittingResult:
    project_root = Path(__file__).resolve().parents[2]
    ssm_file = Path(ssm_path) if ssm_path else default_ssm_path(project_root, bone)
    target_xyz = load_target_xyz(target_xyz_path)
    resolved_surface_nodes = surface_nodes if surface_nodes is not None else default_surface_nodes(bone)

    if fit_comps is None:
        fit_comps = list(range(30))

    config = SOOptimisationConfig(
        fit_comps=list(fit_comps),
        fit_mode=fit_mode,
        fit_scale=fit_scale,
        auto_align=auto_align,
        sample=sample,
        verbose=verbose,
        workers=workers,
        mweight=mweight,
        surface_nodes=resolved_surface_nodes,
    )
    result = run_so_optimisation(
        target_xyz=target_xyz,
        ssm_path=ssm_file,
        config=config,
    )

    out_dir = Path(output_dir) / "recon"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_stem = Path(target_xyz_path).stem
    out_path = out_dir / f"{bone.lower()}_so_fitting_{target_stem}.txt"
    np.savetxt(out_path, result.recon_xyzd, delimiter=",")

    return SOFittingResult(
        recon_xyzd_path=out_path,
        reg_rms=result.reg_rms,
        opt_error=result.opt_error,
        opt_params=result.opt_params,
        mahalanobis_dist=result.mahalanobis_dist,
    )

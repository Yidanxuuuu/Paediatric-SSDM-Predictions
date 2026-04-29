from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from src.fitting.legacy_sd_adapter import fit_ssdm_to_target


def default_ssdm_path(project_root: str | Path, bone: str) -> Path:
    bone_key = bone.lower()
    if bone_key not in {"femur", "tibia"}:
        raise ValueError(f"Unsupported bone: {bone}")
    return Path(project_root) / "models" / f"{bone_key}_pc30.npz"


def load_target_xyzd(target_path: str | Path) -> np.ndarray:
    arr = np.loadtxt(target_path, delimiter=",")
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 4:
        raise ValueError(
            f"Target file must have 4 columns (x,y,z,d). Got shape: {arr.shape}"
        )
    return arr.astype(np.float64)


@dataclass
class SDFittingResult:
    recon_xyzd_path: Path
    reg_rms: float
    opt_error: np.ndarray
    opt_params: np.ndarray


@dataclass
class SDOptimisationConfig:
    fit_comps: Sequence[int]
    fit_mode: str = "st"  # st, ts, 2way, corr
    fit_scale: bool = True
    auto_align: bool = True
    sample: Optional[int] = None
    verbose: bool = False
    workers: int = -1
    pop_size: int = 200
    n_gen: int = 400
    random_seed: int = 42


@dataclass
class SDOptimisationResult:
    recon_xyzd: np.ndarray
    reg_rms: float
    opt_params: np.ndarray
    opt_error: np.ndarray
    pareto_solutions: list[Any]
    init_transform: np.ndarray


def run_sd_optimisation(
    target_xyzd: np.ndarray,
    ssdm_path: str | Path,
    config: SDOptimisationConfig,
) -> SDOptimisationResult:
    """
    Wrapper around legacy shape-density optimisation for sd_fitting usage.

    Parameters
    ----------
    target_xyzd
        Nx4 array: x, y, z, density.
    ssdm_path
        Path to SSDM model file (.pc/.pc.npz) compatible with legacy PCA loader.
    config
        Optimisation settings.
    """
    if target_xyzd.ndim != 2 or target_xyzd.shape[1] != 4:
        raise ValueError(f"target_xyzd must be Nx4, got {target_xyzd.shape}")

    legacy_result = fit_ssdm_to_target(
        target_xyzd=target_xyzd,
        ssdm_path=ssdm_path,
        fit_comps=config.fit_comps,
        fit_mode=config.fit_mode,
        fit_scale=config.fit_scale,
        auto_align=config.auto_align,
        sample=config.sample,
        verbose=config.verbose,
        workers=config.workers,
        pop_size=config.pop_size,
        n_gen=config.n_gen,
        random_seed=config.random_seed,
    )

    return SDOptimisationResult(
        recon_xyzd=legacy_result.recon_xyzd,
        reg_rms=legacy_result.reg_rms,
        opt_params=legacy_result.opt_params,
        opt_error=legacy_result.opt_error,
        pareto_solutions=legacy_result.pareto_solutions,
        init_transform=legacy_result.init_transform,
    )


def run_sd_fitting(
    bone: str,
    target_xyzd_path: str | Path,
    output_dir: str | Path,
    ssdm_path: Optional[str | Path] = None,
    fit_comps: Optional[Sequence[int]] = None,
    fit_mode: str = "ts",
    fit_scale: bool = True,
    auto_align: bool = True,
    sample: Optional[int] = None,
    verbose: bool = False,
    workers: int = -1,
    pop_size: int = 200,
    n_gen: int = 400,
    random_seed: int = 42,
) -> SDFittingResult:
    project_root = Path(__file__).resolve().parents[2]
    ssdm_file = Path(ssdm_path) if ssdm_path else default_ssdm_path(project_root, bone)
    target_xyzd = load_target_xyzd(target_xyzd_path)

    if fit_comps is None:
        fit_comps = list(range(30))

    config = SDOptimisationConfig(
        fit_comps=list(fit_comps),
        fit_mode=fit_mode,
        fit_scale=fit_scale,
        auto_align=auto_align,
        sample=sample,
        verbose=verbose,
        workers=workers,
        pop_size=pop_size,
        n_gen=n_gen,
        random_seed=random_seed,
    )
    result = run_sd_optimisation(
        target_xyzd=target_xyzd,
        ssdm_path=ssdm_file,
        config=config,
    )

    out_dir = Path(output_dir) / "recon"
    out_dir.mkdir(parents=True, exist_ok=True)
    target_stem = Path(target_xyzd_path).stem
    out_path = out_dir / f"{bone.lower()}_sd_fitting_{target_stem}.txt"
    np.savetxt(out_path, result.recon_xyzd, delimiter=",")

    return SDFittingResult(
        recon_xyzd_path=out_path,
        reg_rms=result.reg_rms,
        opt_error=result.opt_error,
        opt_params=result.opt_params,
    )

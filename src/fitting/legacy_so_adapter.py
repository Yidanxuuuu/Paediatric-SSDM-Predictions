from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from gias3.common import transform3D
from scipy.optimize import leastsq
from scipy.spatial import cKDTree

from src.fitting.legacy_sd_adapter import (
    PrincipalComponentsLite,
    _load_principal_components,
    fit_data_rigid_dpep,
    fit_data_rigid_scale_dpep,
    r2c14,
)


@dataclass(frozen=True)
class LegacySOFitResult:
    recon_xyzd: np.ndarray
    reg_rms: float
    opt_params: np.ndarray
    opt_error: np.ndarray
    init_transform: np.ndarray
    mahalanobis_dist: float


def _sample_data(data: np.ndarray, n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("N must be > 1")
    if n > len(data):
        return data
    i = np.linspace(0, len(data) - 1, n).astype(int)
    return data[i, :]


def r2c13(x_recon: np.ndarray) -> np.ndarray:
    return x_recon.reshape((-1, 3))


def mahalanobis(x: np.ndarray) -> float:
    return float(np.sqrt(np.multiply(x, x).sum()))


def fit_ssm_to_3d_points(
    data: np.ndarray,
    ssm: PrincipalComponentsLite,
    fit_comps: list[int],
    fit_mode: str,
    *,
    fit_inds: Optional[Union[list, np.ndarray]] = None,
    mw: float = 0.0,
    init_t: Optional[np.ndarray] = None,
    fit_scale: bool = False,
    ftol: float = 1e-6,
    sample: Optional[int] = None,
    recon2coords=None,
    verbose: bool = False,
    workers: int = 1,
    surface_nodes: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, float, float]]:
    if recon2coords is None:
        recon2coords = r2c13

    if init_t is None:
        init_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        if fit_scale:
            init_t = np.hstack([init_t, 1.0])
    else:
        init_t = np.array(init_t)

    if fit_scale:
        if len(init_t) != 7:
            raise ValueError(f"Expected init_t length 7 when fit_scale=True, got {len(init_t)}")
    else:
        if len(init_t) != 6:
            raise ValueError(f"Expected init_t length 6 when fit_scale=False, got {len(init_t)}")

    if sample is not None:
        data = _sample_data(data, sample)

    def _recon_no_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        recon = ssm.reconstruct(
            ssm.getWeightsBySD(fit_comps, x[6:]), fit_comps
        )
        recon_shape_xyz = r2c14(recon)[:, :3]
        recon_density = r2c14(recon)[:, 3]
        recon_shape_xyz = transform3D.transformRigid3DAboutCoM(recon_shape_xyz, x[:6])
        recon_pts = np.hstack([recon_shape_xyz, recon_density.reshape(-1, 1)])
        mdist = mahalanobis(x[6:])
        return recon_pts, recon_shape_xyz, mdist, 1.0

    def _recon_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        recon = ssm.reconstruct(
            ssm.getWeightsBySD(fit_comps, x[7:]), fit_comps
        )
        recon_shape_xyz = r2c14(recon)[:, :3]
        recon_density = r2c14(recon)[:, 3]
        recon_shape_xyz = transform3D.transformRigidScale3DAboutCoM(recon_shape_xyz, x[:7])
        recon_pts = np.hstack([recon_shape_xyz, recon_density.reshape(-1, 1)])
        mdist = mahalanobis(x[7:])
        return recon_pts, recon_shape_xyz, mdist, float(x[6])

    _recon = _recon_scale if fit_scale else _recon_no_scale

    targ_tree = cKDTree(data)

    def _dist_sptp(recon_pts: np.ndarray, m: float) -> np.ndarray:
        return targ_tree.query(recon_pts, eps=1e-9, workers=workers)[0] + mw * m

    def _dist_tpsp(recon_pts: np.ndarray, m: float) -> np.ndarray:
        recon_tree = cKDTree(recon_pts)
        return recon_tree.query(data, eps=1e-9, workers=workers)[0] + mw * m

    def _dist_2way(recon_pts: np.ndarray, m: float) -> np.ndarray:
        recon_tree = cKDTree(recon_pts)
        d_sptp = targ_tree.query(recon_pts, eps=1e-9, workers=workers)[0]
        d_tpsp = recon_tree.query(data, eps=1e-9, workers=workers)[0]
        return np.hstack([d_sptp, d_tpsp]) + mw * m

    def _dist_corr(recon_pts: np.ndarray, m: float) -> np.ndarray:
        return np.sqrt(((data - recon_pts) ** 2.0).sum(1))

    fit_modes_map = {
        "st": _dist_sptp,
        "ts": _dist_tpsp,
        "2way": _dist_2way,
        "corr": _dist_corr,
    }
    if fit_mode not in fit_modes_map:
        raise ValueError(f"invalid fit mode {fit_mode}")
    _dist = fit_modes_map[fit_mode]

    def _obj_no_ldmks(x: np.ndarray) -> np.ndarray:
        recon_data, recon_shape_xyz, mdist, _ = _recon(x)
        if fit_inds is not None:
            recon_shape_xyz = recon_shape_xyz[fit_inds, :]
        elif surface_nodes is not None:
            recon_shape_xyz = recon_shape_xyz[:surface_nodes, :]
        err = _dist(recon_shape_xyz, mdist)
        if verbose:
            print(f"\robj rms:{np.sqrt(err.mean())}", end="", flush=True)
        return err

    _obj = _obj_no_ldmks

    x0 = np.hstack([init_t, np.zeros(len(fit_comps), dtype=float)])
    # Keep SciPy's default maxfev, matching original shapeonly_model behavior.
    x_opt = leastsq(_obj, x0, ftol=ftol)[0]

    recon_data_opt, recon_shape_xyz_opt, mdist_opt, _ = _recon(x_opt)
    err_opt = _obj(x_opt)
    dist_opt_rms = np.sqrt((_dist(recon_shape_xyz_opt, 0.0) ** 2.0).mean())

    return x_opt, recon_data_opt, (err_opt, float(dist_opt_rms), float(mdist_opt))


def fit_ssm_to_target(
    target_xyz: np.ndarray,
    ssm_path: str | Path,
    *,
    fit_comps: Sequence[int],
    fit_mode: str,
    fit_scale: bool,
    auto_align: bool,
    sample: Optional[int],
    verbose: bool,
    workers: int,
    mweight: float,
    surface_nodes: Optional[int],
) -> LegacySOFitResult:
    ssm = _load_principal_components(ssm_path)

    full_mean = ssm.getMean().reshape((-1, 4))
    if surface_nodes is None:
        surface_nodes = int(target_xyz.shape[0])
    source_points = full_mean[:surface_nodes, :3]
    target_points = target_xyz

    src_span = np.linalg.norm(source_points.max(axis=0) - source_points.min(axis=0))
    trg_span = np.linalg.norm(target_points.max(axis=0) - target_points.min(axis=0))
    scale_factor = float(trg_span / src_span) if src_span > 0 else 1.0

    if auto_align:
        init_rot = np.deg2rad((0.0, 0.0, 0.0))
        init_trans = target_points.mean(axis=0) - source_points.mean(axis=0)
        t0 = np.hstack([init_trans, init_rot])
        reg1_t, source_points_reg1, _ = fit_data_rigid_dpep(
            source_points,
            target_points,
            xtol=1e-5,
            sample=surface_nodes,
            t0=t0,
            output_errors=1,
        )
        if fit_scale:
            reg2_t, _, _ = fit_data_rigid_scale_dpep(
                source_points_reg1,
                target_points,
                xtol=1e-5,
                sample=surface_nodes,
                t0=np.hstack([reg1_t[:6], scale_factor]),
                output_errors=1,
            )
        else:
            reg2_t = reg1_t
    else:
        reg2_t = np.zeros(6)
        if fit_scale:
            reg2_t = np.hstack([reg2_t, scale_factor])

    x_opt, recon_xyzd, (err_opt, reg_rms, reg_mdist) = fit_ssm_to_3d_points(
        target_points,
        ssm,
        list(fit_comps),
        fit_mode,
        mw=mweight,
        init_t=np.asarray(reg2_t, dtype=float),
        fit_scale=fit_scale,
        ftol=1e-6,
        sample=surface_nodes,
        recon2coords=r2c13,
        verbose=verbose,
        workers=workers,
        surface_nodes=surface_nodes,
    )

    return LegacySOFitResult(
        recon_xyzd=np.asarray(recon_xyzd),
        reg_rms=float(reg_rms),
        opt_params=np.asarray(x_opt),
        opt_error=np.asarray(err_opt),
        init_transform=np.asarray(reg2_t, dtype=float),
        mahalanobis_dist=float(reg_mdist),
    )

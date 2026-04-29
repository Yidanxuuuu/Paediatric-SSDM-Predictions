from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence
from functools import partial
import multiprocessing
import random

import numpy as np
from deap import base, creator, tools, algorithms
from gias3.common import transform3D
from scipy.optimize import leastsq
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LegacySDFitResult:
    recon_xyzd: np.ndarray
    reg_rms: float
    opt_params: np.ndarray
    opt_error: np.ndarray
    pareto_solutions: list[Any]
    init_transform: np.ndarray


@dataclass
class ParetoSolution:
    shape_error: float
    density_error: float
    parameters: np.ndarray
    recon_points: np.ndarray
    recon_density: np.ndarray


# Shared state for multiprocessing evaluation (mirrors legacy shapemodel behavior).
_shared_targ_tree: Optional[cKDTree] = None
_shared_ssdm: Optional[PrincipalComponentsLite] = None
_shared_data_xyz: Optional[np.ndarray] = None
_shared_data_density: Optional[np.ndarray] = None


class PrincipalComponentsLite:
    """
    Minimal local replacement of legacy shapedensitymodel.PCA.PrincipalComponents
    for inference-only usage in sd_fitting.
    """

    def __init__(
        self,
        *,
        mean: np.ndarray,
        weights: np.ndarray,
        modes: np.ndarray,
        sd: np.ndarray,
    ) -> None:
        self.mean = np.asarray(mean)
        self.weights = np.asarray(weights)
        self.modes = np.asarray(modes)
        self.SD = np.asarray(sd)
        self.sdnorm = self.SD is not None and len(self.SD.shape) != 0

    def getMean(self) -> np.ndarray:
        return self.mean.copy()

    def getMode(self, n: int = -1) -> np.ndarray:
        if n == -1:
            return self.modes.copy()
        return self.modes[:, n].copy()

    def getSD(self) -> np.ndarray:
        return self.SD.copy()

    def getWeightsBySD(self, modes: Sequence[int], sd: Sequence[float]) -> np.ndarray:
        return np.asarray(sd) * np.sqrt(self.weights[np.asarray(modes)])

    def reconstruct(self, weights: Sequence[float], modes: Sequence[int]) -> np.ndarray:
        if len(weights) != len(modes):
            raise ValueError(f"length mismatch between weights and modes: {len(weights)} vs {len(modes)}")

        f = np.array([self.getMode(p) for p in modes]).T
        if self.SD is None or len(self.SD.shape) == 0:
            raise ValueError("SD values are missing; cannot reconstruct with SD normalization.")

        return np.dot(f, np.asarray(weights)).squeeze() * self.getSD() + self.getMean()


def _load_principal_components(filename: str | Path) -> PrincipalComponentsLite:
    """
    Minimal loader for legacy .pc/.pc.npz npz payloads.
    """
    try:
        s = np.load(str(filename), allow_pickle=True)
    except OSError as exc:
        raise IOError(f"unable to np.load {filename}") from exc

    use_bytes_keys = b"mean" in s
    if use_bytes_keys:
        mean = s[b"mean"]
        weights = s[b"weights"]
        modes = s[b"modes"]
        sd = s[b"SD"]
    else:
        mean = s["mean"]
        weights = s["weights"]
        modes = s["modes"]
        sd = s["SD"]

    if isinstance(sd, np.ndarray) and sd.dtype == object and sd.size == 1 and sd.item() is None:
        raise ValueError("SSDM file has no SD field values; SD-normalized reconstruction is required.")

    return PrincipalComponentsLite(mean=mean, weights=weights, modes=modes, sd=sd)


def r2c14(x_recon: np.ndarray) -> np.ndarray:
    return x_recon.reshape((-1, 4))


def _sample_data(data: np.ndarray, n_points: int) -> np.ndarray:
    if n_points < 1:
        raise ValueError("N must be > 1")
    if n_points > len(data):
        return data
    i = np.linspace(0, len(data) - 1, n_points).astype(int)
    return data[i, :]


def fit_data_rigid_dpep(
    data: np.ndarray,
    target: np.ndarray,
    *,
    xtol: float = 1e-3,
    maxfev: int = 0,
    t0: Optional[np.ndarray] = None,
    sample: Optional[int] = None,
    output_errors: int = 0,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]] | tuple[np.ndarray, np.ndarray]:
    if sample is not None:
        d = _sample_data(data, sample)
        t = _sample_data(target, sample)
    else:
        d = data
        t = target

    if t0 is None:
        t0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    d = np.asarray(d)

    def obj(params: np.ndarray) -> np.ndarray:
        dt = transform3D.transformRigid3DAboutCoM(d, params)
        dt_tree = cKDTree(dt)
        dist = dt_tree.query(t)[0]
        return dist * dist

    initial_rmse = np.sqrt(obj(t0).mean())
    t_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]
    data_fitted = transform3D.transformRigid3DAboutCoM(data, t_opt)
    final_rmse = np.sqrt(obj(t_opt).mean())

    if output_errors:
        return t_opt, data_fitted, (initial_rmse, final_rmse)
    return t_opt, data_fitted


def fit_data_rigid_scale_dpep(
    data: np.ndarray,
    target: np.ndarray,
    *,
    xtol: float = 1e-3,
    maxfev: int = 0,
    t0: Optional[np.ndarray] = None,
    sample: Optional[int] = None,
    output_errors: int = 0,
    scale_threshold: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]] | tuple[np.ndarray, np.ndarray]:
    if sample is not None:
        d = _sample_data(data, sample)
        t = _sample_data(target, sample)
    else:
        d = data
        t = target

    if t0 is None:
        t0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    d = np.asarray(d)

    if scale_threshold is not None:

        def obj(params: np.ndarray) -> np.ndarray:
            dt = transform3D.transformRigidScale3DAboutCoM(d, params)
            dt_tree = cKDTree(dt)
            dist = dt_tree.query(t)[0]
            s = params[6]
            sw = 1000.0 * s if s > scale_threshold else 1.0
            return dist * dist + sw

    else:

        def obj(params: np.ndarray) -> np.ndarray:
            dt = transform3D.transformRigidScale3DAboutCoM(d, params)
            dt_tree = cKDTree(dt)
            dist = dt_tree.query(t)[0]
            return dist * dist

    initial_rmse = np.sqrt(obj(t0).mean())
    t_opt = leastsq(obj, t0, xtol=xtol, maxfev=maxfev)[0]
    data_fitted = transform3D.transformRigidScale3DAboutCoM(data, t_opt)
    final_rmse = np.sqrt(obj(t_opt).mean())

    if output_errors:
        return t_opt, data_fitted, (initial_rmse, final_rmse)
    return t_opt, data_fitted


def estimate_initial_transform(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    fit_scale: bool,
    auto_align: bool,
    sample: Optional[int],
) -> np.ndarray:
    if not auto_align:
        if fit_scale:
            src_span = np.linalg.norm(source_xyz.max(axis=0) - source_xyz.min(axis=0))
            trg_span = np.linalg.norm(target_xyz.max(axis=0) - target_xyz.min(axis=0))
            scale_factor = float(trg_span / src_span) if src_span > 0 else 1.0
            return np.hstack([np.zeros(6), scale_factor])
        return np.zeros(6)

    init_rot = np.deg2rad((0.0, 0.0, 0.0))
    init_trans = target_xyz.mean(axis=0) - source_xyz.mean(axis=0)
    t0 = np.hstack([init_trans, init_rot])

    reg1_t, _, _ = fit_data_rigid_dpep(
        source_xyz,
        target_xyz,
        xtol=1e-5,
        sample=sample,
        t0=t0,
        output_errors=1,
    )
    if not fit_scale:
        return np.asarray(reg1_t, dtype=float)

    src_span = np.linalg.norm(source_xyz.max(axis=0) - source_xyz.min(axis=0))
    trg_span = np.linalg.norm(target_xyz.max(axis=0) - target_xyz.min(axis=0))
    scale_factor = float(trg_span / src_span) if src_span > 0 else 1.0

    reg2_t, _, _ = fit_data_rigid_scale_dpep(
        source_xyz,
        target_xyz,
        xtol=1e-5,
        sample=sample,
        t0=np.hstack([reg1_t[:6], scale_factor]),
        output_errors=1,
    )
    return np.asarray(reg2_t, dtype=float)


def _initializer(
    targ_data_xyz: np.ndarray,
    targ_data_density: np.ndarray,
    ssdm_obj: PrincipalComponentsLite,
) -> None:
    global _shared_targ_tree
    global _shared_ssdm
    global _shared_data_xyz
    global _shared_data_density

    _shared_data_xyz = targ_data_xyz
    _shared_data_density = targ_data_density
    _shared_targ_tree = cKDTree(_shared_data_xyz, leafsize=40)
    _shared_ssdm = ssdm_obj


def _evaluate_individual(
    individual: Sequence[float],
    *,
    fit_comps: Sequence[int],
    fit_mode: str,
    fit_scale: bool,
    workers_kdtree_query: int,
) -> tuple[float, float]:
    if _shared_ssdm is None or _shared_targ_tree is None or _shared_data_xyz is None or _shared_data_density is None:
        raise RuntimeError("Shared evaluator state is not initialized.")

    ssdm = _shared_ssdm
    targ_tree = _shared_targ_tree
    data_xyz = _shared_data_xyz
    data_density = _shared_data_density
    x = np.asarray(individual, dtype=float)

    if fit_scale:
        recon = ssdm.reconstruct(ssdm.getWeightsBySD(fit_comps, x[7:]), fit_comps)
        recon_shape_xyz = r2c14(recon)[:, :3]
        recon_density = r2c14(recon)[:, 3]
        recon_pts = transform3D.transformRigidScale3DAboutCoM(recon_shape_xyz, x[:7])
    else:
        recon = ssdm.reconstruct(ssdm.getWeightsBySD(fit_comps, x[6:]), fit_comps)
        recon_shape_xyz = r2c14(recon)[:, :3]
        recon_density = r2c14(recon)[:, 3]
        recon_pts = transform3D.transformRigid3DAboutCoM(recon_shape_xyz, x[:6])

    if fit_mode == "st":
        dists_shape, indices = targ_tree.query(recon_pts, eps=1e-9, workers=workers_kdtree_query, p=2)
        target_density_corr = data_density[indices]
        shape_error = np.mean(dists_shape**2)
        density_error = np.mean((recon_density - target_density_corr) ** 2)
    elif fit_mode == "ts":
        recon_tree = cKDTree(recon_pts, leafsize=40)
        dists_shape, indices = recon_tree.query(data_xyz, eps=1e-9, workers=workers_kdtree_query, p=2)
        recon_density_corr = recon_density[indices]
        shape_error = np.mean(dists_shape**2)
        density_error = np.mean((data_density - recon_density_corr) ** 2)
    elif fit_mode == "2way":
        dists_st, indices_st = targ_tree.query(recon_pts, eps=1e-9, workers=workers_kdtree_query, p=2)
        target_density_corr_st = data_density[indices_st]
        shape_error_st = np.mean(dists_st**2)
        density_error_st = np.mean((recon_density - target_density_corr_st) ** 2)

        recon_tree = cKDTree(recon_pts, leafsize=40)
        dists_ts, indices_ts = recon_tree.query(data_xyz, eps=1e-9, workers=workers_kdtree_query, p=2)
        recon_density_corr_ts = recon_density[indices_ts]
        shape_error_ts = np.mean(dists_ts**2)
        density_error_ts = np.mean((data_density - recon_density_corr_ts) ** 2)

        shape_error = (shape_error_st + shape_error_ts) / 2.0
        density_error = (density_error_st + density_error_ts) / 2.0
    else:
        shape_error = np.mean(((data_xyz - recon_pts) ** 2.0).sum(1))
        density_error = np.mean((data_density - recon_density) ** 2)

    return float(shape_error), float(density_error)


def _create_deap_types() -> tuple[type, type]:
    fitness_name = "FitnessMultiSD"
    individual_name = "IndividualSD"
    if not hasattr(creator, fitness_name):
        creator.create(fitness_name, base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, individual_name):
        creator.create(individual_name, list, fitness=getattr(creator, fitness_name))
    return getattr(creator, fitness_name), getattr(creator, individual_name)


def fit_ssdm_pareto(
    data: np.ndarray,
    ssdm: PrincipalComponentsLite,
    fit_comps: Sequence[int],
    fit_mode: str,
    *,
    init_t: np.ndarray,
    fit_scale: bool,
    sample: Optional[int],
    verbose: bool,
    workers: int,
    pop_size: int,
    n_gen: int,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, float], list[ParetoSolution]]:
    random.seed(random_seed)
    np.random.seed(random_seed)

    if sample is not None:
        data = _sample_data(data, sample)
    if data.shape[1] != 4:
        raise ValueError("Input data must be an Nx4 array (x,y,z,density).")

    data_xyz = data[:, :3]
    data_density = data[:, 3]
    if workers == -1:
        workers = multiprocessing.cpu_count()
    if workers < 1:
        workers = 1
    workers_kdtree_query = 1

    _, individual_type = _create_deap_types()
    toolbox = base.Toolbox()

    n_params = len(init_t) + len(fit_comps)

    def init_individual() -> Any:
        vals = [random.uniform(-1.0, 1.0) for _ in range(n_params)]
        return individual_type(vals)

    evaluate_func = partial(
        _evaluate_individual,
        fit_comps=fit_comps,
        fit_mode=fit_mode,
        fit_scale=fit_scale,
        workers_kdtree_query=workers_kdtree_query,
    )

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    pop = []
    init_candidate = np.zeros(n_params)
    init_candidate[: len(init_t)] = init_t
    pop.append(individual_type(init_candidate.tolist()))
    for _ in range(max(0, pop_size - 1)):
        pop.append(toolbox.individual())

    for ind in pop:
        ind.fitness.values = float("nan"), float("nan")

    pool = multiprocessing.Pool(
        processes=workers,
        initializer=_initializer,
        initargs=(data_xyz, data_density, ssdm),
    )
    toolbox.register("map", pool.map)

    try:
        fitnesses = list(toolbox.map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        stagnation_count = 0
        max_stagnation = 20
        last_best_error = float("inf")
        generations_run = 0

        for gen in range(n_gen):
            generations_run = gen + 1
            if verbose:
                print(f"\rGeneration: {generations_run}/{n_gen}", end="", flush=True)

            offspring = algorithms.varOr(pop, toolbox, lambda_=pop_size, cxpb=0.8, mutpb=0.2)
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.select(pop + offspring, pop_size)

            front = tools.sortNondominated(pop, pop_size)[0]
            if not front:
                continue

            best_total = min((sum(ind.fitness.values) for ind in front if ind.fitness.valid), default=float("inf"))
            if best_total < last_best_error:
                last_best_error = best_total
                stagnation_count = 0
            else:
                stagnation_count += 1
                if stagnation_count >= max_stagnation:
                    break
    finally:
        pool.close()
        pool.join()

    if verbose:
        print(f"\nFinal generations run: {generations_run}")

    final_front = tools.sortNondominated(pop, len(pop))[0]
    best_solution = None
    min_total_error = float("inf")
    for ind in final_front:
        if ind.fitness.valid:
            total_error = sum(ind.fitness.values)
            if total_error < min_total_error:
                min_total_error = total_error
                best_solution = ind

    x_opt = np.zeros(n_params)
    recon_data_opt = np.zeros((data.shape[0], 4))
    err_opt = np.zeros(2)
    dist_opt_rms = 0.0

    if best_solution is not None:
        x_opt = np.asarray(best_solution, dtype=float)
        if fit_scale:
            recon = ssdm.reconstruct(ssdm.getWeightsBySD(fit_comps, x_opt[7:]), fit_comps)
            shape_xyz = r2c14(recon)[:, :3]
            recon_density = r2c14(recon)[:, 3]
            recon_pts = transform3D.transformRigidScale3DAboutCoM(shape_xyz, x_opt[:7])
        else:
            recon = ssdm.reconstruct(ssdm.getWeightsBySD(fit_comps, x_opt[6:]), fit_comps)
            shape_xyz = r2c14(recon)[:, :3]
            recon_density = r2c14(recon)[:, 3]
            recon_pts = transform3D.transformRigid3DAboutCoM(shape_xyz, x_opt[:6])

        recon_data_opt = np.column_stack([recon_pts, recon_density])
        err_opt = np.asarray(best_solution.fitness.values, dtype=float)

        temp_targ_tree = cKDTree(data_xyz, leafsize=40)
        if fit_mode == "ts":
            temp_recon_tree = cKDTree(recon_pts, leafsize=40)
            dists, _ = temp_recon_tree.query(data_xyz, eps=1e-9, workers=workers_kdtree_query)
            dist_opt_rms = float(np.sqrt(np.mean(dists**2)))
        elif fit_mode == "st":
            dists, _ = temp_targ_tree.query(recon_pts, eps=1e-9, workers=workers_kdtree_query)
            dist_opt_rms = float(np.sqrt(np.mean(dists**2)))
        elif fit_mode == "2way":
            d_st, _ = temp_targ_tree.query(recon_pts, eps=1e-9, workers=workers_kdtree_query)
            temp_recon_tree = cKDTree(recon_pts, leafsize=40)
            d_ts, _ = temp_recon_tree.query(data_xyz, eps=1e-9, workers=workers_kdtree_query)
            dist_opt_rms = float(np.sqrt((np.mean(d_st**2) + np.mean(d_ts**2)) / 2.0))
        else:
            dist_opt_rms = float(np.sqrt(np.mean(((data_xyz - recon_pts) ** 2.0).sum(1))))

    pareto_solutions: list[ParetoSolution] = []
    for ind in final_front:
        if not ind.fitness.valid:
            continue
        x = np.asarray(ind, dtype=float)
        if fit_scale:
            recon = ssdm.reconstruct(ssdm.getWeightsBySD(fit_comps, x[7:]), fit_comps)
            shape_xyz = r2c14(recon)[:, :3]
            recon_density = r2c14(recon)[:, 3]
            recon_pts = transform3D.transformRigidScale3DAboutCoM(shape_xyz, x[:7])
        else:
            recon = ssdm.reconstruct(ssdm.getWeightsBySD(fit_comps, x[6:]), fit_comps)
            shape_xyz = r2c14(recon)[:, :3]
            recon_density = r2c14(recon)[:, 3]
            recon_pts = transform3D.transformRigid3DAboutCoM(shape_xyz, x[:6])

        pareto_solutions.append(
            ParetoSolution(
                shape_error=float(ind.fitness.values[0]),
                density_error=float(ind.fitness.values[1]),
                parameters=x,
                recon_points=recon_pts,
                recon_density=recon_density,
            )
        )

    return x_opt, recon_data_opt, (err_opt, dist_opt_rms), pareto_solutions


def fit_ssdm_to_target(
    target_xyzd: np.ndarray,
    ssdm_path: str | Path,
    *,
    fit_comps: Sequence[int],
    fit_mode: str,
    fit_scale: bool,
    auto_align: bool,
    sample: Optional[int],
    verbose: bool,
    workers: int,
    pop_size: int,
    n_gen: int,
    random_seed: int,
) -> LegacySDFitResult:
    ssdm = _load_principal_components(ssdm_path)
    source_xyz = r2c14(ssdm.getMean())[:, :3]
    target_xyz = target_xyzd[:, :3]

    init_sample = sample if sample is not None else 5000
    init_t = estimate_initial_transform(
        source_xyz=source_xyz,
        target_xyz=target_xyz,
        fit_scale=fit_scale,
        auto_align=auto_align,
        sample=init_sample,
    )

    final_sample = sample if sample is not None else 21900
    x_opt, recon_xyzd, (err_opt, reg_rms), pareto_solutions = fit_ssdm_pareto(
        target_xyzd,
        ssdm,
        list(fit_comps),
        fit_mode,
        init_t=init_t,
        fit_scale=fit_scale,
        sample=final_sample,
        verbose=verbose,
        workers=workers,
        pop_size=pop_size,
        n_gen=n_gen,
        random_seed=random_seed,
    )

    return LegacySDFitResult(
        recon_xyzd=np.asarray(recon_xyzd),
        reg_rms=float(reg_rms),
        opt_params=np.asarray(x_opt),
        opt_error=np.asarray(err_opt),
        pareto_solutions=list(pareto_solutions),
        init_transform=np.asarray(init_t),
    )

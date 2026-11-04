#!/usr/bin/env python3
"""Shared invariants and I/O for the modular ThermoRDF campaign.

This module contains no campaign loop.  Each scientific method has its own
entry point; the common code only guarantees that all of them read the same
states, coordinates, job manifest, and artifact schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


IMPLEMENTATION_ID = "thermordf-low-data-modular-campaign-2026-10-07"
FORMAT_VERSION = 1
BUDGETS = np.asarray([4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52], dtype=int)
PREDICTIVE_REPETITIONS = 10
GEOMETRIC_REPETITIONS = 250
MASTER_SEED = 20261003
BOOTSTRAP_RESAMPLES = 2000
EXPECTED_STATES = 372
CHANNELS = ("OO", "OH_inter", "HH_inter", "COM_COM")
RADIAL_BINS = 725
RDF_DIMENSION = len(CHANNELS) * RADIAL_BINS
IDW_POWER = 2.0
RIDGE_RELATIVE_FACTOR = 1.0e-6
EOS_TEMPERATURE_DEGREE = 3
EOS_PRESSURE_BASIS_SIZE = 4
MLP_HIDDEN_WIDTHS = (8, 16)
MLP_LEARNING_RATE = 1.0e-3
MLP_L2_ALPHA = 1.0e-6
MLP_MAX_EPOCHS = 30_000
MLP_PATIENCE = 1_000
MLP_RELATIVE_MINIMUM_IMPROVEMENT = 1.0e-4
CORNER_ORDER = ((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0))
METRIC_COLUMNS = ("fill_distance", "separation_distance", "mean_nearest_distance")

ALIASES: Dict[str, Sequence[str]] = {
    "temperature": ("T_K", "temperature", "temp", "T"),
    "pressure": ("P_bar", "pressure", "press", "P"),
    "density": ("rho_mean_kg_m3", "density", "rho", "mass_density"),
    "radial": ("r_nm", "r", "radius", "distance", "radial_distance"),
    "rdf": ("g_mean", "rdf_mean", "g", "rdf"),
    "channel": ("channel", "rdf_channel"),
    "state_id": ("state_id", "state_label", "id"),
}


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class DataBundle:
    temperature: np.ndarray
    pressure: np.ndarray
    density: np.ndarray
    state_id: np.ndarray
    radial: np.ndarray
    rdf: np.ndarray
    rdf_target: np.ndarray
    coords_range: np.ndarray
    coords_grid: np.ndarray
    eos_coords: np.ndarray
    lambda_pressure_grid: float
    delta_temperature_normalised: float
    delta_pressure_normalised: float
    units: Mapping[str, str]
    names: Mapping[str, str]


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def resolve_name(ds: xr.Dataset, role: str, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in ds.variables:
            raise CampaignError(f"Variable {explicit!r} supplied for {role} is absent.")
        return explicit
    exact = [name for name in ALIASES[role] if name in ds.variables]
    if len(exact) == 1:
        return exact[0]
    lower: Dict[str, list[str]] = {}
    for name in ds.variables:
        lower.setdefault(name.lower(), []).append(name)
    matches: list[str] = []
    for alias in ALIASES[role]:
        matches.extend(lower.get(alias.lower(), []))
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise CampaignError(f"Could not resolve exactly one variable for {role}: {matches}.")
    return matches[0]


def _vector(da: xr.DataArray, state_dim: str, role: str) -> np.ndarray:
    extra = [dim for dim in da.dims if dim != state_dim]
    bad = [dim for dim in extra if da.sizes[dim] != 1]
    if bad:
        raise CampaignError(f"Variable {da.name!r} for {role} has unsupported dimensions {bad}.")
    if extra:
        da = da.squeeze(extra, drop=True)
    if da.dims != (state_dim,):
        raise CampaignError(f"Variable {da.name!r} for {role} must use {state_dim!r}.")
    return np.asarray(da.values)


def _normalise(values: np.ndarray, label: str) -> np.ndarray:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if not maximum > minimum:
        raise CampaignError(f"{label} must have positive span.")
    return (values - minimum) / (maximum - minimum)


def _typical_increment(values: np.ndarray, label: str) -> float:
    unique = np.unique(values)
    if unique.size < 2:
        raise CampaignError(f"{label} needs at least two levels.")
    increments = np.diff(unique) / float(unique[-1] - unique[0])
    value = float(np.median(increments))
    if not np.isfinite(value) or value <= 0.0:
        raise CampaignError(f"Invalid typical increment for {label}.")
    return value


def load_data(netcdf: Path) -> DataBundle:
    if not netcdf.is_file():
        raise FileNotFoundError(netcdf)
    with xr.open_dataset(netcdf) as ds:
        names = {role: resolve_name(ds, role) for role in ALIASES}
        temperature_da = ds[names["temperature"]]
        if temperature_da.ndim != 1:
            raise CampaignError("Temperature must be one-dimensional.")
        state_dim = temperature_da.dims[0]
        temperature = _vector(temperature_da, state_dim, "temperature").astype(float)
        pressure = _vector(ds[names["pressure"]], state_dim, "pressure").astype(float)
        density = _vector(ds[names["density"]], state_dim, "density").astype(float)
        state_id = _vector(ds[names["state_id"]], state_dim, "state_id").astype(str)
        radial_da = ds[names["radial"]]
        if radial_da.ndim != 1:
            raise CampaignError("Radial coordinate must be one-dimensional.")
        radial_dim = radial_da.dims[0]
        radial = np.asarray(radial_da.values, dtype=float)
        channel_da = ds[names["channel"]]
        if channel_da.ndim != 1:
            raise CampaignError("Channel coordinate must be one-dimensional.")
        channel_dim = channel_da.dims[0]
        channels = tuple(str(value) for value in channel_da.values)
        if channels != CHANNELS:
            raise CampaignError(f"Expected channels {CHANNELS}, found {channels}.")
        rdf_da = ds[names["rdf"]]
        required = {state_dim, channel_dim, radial_dim}
        if not required.issubset(rdf_da.dims):
            raise CampaignError(f"RDF variable must contain dimensions {sorted(required)}.")
        extra = [dim for dim in rdf_da.dims if dim not in required]
        bad = [dim for dim in extra if rdf_da.sizes[dim] != 1]
        if bad:
            raise CampaignError(f"RDF mean has unsupported dimensions {bad}.")
        if extra:
            rdf_da = rdf_da.squeeze(extra, drop=True)
        rdf = np.asarray(rdf_da.transpose(state_dim, channel_dim, radial_dim).values, dtype=float)
        units = {
            "temperature": str(temperature_da.attrs.get("units", "unspecified")),
            "pressure": str(ds[names["pressure"]].attrs.get("units", "unspecified")),
            "density": str(ds[names["density"]].attrs.get("units", "unspecified")),
            "radial": str(radial_da.attrs.get("units", "unspecified")),
        }
    if temperature.size != EXPECTED_STATES:
        raise CampaignError(f"Expected {EXPECTED_STATES} states, found {temperature.size}.")
    if radial.shape != (RADIAL_BINS,):
        raise CampaignError(f"Expected {RADIAL_BINS} radial bins, found {radial.shape}.")
    if rdf.shape != (EXPECTED_STATES, len(CHANNELS), RADIAL_BINS):
        raise CampaignError(f"Unexpected RDF shape: {rdf.shape}.")
    complete = (
        np.isfinite(temperature) & np.isfinite(pressure) & np.isfinite(density)
        & np.all(np.isfinite(rdf.reshape(EXPECTED_STATES, -1)), axis=1)
    )
    if not np.all(complete):
        raise CampaignError("The campaign requires the complete verified state domain.")
    if np.unique(state_id).size != state_id.size:
        raise CampaignError("State identifiers must be unique.")
    coordinates = pd.DataFrame({"T": temperature, "P": pressure})
    if coordinates.duplicated().any():
        raise CampaignError("Thermodynamic coordinates must be unique.")
    t_unit = _normalise(temperature, "temperature")
    p_unit = _normalise(pressure, "pressure")
    delta_t = _typical_increment(temperature, "temperature")
    delta_p = _typical_increment(pressure, "pressure")
    lambda_pressure = delta_t / delta_p
    if not 0.0 < lambda_pressure <= 1.0 + 1.0e-12:
        raise CampaignError(f"Invalid grid-derived pressure weight: {lambda_pressure}.")
    return DataBundle(
        temperature=temperature,
        pressure=pressure,
        density=density,
        state_id=state_id,
        radial=radial,
        rdf=rdf,
        rdf_target=rdf.reshape(EXPECTED_STATES, RDF_DIMENSION),
        coords_range=np.column_stack((t_unit, p_unit)),
        coords_grid=np.column_stack((t_unit, lambda_pressure * p_unit)),
        eos_coords=np.column_stack((2.0 * t_unit - 1.0, 2.0 * p_unit - 1.0)),
        lambda_pressure_grid=float(lambda_pressure),
        delta_temperature_normalised=delta_t,
        delta_pressure_normalised=delta_p,
        units=units,
        names=names,
    )


def campaign_configuration(netcdf: Path, data: DataBundle) -> Dict[str, Any]:
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "format_version": FORMAT_VERSION,
        "source_netcdf": file_identity(netcdf),
        "resolved_variables": dict(data.names),
        "units": dict(data.units),
        "budgets": BUDGETS.tolist(),
        "geometries": ["range", "grid_resolution"],
        "geometric_repetitions": GEOMETRIC_REPETITIONS,
        "predictive_repetitions": PREDICTIVE_REPETITIONS,
        "master_seed": MASTER_SEED,
        "grid_pressure_weight": data.lambda_pressure_grid,
        "models": {
            "EOS": {
                "temperature_degree": EOS_TEMPERATURE_DEGREE,
                "pressure_basis_size": EOS_PRESSURE_BASIS_SIZE,
                "target_scaling": "training_mean_and_population_standard_deviation",
                "relative_ridge_factor": RIDGE_RELATIVE_FACTOR,
            },
            "IDW": {"power": IDW_POWER},
            "Ridge": {"relative_factor": RIDGE_RELATIVE_FACTOR},
            "MLP": {
                "architecture": [2, *MLP_HIDDEN_WIDTHS, RDF_DIMENSION],
                "activation": "tanh",
                "learning_rate": MLP_LEARNING_RATE,
                "l2_alpha": MLP_L2_ALPHA,
                "maximum_epochs": MLP_MAX_EPOCHS,
                "patience": MLP_PATIENCE,
                "relative_minimum_improvement": MLP_RELATIVE_MINIMUM_IMPROVEMENT,
                "dtype": "float32",
                "seed": MASTER_SEED,
            },
        },
        "previous_results_reused": False,
    }


def initialise_root(root: Path, netcdf: Path, data: DataBundle, resume: bool) -> Dict[str, Any]:
    config = campaign_configuration(netcdf, data)
    config_path = root / "campaign_configuration.json"
    if not root.exists():
        root.mkdir(parents=True)
        atomic_json(config_path, config)
    elif not resume:
        raise CampaignError(f"Output directory already exists: {root}. Use --resume or a new path.")
    elif not config_path.is_file():
        raise CampaignError("Resume requires campaign_configuration.json.")
    else:
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        if stored != json_ready(config):
            raise CampaignError("Resume refused: NetCDF or frozen configuration changed.")
    for name in ("sampling", "models", "status", "evaluation", "report", "logs"):
        (root / name).mkdir(exist_ok=True)
    return config


def subset_id(indices: np.ndarray, state_ids: np.ndarray) -> str:
    selected = sorted(str(state_ids[int(index)]) for index in np.asarray(indices, dtype=int))
    return hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()[:20]


def read_jobs(root: Path) -> pd.DataFrame:
    path = root / "sampling" / "model_jobs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run thermordf_sampling.py first: {path}")
    frame = pd.read_csv(path, dtype={"subset_id": str})
    required = {"geometry", "strategy", "repetition", "budget", "subset_id", "state_indices"}
    missing = required.difference(frame.columns)
    if missing:
        raise CampaignError(f"model_jobs.csv is missing {sorted(missing)}.")
    return frame


def unique_jobs(root: Path) -> pd.DataFrame:
    logical = read_jobs(root)
    columns = ["subset_id", "budget", "state_indices"]
    unique = logical[columns].drop_duplicates().sort_values(["budget", "subset_id"])
    counts = logical.groupby("subset_id").size().rename("logical_alias_count")
    unique = unique.merge(counts, on="subset_id", how="left")
    return unique.reset_index(drop=True)


def parse_indices(text: str, budget: int) -> np.ndarray:
    values = np.asarray([int(value) for value in str(text).split(";") if value != ""], dtype=int)
    if values.shape != (int(budget),) or np.unique(values).size != values.size:
        raise CampaignError(f"Invalid selected-state list for N={budget}.")
    if np.min(values) < 0 or np.max(values) >= EXPECTED_STATES:
        raise CampaignError(f"Selected-state list is outside the valid domain for N={budget}.")
    return values


def aliases_for_subset(root: Path, identifier: str) -> list[Dict[str, Any]]:
    rows = read_jobs(root)
    rows = rows[rows["subset_id"].astype(str) == str(identifier)]
    return rows[["geometry", "strategy", "repetition", "budget"]].to_dict("records")


def artifact_paths(root: Path, method: str, identifier: str) -> Tuple[Path, Path]:
    directory = root / "models" / method.lower()
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / f"{method.lower()}_{identifier}"
    suffix = ".pt" if method.upper() == "MLP" else ".npz"
    return base.with_suffix(suffix), base.with_suffix(".json")


def is_complete(root: Path, method: str, identifier: str) -> bool:
    artifact, metadata_path = artifact_paths(root, method, identifier)
    if not artifact.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        identity = file_identity(artifact)
    except Exception:
        return False
    expected = metadata.get("artifact", {})
    return (
        metadata.get("status") == "COMPLETE"
        and metadata.get("implementation_id") == IMPLEMENTATION_ID
        and str(metadata.get("subset_id")) == str(identifier)
        and expected.get("sha256") == identity["sha256"]
        and int(expected.get("size_bytes", -1)) == identity["size_bytes"]
    )


def failure_record(root: Path, method: str, identifier: str, budget: int, exc: BaseException) -> None:
    path = root / "status" / method.lower() / f"{method.lower()}_{identifier}.json"
    atomic_json(path, {
        "status": "FAILED",
        "implementation_id": IMPLEMENTATION_ID,
        "method": method,
        "subset_id": identifier,
        "budget": int(budget),
        "error_type": type(exc).__name__,
        "error": str(exc),
    })


def completion_record(root: Path, method: str, identifier: str, budget: int, duration: float) -> None:
    path = root / "status" / method.lower() / f"{method.lower()}_{identifier}.json"
    atomic_json(path, {
        "status": "COMPLETE",
        "implementation_id": IMPLEMENTATION_ID,
        "method": method,
        "subset_id": identifier,
        "budget": int(budget),
        "duration_seconds": float(duration),
    })


def execution_name(method: str, root: Path, identifier: str, budget: int) -> str:
    aliases = aliases_for_subset(root, identifier)
    if len(aliases) == 1:
        item = aliases[0]
        rep = int(item["repetition"])
        rep_text = "designed" if rep == 0 else f"rep{rep:02d}"
        return (
            f"ThermoRDF_{method.upper()}__{item['geometry']}__{item['strategy']}__"
            f"{rep_text}__N{int(budget):03d}"
        )
    return f"ThermoRDF_{method.upper()}__shared_{len(aliases)}aliases__N{int(budget):03d}__{identifier[:8]}"


def notify(package_dir: Path, name: str, status: str, duration_seconds: float) -> None:
    helper = package_dir / "notify_job.sh"
    if not helper.is_file():
        print(f"NOTIFICATION SKIPPED: missing {helper}", flush=True)
        return
    result = subprocess.run(
        ["bash", str(helper), name, status, str(int(round(duration_seconds)))],
        check=False,
    )
    if result.returncode != 0:
        print(f"NOTIFICATION WARNING: {name} helper exit={result.returncode}", flush=True)


def pairwise_distance(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def distances(query: np.ndarray, train: np.ndarray) -> np.ndarray:
    delta = query[:, None, :] - train[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def idw_predict(train_x: np.ndarray, query_x: np.ndarray, train_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    distance = distances(query_x, train_x)
    exact = np.isclose(distance, 0.0, rtol=0.0, atol=1.0e-15)
    prediction = np.empty((query_x.shape[0], train_y.shape[1]), dtype=float)
    exact_rows = np.any(exact, axis=1)
    for row in np.flatnonzero(exact_rows):
        prediction[row] = np.mean(train_y[exact[row]], axis=0)
    regular = np.flatnonzero(~exact_rows)
    if regular.size:
        weights = distance[regular] ** (-IDW_POWER)
        weights /= np.sum(weights, axis=1, keepdims=True)
        prediction[regular] = weights @ train_y
    if prediction.shape != (query_x.shape[0], RDF_DIMENSION):
        raise CampaignError("Unexpected IDW prediction shape.")
    return prediction, np.min(distance, axis=1)


def relative_ridge_fit(inputs: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
    inputs = np.asarray(inputs, dtype=float)
    targets = np.asarray(targets, dtype=float)
    one_target = targets.ndim == 1
    if one_target:
        targets = targets[:, None]
    x_mean = np.mean(inputs, axis=0)
    y_mean = np.mean(targets, axis=0)
    xc = inputs - x_mean
    yc = targets - y_mean
    gram = (xc.T @ xc) / float(inputs.shape[0])
    scale = float(np.trace(gram) / max(1, inputs.shape[1]))
    if not np.isfinite(scale) or scale <= 0.0:
        raise CampaignError("Cannot define the approved relative Ridge penalty.")
    ridge_lambda = RIDGE_RELATIVE_FACTOR * scale
    coefficients = np.linalg.solve(
        gram + ridge_lambda * np.eye(inputs.shape[1]),
        (xc.T @ yc) / float(inputs.shape[0]),
    )
    intercept = y_mean - x_mean @ coefficients
    if one_target:
        coefficients = coefficients[:, 0]
        intercept = np.asarray(intercept[0])
    return {"coefficients": coefficients, "intercept": intercept, "lambda": ridge_lambda}


def linear_predict(model: Mapping[str, Any], inputs: np.ndarray) -> np.ndarray:
    return np.asarray(inputs, dtype=float) @ np.asarray(model["coefficients"], dtype=float) + np.asarray(model["intercept"], dtype=float)


def natural_cubic_pressure_basis(values: np.ndarray, knots: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    knots = np.asarray(knots, dtype=float).reshape(-1)
    if knots.shape != (4,) or not np.all(np.diff(knots) > 0.0):
        raise CampaignError("EOS pressure knots must be four strictly increasing values.")
    last_gap = knots[-1] - knots[-2]
    positive_cube = lambda x, knot: np.maximum(x - knot, 0.0) ** 3
    columns = [np.ones_like(values), values]
    for knot in knots[:-2]:
        columns.append(
            positive_cube(values, knot)
            - ((knots[-1] - knot) / last_gap) * positive_cube(values, knots[-2])
            + ((knots[-2] - knot) / last_gap) * positive_cube(values, knots[-1])
        )
    return np.column_stack(columns)


def eos_design_matrix(coords: np.ndarray, knots: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=float)
    pressure_basis = natural_cubic_pressure_basis(coords[:, 1], knots)
    temperature_basis = np.column_stack(
        [coords[:, 0] ** degree for degree in range(EOS_TEMPERATURE_DEGREE + 1)]
    )
    design = np.einsum("ni,nj->nij", temperature_basis, pressure_basis).reshape(coords.shape[0], -1)
    if design.shape[1] != 16 or not np.all(np.isfinite(design)):
        raise CampaignError("Invalid EOS design matrix.")
    return design


def eos_knots(data: DataBundle) -> np.ndarray:
    unique = np.unique(data.eos_coords[:, 1])
    knots = np.quantile(unique, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    if not np.all(np.diff(knots) > 0.0):
        raise CampaignError("Pressure levels cannot define four EOS knots.")
    return knots


def predict_eos(model: Mapping[str, Any], coords: np.ndarray) -> np.ndarray:
    design = eos_design_matrix(coords, np.asarray(model["pressure_knots"], dtype=float))
    standardized = np.asarray(linear_predict(model, design), dtype=float).reshape(-1)
    return float(model["density_mean"]) + float(model["density_scale"]) * standardized


def selected_and_evaluation(data: DataBundle, text: str, budget: int) -> Tuple[np.ndarray, np.ndarray]:
    selected = parse_indices(text, budget)
    evaluation = np.setdiff1d(np.arange(EXPECTED_STATES, dtype=int), selected, assume_unique=True)
    return selected, evaluation


def method_summary(root: Path, method: str, failures: Sequence[str], completed: int, skipped: int) -> None:
    atomic_json(root / "models" / method.lower() / "run_summary.json", {
        "status": "COMPLETE" if not failures else "INCOMPLETE",
        "implementation_id": IMPLEMENTATION_ID,
        "method": method,
        "completed_in_this_invocation": int(completed),
        "skipped_complete": int(skipped),
        "failed_subset_ids": list(failures),
    })


def run_independent_jobs(
    root: Path,
    method: str,
    package_dir: Path,
    resume: bool,
    worker,
) -> int:
    jobs = unique_jobs(root)
    if not resume:
        existing = []
        for row in jobs.itertuples(index=False):
            artifact, metadata = artifact_paths(root, method, str(row.subset_id))
            if artifact.exists() or metadata.exists():
                existing.append(str(row.subset_id))
        if existing:
            raise CampaignError(
                f"{method} already has {len(existing)} artifact(s). Use --resume or a fresh output directory."
            )
    failures: list[str] = []
    completed = 0
    skipped = 0
    for row in jobs.itertuples(index=False):
        identifier = str(row.subset_id)
        budget = int(row.budget)
        if resume and is_complete(root, method, identifier):
            skipped += 1
            print(f"{method} N={budget} {identifier}: COMPLETE, skipped", flush=True)
            continue
        name = execution_name(method, root, identifier, budget)
        started = time.monotonic()
        status = "OK"
        try:
            worker(identifier, budget, str(row.state_indices))
            duration = time.monotonic() - started
            completion_record(root, method, identifier, budget, duration)
            completed += 1
            print(f"{name}: OK in {duration:.1f} s", flush=True)
        except Exception as exc:
            duration = time.monotonic() - started
            status = "FAIL"
            failures.append(identifier)
            failure_record(root, method, identifier, budget, exc)
            print(f"{name}: FAIL after {duration:.1f} s: {exc}", flush=True)
        notify(package_dir, name, status, duration)
    method_summary(root, method, failures, completed, skipped)
    return 0 if not failures else 1

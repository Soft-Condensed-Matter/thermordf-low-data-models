#!/usr/bin/env python3
"""Fit the independent density EOS surrogate for every unique subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import thermordf_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit all fresh ThermoRDF EOS models.")
    parser.add_argument("netcdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("thermordf_low_data_campaign"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def fit_one(data: common.DataBundle, root: Path, identifier: str, budget: int, text: str) -> None:
    selected, evaluation = common.selected_and_evaluation(data, text, budget)
    mean = float(np.mean(data.density[selected]))
    scale = float(np.std(data.density[selected], ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise common.CampaignError("Density scaling standard deviation must be positive.")
    knots = common.eos_knots(data)
    design = common.eos_design_matrix(data.eos_coords[selected], knots)
    model = common.relative_ridge_fit(design, (data.density[selected] - mean) / scale)
    artifact, metadata_path = common.artifact_paths(root, "EOS", identifier)
    common.atomic_npz(
        artifact,
        coefficients=np.asarray(model["coefficients"], dtype=np.float64),
        intercept=np.asarray(model["intercept"], dtype=np.float64),
        pressure_knots=knots.astype(np.float64),
        density_mean=np.asarray(mean, dtype=np.float64),
        density_scale=np.asarray(scale, dtype=np.float64),
        ridge_lambda=np.asarray(model["lambda"], dtype=np.float64),
        selected_indices=selected.astype(np.int64),
        evaluation_indices=evaluation.astype(np.int64),
    )
    with np.load(artifact, allow_pickle=False) as saved:
        restored = {name: saved[name].copy() for name in saved.files}
    prediction = common.predict_eos({
        "coefficients": restored["coefficients"],
        "intercept": restored["intercept"],
        "pressure_knots": restored["pressure_knots"],
        "density_mean": float(restored["density_mean"]),
        "density_scale": float(restored["density_scale"]),
    }, data.eos_coords[selected[:1]])
    if prediction.shape != (1,) or not np.all(np.isfinite(prediction)):
        raise common.CampaignError("EOS round-trip validation failed.")
    common.atomic_json(metadata_path, {
        "status": "COMPLETE", "implementation_id": common.IMPLEMENTATION_ID,
        "format_version": common.FORMAT_VERSION, "method": "EOS",
        "subset_id": identifier, "budget": budget,
        "selected_state_ids": data.state_id[selected],
        "logical_aliases": common.aliases_for_subset(root, identifier),
        "model": "cubic_temperature_natural_cubic_pressure_relative_ridge",
        "target_scaling": "training_mean_and_population_standard_deviation",
        "ridge_lambda": float(model["lambda"]),
        "artifact": common.file_identity(artifact),
    })


def main() -> int:
    args = parse_args()
    try:
        data = common.load_data(args.netcdf)
        common.initialise_root(args.output_dir, args.netcdf, data, True)
        result = common.run_independent_jobs(
            args.output_dir, "EOS", Path(__file__).resolve().parent, args.resume,
            lambda identifier, budget, text: fit_one(data, args.output_dir, identifier, budget, text),
        )
        print("STATUS: COMPLETE" if result == 0 else "STATUS: INCOMPLETE", flush=True)
        return result
    except Exception as exc:
        print("STATUS: FAILED", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

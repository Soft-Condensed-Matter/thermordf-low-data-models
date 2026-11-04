#!/usr/bin/env python3
"""Fit and validate every independent joint-output Ridge model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import thermordf_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit all fresh ThermoRDF Ridge models.")
    parser.add_argument("netcdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("thermordf_low_data_campaign"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def fit_one(data: common.DataBundle, root: Path, identifier: str, budget: int, text: str) -> None:
    selected, evaluation = common.selected_and_evaluation(data, text, budget)
    train_x = data.coords_range[selected]
    model = common.relative_ridge_fit(train_x, data.rdf_target[selected])
    artifact, metadata_path = common.artifact_paths(root, "Ridge", identifier)
    common.atomic_npz(
        artifact,
        coefficients=np.asarray(model["coefficients"], dtype=np.float64),
        intercept=np.asarray(model["intercept"], dtype=np.float64),
        ridge_lambda=np.asarray(model["lambda"], dtype=np.float64),
        selected_indices=selected.astype(np.int64),
        evaluation_indices=evaluation.astype(np.int64),
        radial=data.radial.astype(np.float64),
    )
    with np.load(artifact, allow_pickle=False) as saved:
        prediction = common.linear_predict(
            {"coefficients": saved["coefficients"], "intercept": saved["intercept"]},
            train_x[:1],
        )
    if prediction.shape != (1, common.RDF_DIMENSION) or not np.all(np.isfinite(prediction)):
        raise common.CampaignError("Ridge round-trip validation failed.")
    common.atomic_json(metadata_path, {
        "status": "COMPLETE", "implementation_id": common.IMPLEMENTATION_ID,
        "format_version": common.FORMAT_VERSION, "method": "Ridge",
        "subset_id": identifier, "budget": budget,
        "selected_state_ids": data.state_id[selected],
        "logical_aliases": common.aliases_for_subset(root, identifier),
        "input_coordinates": "global_minmax_temperature_pressure",
        "relative_factor": common.RIDGE_RELATIVE_FACTOR,
        "ridge_lambda": float(model["lambda"]),
        "joint_output_dimension": common.RDF_DIMENSION,
        "channel_order": common.CHANNELS,
        "artifact": common.file_identity(artifact),
    })


def main() -> int:
    args = parse_args()
    try:
        data = common.load_data(args.netcdf)
        common.initialise_root(args.output_dir, args.netcdf, data, True)
        result = common.run_independent_jobs(
            args.output_dir, "Ridge", Path(__file__).resolve().parent, args.resume,
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

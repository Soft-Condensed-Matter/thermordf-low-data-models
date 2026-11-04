#!/usr/bin/env python3
"""Create and validate every independent IDW reconstruction artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import thermordf_common as common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare all fresh ThermoRDF IDW models.")
    parser.add_argument("netcdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("thermordf_low_data_campaign"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def fit_one(data: common.DataBundle, root: Path, identifier: str, budget: int, text: str) -> None:
    selected, evaluation = common.selected_and_evaluation(data, text, budget)
    artifact, metadata_path = common.artifact_paths(root, "IDW", identifier)
    common.atomic_npz(
        artifact,
        train_inputs=data.coords_range[selected].astype(np.float64),
        train_rdf_target=data.rdf_target[selected].astype(np.float32),
        selected_indices=selected.astype(np.int64),
        evaluation_indices=evaluation.astype(np.int64),
        power=np.asarray(common.IDW_POWER, dtype=np.float64),
        radial=data.radial.astype(np.float64),
    )
    with np.load(artifact, allow_pickle=False) as saved:
        prediction, nearest = common.idw_predict(
            saved["train_inputs"], saved["train_inputs"][:1], saved["train_rdf_target"]
        )
    if prediction.shape != (1, common.RDF_DIMENSION) or not np.all(np.isfinite(prediction)):
        raise common.CampaignError("IDW round-trip validation failed.")
    if not np.isclose(float(nearest[0]), 0.0, atol=1.0e-15):
        raise common.CampaignError("IDW saved training state is not reproduced exactly.")
    common.atomic_json(metadata_path, {
        "status": "COMPLETE", "implementation_id": common.IMPLEMENTATION_ID,
        "format_version": common.FORMAT_VERSION, "method": "IDW",
        "subset_id": identifier, "budget": budget,
        "selected_state_ids": data.state_id[selected],
        "logical_aliases": common.aliases_for_subset(root, identifier),
        "input_coordinates": "global_minmax_temperature_pressure",
        "power": common.IDW_POWER,
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
            args.output_dir, "IDW", Path(__file__).resolve().parent, args.resume,
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

#!/usr/bin/env python3
"""Evaluate every saved model without fitting or changing model artifacts."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd

import thermordf_common as common
import thermordf_inference as inference


METHODS = ("IDW", "Ridge", "MLP")
FEATURES = ("first_peak_position", "first_peak_height", "first_minimum_position", "first_minimum_value")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the completed modular ThermoRDF campaign.")
    parser.add_argument("netcdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("thermordf_low_data_campaign"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def feature_windows(reference: np.ndarray, radial: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
    result = {}
    mean_curve = np.mean(reference, axis=0)
    spacing = float(np.median(np.diff(radial)))
    for channel_index, channel in enumerate(common.CHANNELS):
        candidates = np.flatnonzero((radial >= 0.10) & (radial <= 0.65))
        peak_center = int(candidates[np.argmax(mean_curve[channel_index, candidates])])
        peak_half = max(4, int(round(0.045 / spacing)))
        peak_lo = max(0, peak_center - peak_half)
        peak_hi = min(radial.size, peak_center + peak_half + 1)
        candidates = np.flatnonzero(
            (radial >= radial[peak_center] + 0.025)
            & (radial <= min(radial[-1], radial[peak_center] + 0.45))
        )
        if candidates.size == 0:
            raise common.CampaignError(f"No first-minimum window for {channel}.")
        minimum_center = int(candidates[np.argmin(mean_curve[channel_index, candidates])])
        minimum_half = max(5, int(round(0.065 / spacing)))
        minimum_lo = max(peak_hi, minimum_center - minimum_half)
        minimum_hi = min(radial.size, minimum_center + minimum_half + 1)
        result[channel] = (peak_lo, peak_hi, minimum_lo, minimum_hi)
    return result


def extract_features(curves: np.ndarray, radial: np.ndarray, windows) -> Dict[str, np.ndarray]:
    count = curves.shape[0]
    values = {name: np.empty((count, len(common.CHANNELS)), dtype=float) for name in FEATURES}
    rows = np.arange(count)
    for channel_index, channel in enumerate(common.CHANNELS):
        p0, p1, m0, m1 = windows[channel]
        peaks = np.argmax(curves[:, channel_index, p0:p1], axis=1) + p0
        minima = np.argmin(curves[:, channel_index, m0:m1], axis=1) + m0
        values["first_peak_position"][:, channel_index] = radial[peaks]
        values["first_peak_height"][:, channel_index] = curves[rows, channel_index, peaks]
        values["first_minimum_position"][:, channel_index] = radial[minima]
        values["first_minimum_value"][:, channel_index] = curves[rows, channel_index, minima]
    return values


def distribution(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def require_all_models(root: Path) -> None:
    missing = []
    for row in common.unique_jobs(root).itertuples(index=False):
        for method in ("EOS", *METHODS):
            if not common.is_complete(root, method, str(row.subset_id)):
                missing.append((method, str(row.subset_id)))
    if missing:
        preview = ", ".join(f"{method}:{identifier}" for method, identifier in missing[:8])
        raise common.CampaignError(f"Evaluation requires all model artifacts; missing {len(missing)} ({preview}).")


def run(args: argparse.Namespace) -> None:
    data = common.load_data(args.netcdf)
    common.initialise_root(args.output_dir, args.netcdf, data, True)
    require_all_models(args.output_dir)
    output = args.output_dir / "evaluation"
    summary_path = output / "evaluation_summary.json"
    if args.resume and summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") == "COMPLETE":
            print("Evaluation already COMPLETE; nothing recomputed.", flush=True)
            return
    windows = feature_windows(data.rdf, data.radial)
    reference_features = extract_features(data.rdf, data.radial, windows)
    unique_rdf_rows = []
    unique_feature_rows = []
    unique_density_rows = []
    diagnostic_rows = []
    tail_count = max(1, int(math.ceil(0.10 * common.RADIAL_BINS)))
    for row in common.unique_jobs(args.output_dir).itertuples(index=False):
        identifier = str(row.subset_id)
        budget = int(row.budget)
        selected, evaluation = common.selected_and_evaluation(data, str(row.state_indices), budget)
        density_prediction = inference.predict_eos(args.output_dir, identifier, data.eos_coords)
        density_error = density_prediction[evaluation] - data.density[evaluation]
        relative = density_error / data.density[evaluation]
        density_scale = float(np.std(data.density[selected], ddof=0))
        unique_density_rows.append({
            "subset_id": identifier, "budget": budget,
            "mean_squared_relative_error": float(np.mean(relative**2)),
            "mean_absolute_relative_error": float(np.mean(np.abs(relative))),
            "p95_absolute_relative_error": float(np.quantile(np.abs(relative), 0.95)),
            "maximum_absolute_relative_error": float(np.max(np.abs(relative))),
            "bias_kg_m3": float(np.mean(density_error)),
            "sigma_normalized_rmse": float(np.sqrt(np.mean((density_error / density_scale) ** 2))),
        })
        predictions = {
            "IDW": inference.predict_idw(args.output_dir, identifier, data.coords_range),
            "Ridge": inference.predict_ridge(args.output_dir, identifier, data.coords_range),
            "MLP": inference.predict_mlp(args.output_dir, identifier, data.coords_range, args.device),
        }
        for method, flat in predictions.items():
            curves = flat.reshape(common.EXPECTED_STATES, len(common.CHANNELS), common.RADIAL_BINS)
            difference = curves - data.rdf
            predicted_features = extract_features(curves, data.radial, windows)
            for channel_index, channel in enumerate(common.CHANNELS):
                rmse = np.sqrt(np.mean(difference[evaluation, channel_index] ** 2, axis=1))
                mae = np.mean(np.abs(difference[evaluation, channel_index]), axis=1)
                tail = np.mean(np.abs(difference[evaluation, channel_index, -tail_count:]), axis=1)
                negative = np.mean(curves[evaluation, channel_index] < 0.0, axis=1)
                for metric, values in (
                    ("curve_rmse", rmse), ("curve_mae", mae),
                    ("tail_mae", tail), ("negative_fraction", negative),
                ):
                    unique_rdf_rows.append({
                        "subset_id": identifier, "budget": budget, "method": method,
                        "channel": channel, "metric": metric, **distribution(values),
                    })
                for feature in FEATURES:
                    reference = reference_features[feature][evaluation, channel_index]
                    predicted = predicted_features[feature][evaluation, channel_index]
                    denominator = np.maximum(np.abs(reference), 1.0e-12)
                    signed_relative = (predicted - reference) / denominator
                    unique_feature_rows.append({
                        "subset_id": identifier, "budget": budget, "method": method,
                        "channel": channel, "property": feature,
                        **{f"absolute_relative_{key}": value for key, value in distribution(np.abs(signed_relative)).items()},
                        "signed_relative_mean": float(np.mean(signed_relative)),
                    })
        mlp_metadata = inference.metadata(args.output_dir, "MLP", identifier)
        diagnostic_rows.append({
            "subset_id": identifier, "budget": budget,
            **dict(mlp_metadata.get("summary", {})),
        })
        print(f"Evaluated N={budget} subset={identifier}", flush=True)
    aliases = common.read_jobs(args.output_dir)
    alias_columns = ["geometry", "strategy", "repetition", "budget", "subset_id"]
    alias_frame = aliases[alias_columns].copy()
    density_frame = alias_frame.merge(pd.DataFrame(unique_density_rows), on=["subset_id", "budget"], how="left")
    rdf_frame = alias_frame.merge(pd.DataFrame(unique_rdf_rows), on=["subset_id", "budget"], how="left")
    feature_frame = alias_frame.merge(pd.DataFrame(unique_feature_rows), on=["subset_id", "budget"], how="left")
    diagnostics = alias_frame.merge(pd.DataFrame(diagnostic_rows), on=["subset_id", "budget"], how="left")
    common.atomic_csv(output / "density_metrics.csv", density_frame)
    common.atomic_csv(output / "rdf_metrics.csv", rdf_frame)
    common.atomic_csv(output / "physical_feature_metrics.csv", feature_frame)
    common.atomic_csv(output / "mlp_diagnostics.csv", diagnostics)
    common.atomic_json(output / "feature_windows.json", {
        channel: {
            "peak_index_start": int(bounds[0]), "peak_index_stop_exclusive": int(bounds[1]),
            "minimum_index_start": int(bounds[2]), "minimum_index_stop_exclusive": int(bounds[3]),
            "peak_radial_interval": [float(data.radial[bounds[0]]), float(data.radial[bounds[1] - 1])],
            "minimum_radial_interval": [float(data.radial[bounds[2]]), float(data.radial[bounds[3] - 1])],
        } for channel, bounds in windows.items()
    })
    common.atomic_json(summary_path, {
        "status": "COMPLETE", "implementation_id": common.IMPLEMENTATION_ID,
        "unique_subsets_evaluated": int(common.unique_jobs(args.output_dir).shape[0]),
        "logical_jobs_reported": int(alias_frame.shape[0]),
        "methods": METHODS, "channels": common.CHANNELS,
        "full_rdf_predictions_stored": False,
    })


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    try:
        run(args)
        duration = time.monotonic() - started
        common.notify(Path(__file__).resolve().parent, "ThermoRDF_EVALUATION", "OK", duration)
        print("STATUS: COMPLETE", flush=True)
        return 0
    except Exception as exc:
        duration = time.monotonic() - started
        common.notify(Path(__file__).resolve().parent, "ThermoRDF_EVALUATION", "FAIL", duration)
        print("STATUS: FAILED", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

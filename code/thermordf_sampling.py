#!/usr/bin/env python3
"""Generate fresh deterministic and control selections for both geometries."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

import thermordf_common as common


@dataclass
class Stratum:
    members: np.ndarray
    selected: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the fresh ThermoRDF sampling campaign.")
    parser.add_argument("netcdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("thermordf_low_data_campaign"))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def exact_corners(coords: np.ndarray) -> np.ndarray:
    result: list[int] = []
    for corner in common.CORNER_ORDER:
        match = np.flatnonzero(
            np.isclose(coords[:, 0], corner[0], rtol=0.0, atol=1.0e-12)
            & np.isclose(coords[:, 1], corner[1], rtol=0.0, atol=1.0e-12)
        )
        if match.size != 1:
            raise common.CampaignError(f"Expected one state at corner {corner}; found {match.size}.")
        result.append(int(match[0]))
    return np.asarray(result, dtype=int)


def deterministic_tie_order(data: common.DataBundle, indices: np.ndarray) -> np.ndarray:
    return np.lexsort((data.state_id[indices], data.pressure[indices], data.temperature[indices]))


def maximin_sequence(
    data: common.DataBundle,
    distance: np.ndarray,
    corners: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    selected = corners.tolist()
    insertion = [math.nan] * len(selected)
    reasons = ["corner_00", "corner_01", "corner_10", "corner_11"]
    chosen = np.zeros(common.EXPECTED_STATES, dtype=bool)
    chosen[corners] = True
    maximum_budget = int(common.BUDGETS[-1])
    while len(selected) < maximum_budget:
        remaining = np.flatnonzero(~chosen)
        nearest = np.min(distance[np.ix_(remaining, np.asarray(selected))], axis=1)
        maximum = float(np.max(nearest))
        tied = remaining[np.isclose(nearest, maximum, rtol=1.0e-12, atol=1.0e-14)]
        next_state = int(tied[deterministic_tie_order(data, tied)[0]])
        selected.append(next_state)
        insertion.append(maximum)
        reasons.append("maximin")
        chosen[next_state] = True
    return np.asarray(selected, dtype=int), np.asarray(insertion), reasons


def leaf_key(leaf: Stratum, coords: np.ndarray, data: common.DataBundle):
    points = coords[leaf.members]
    ranges = np.ptp(points, axis=0)
    diagonal = float(np.sqrt(np.sum(ranges * ranges)))
    return (
        -diagonal,
        -leaf.members.size,
        float(np.min(data.temperature[leaf.members])),
        float(np.min(data.pressure[leaf.members])),
        min(str(value) for value in data.state_id[leaf.members]),
    )


def split_stratum(leaf: Stratum, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    members = leaf.members
    points = coords[members]
    ranges = np.ptp(points, axis=0)
    dimension_order = [0, 1] if ranges[0] >= ranges[1] else [1, 0]
    for dimension in dimension_order:
        unique = np.unique(points[:, dimension])
        if unique.size < 2:
            continue
        target = members.size / 2.0
        candidates = []
        for left_value in unique[:-1]:
            count = int(np.count_nonzero(points[:, dimension] <= left_value))
            candidates.append((abs(count - target), float(left_value)))
        _, cut = min(candidates, key=lambda item: (item[0], item[1]))
        left = members[points[:, dimension] <= cut]
        right = members[points[:, dimension] > cut]
        if left.size and right.size:
            return left, right
    raise common.CampaignError(f"Could not split stratum with {members.size} states.")


def stratified_sequence(
    data: common.DataBundle,
    coords: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    all_members = np.arange(common.EXPECTED_STATES, dtype=int)
    initial = int(rng.choice(all_members))
    leaves = [Stratum(all_members, initial)]
    sequence = [initial]
    maximum_budget = int(common.BUDGETS[-1])
    while len(sequence) < maximum_budget:
        splittable = [index for index, leaf in enumerate(leaves) if leaf.members.size > 1]
        chosen_index = min(splittable, key=lambda index: leaf_key(leaves[index], coords, data))
        chosen_leaf = leaves.pop(chosen_index)
        left, right = split_stratum(chosen_leaf, coords)
        if chosen_leaf.selected in left:
            retained_members, new_members = left, right
        else:
            retained_members, new_members = right, left
        new_selected = int(rng.choice(new_members))
        leaves.extend((Stratum(retained_members, chosen_leaf.selected), Stratum(new_members, new_selected)))
        sequence.append(new_selected)
    return np.asarray(sequence, dtype=int)


def generate_controls(
    data: common.DataBundle,
    corners: np.ndarray,
    seeds: Sequence[np.random.SeedSequence],
) -> Dict[str, np.ndarray]:
    maximum_budget = int(common.BUDGETS[-1])
    all_indices = np.arange(common.EXPECTED_STATES, dtype=int)
    non_corners = np.setdiff1d(all_indices, corners, assume_unique=True)
    shape = (common.GEOMETRIC_REPETITIONS, maximum_budget)
    controls = {
        "random_unrestricted": np.empty(shape, dtype=int),
        "random_corners": np.empty(shape, dtype=int),
        "stratified_range": np.empty(shape, dtype=int),
        "stratified_grid_resolution": np.empty(shape, dtype=int),
    }
    rng_unrestricted = np.random.default_rng(seeds[0])
    rng_corners = np.random.default_rng(seeds[1])
    rng_range = np.random.default_rng(seeds[2])
    rng_grid = np.random.default_rng(seeds[3])
    for repetition in range(common.GEOMETRIC_REPETITIONS):
        controls["random_unrestricted"][repetition] = rng_unrestricted.permutation(all_indices)[:maximum_budget]
        controls["random_corners"][repetition] = np.concatenate(
            (corners, rng_corners.permutation(non_corners)[: maximum_budget - corners.size])
        )
        controls["stratified_range"][repetition] = stratified_sequence(data, data.coords_range, rng_range)
        controls["stratified_grid_resolution"][repetition] = stratified_sequence(data, data.coords_grid, rng_grid)
    return controls


def coverage_metrics(distance: np.ndarray, selected: np.ndarray) -> Dict[str, float]:
    nearest = np.min(distance[:, selected], axis=1)
    internal = distance[np.ix_(selected, selected)].copy()
    np.fill_diagonal(internal, np.inf)
    return {
        "fill_distance": float(np.max(nearest)),
        "separation_distance": float(np.min(internal)),
        "mean_nearest_distance": float(np.mean(nearest)),
    }


def design_frame(data: common.DataBundle, designs) -> pd.DataFrame:
    frames = []
    maximum_budget = int(common.BUDGETS[-1])
    for geometry, (sequence, insertion, reasons) in designs.items():
        frames.append(pd.DataFrame({
            "geometry": geometry,
            "selection_order": np.arange(1, maximum_budget + 1),
            "source_index": sequence,
            "state_id": data.state_id[sequence],
            "temperature": data.temperature[sequence],
            "pressure": data.pressure[sequence],
            "selection_reason": reasons,
            "insertion_distance": insertion,
        }))
    return pd.concat(frames, ignore_index=True)


def controls_frame(data: common.DataBundle, controls: Dict[str, np.ndarray]) -> pd.DataFrame:
    frames = []
    maximum_budget = int(common.BUDGETS[-1])
    for strategy, sequences in controls.items():
        geometry_basis = "shared"
        if strategy == "stratified_range":
            geometry_basis = "range"
        elif strategy == "stratified_grid_resolution":
            geometry_basis = "grid_resolution"
        for repetition, sequence in enumerate(sequences, start=1):
            frames.append(pd.DataFrame({
                "strategy": strategy,
                "geometry_basis": geometry_basis,
                "repetition": repetition,
                "selection_order": np.arange(1, maximum_budget + 1),
                "source_index": sequence,
                "state_id": data.state_id[sequence],
                "temperature": data.temperature[sequence],
                "pressure": data.pressure[sequence],
            }))
    return pd.concat(frames, ignore_index=True)


def metrics_frame(designs, controls, distance_range, distance_grid) -> pd.DataFrame:
    records = []
    for geometry, distance in (("range", distance_range), ("grid_resolution", distance_grid)):
        design = designs[geometry][0]
        applicable = {
            "random_unrestricted": controls["random_unrestricted"],
            "random_corners": controls["random_corners"],
            "stratified_recursive": controls[f"stratified_{geometry}"],
        }
        for budget in common.BUDGETS:
            records.append({
                "geometry": geometry, "strategy": "designed_maximin", "repetition": 0,
                "budget": int(budget), **coverage_metrics(distance, design[: int(budget)]),
            })
        for strategy, sequences in applicable.items():
            for repetition, sequence in enumerate(sequences, start=1):
                for budget in common.BUDGETS:
                    records.append({
                        "geometry": geometry, "strategy": strategy, "repetition": repetition,
                        "budget": int(budget), **coverage_metrics(distance, sequence[: int(budget)]),
                    })
    return pd.DataFrame.from_records(records)


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> Tuple[float, float]:
    indices = rng.integers(0, values.size, size=(common.BOOTSTRAP_RESAMPLES, values.size))
    means = np.mean(values[indices], axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def summarise_metrics(raw: pd.DataFrame, seed: np.random.SeedSequence) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    summaries = []
    joint = []
    for geometry in ("range", "grid_resolution"):
        design = raw[(raw.geometry == geometry) & (raw.strategy == "designed_maximin")].set_index("budget")
        for strategy in ("random_unrestricted", "random_corners", "stratified_recursive"):
            family = raw[(raw.geometry == geometry) & (raw.strategy == strategy)]
            for budget in common.BUDGETS:
                rows = family[family.budget == int(budget)].sort_values("repetition")
                design_row = design.loc[int(budget)]
                for metric in common.METRIC_COLUMNS:
                    values = rows[metric].to_numpy(float)
                    low, high = bootstrap_interval(values, rng)
                    target = float(design_row[metric])
                    better = values >= target if metric == "separation_distance" else values <= target
                    summaries.append({
                        "geometry": geometry, "strategy": strategy, "budget": int(budget),
                        "metric": metric, "n_repetitions": values.size,
                        "mean": float(np.mean(values)), "standard_deviation": float(np.std(values, ddof=1)),
                        "median": float(np.median(values)), "percentile_05": float(np.quantile(values, 0.05)),
                        "percentile_95": float(np.quantile(values, 0.95)),
                        "bootstrap_mean_ci_95_low": low, "bootstrap_mean_ci_95_high": high,
                        "designed_value": target, "probability_match_or_exceed_designed": float(np.mean(better)),
                    })
                dominated = (
                    (rows.fill_distance.to_numpy() >= float(design_row.fill_distance))
                    & (rows.mean_nearest_distance.to_numpy() >= float(design_row.mean_nearest_distance))
                    & (rows.separation_distance.to_numpy() <= float(design_row.separation_distance))
                )
                joint.append({
                    "geometry": geometry, "strategy": strategy, "budget": int(budget),
                    "n_repetitions": len(rows),
                    "probability_geometrically_dominated_by_designed": float(np.mean(dominated)),
                })
    return pd.DataFrame(summaries), pd.DataFrame(joint)


def model_jobs(data: common.DataBundle, designs, controls) -> pd.DataFrame:
    records = []
    for geometry in ("range", "grid_resolution"):
        families = {
            "designed_maximin": designs[geometry][0][None, :],
            "random_unrestricted": controls["random_unrestricted"][: common.PREDICTIVE_REPETITIONS],
            "random_corners": controls["random_corners"][: common.PREDICTIVE_REPETITIONS],
            "stratified_recursive": controls[f"stratified_{geometry}"][: common.PREDICTIVE_REPETITIONS],
        }
        for strategy, sequences in families.items():
            for repetition_index, sequence in enumerate(sequences):
                repetition = 0 if strategy == "designed_maximin" else repetition_index + 1
                for budget in common.BUDGETS:
                    selected = np.asarray(sequence[: int(budget)], dtype=int)
                    records.append({
                        "geometry": geometry,
                        "strategy": strategy,
                        "repetition": repetition,
                        "budget": int(budget),
                        "subset_id": common.subset_id(selected, data.state_id),
                        "state_indices": ";".join(str(int(value)) for value in selected),
                        "state_ids": ";".join(str(data.state_id[value]) for value in selected),
                    })
    return pd.DataFrame(records).sort_values(["budget", "geometry", "strategy", "repetition"]).reset_index(drop=True)


def run(args: argparse.Namespace) -> None:
    started = time.monotonic()
    data = common.load_data(args.netcdf)
    common.initialise_root(args.output_dir, args.netcdf, data, args.resume)
    summary_path = args.output_dir / "sampling" / "sampling_run_summary.json"
    if args.resume and summary_path.is_file():
        stored = json.loads(summary_path.read_text(encoding="utf-8"))
        if stored.get("status") == "COMPLETE":
            print("Sampling already COMPLETE; nothing regenerated.", flush=True)
            return
    distance_range = common.pairwise_distance(data.coords_range)
    distance_grid = common.pairwise_distance(data.coords_grid)
    corners = exact_corners(data.coords_range)
    designs = {
        "range": maximin_sequence(data, distance_range, corners),
        "grid_resolution": maximin_sequence(data, distance_grid, corners),
    }
    seeds = np.random.SeedSequence(common.MASTER_SEED).spawn(5)
    controls = generate_controls(data, corners, seeds[:4])
    raw = metrics_frame(designs, controls, distance_range, distance_grid)
    summary, joint = summarise_metrics(raw, seeds[4])
    output = args.output_dir / "sampling"
    common.atomic_csv(output / "designed_sequences.csv", design_frame(data, designs))
    common.atomic_csv(output / "control_sequences.csv", controls_frame(data, controls))
    common.atomic_csv(output / "coverage_metrics_raw.csv", raw)
    common.atomic_csv(output / "coverage_metrics_summary.csv", summary)
    common.atomic_csv(output / "control_joint_probabilities.csv", joint)
    jobs = model_jobs(data, designs, controls)
    common.atomic_csv(output / "model_jobs.csv", jobs)
    common.atomic_json(summary_path, {
        "status": "COMPLETE", "implementation_id": common.IMPLEMENTATION_ID,
        "candidate_count": common.EXPECTED_STATES, "budgets": common.BUDGETS,
        "geometric_repetitions": common.GEOMETRIC_REPETITIONS,
        "predictive_repetitions": common.PREDICTIVE_REPETITIONS,
        "geometries": ["range", "grid_resolution"],
        "control_families": ["random_unrestricted", "random_corners", "stratified_recursive"],
        "grid_pressure_weight": data.lambda_pressure_grid,
        "logical_model_jobs": int(len(jobs)),
        "unique_training_subsets": int(jobs.subset_id.nunique()),
        "duration_seconds": time.monotonic() - started,
    })


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    try:
        run(args)
        duration = time.monotonic() - started
        common.notify(Path(__file__).resolve().parent, "ThermoRDF_SAMPLING", "OK", duration)
        print("STATUS: COMPLETE", flush=True)
        print(f"Outputs: {args.output_dir.resolve()}", flush=True)
        return 0
    except Exception as exc:
        duration = time.monotonic() - started
        common.notify(Path(__file__).resolve().parent, "ThermoRDF_SAMPLING", "FAIL", duration)
        print("STATUS: FAILED", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

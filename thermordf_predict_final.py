#!/usr/bin/env python3
"""Predict density and four RDF channels from the released final models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("netcdf", type=Path, help="thermordf_master.nc from the Zenodo record")
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--pressure", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = args.repository.expanduser().resolve()
    code_dir = repository / "code"
    sys.path.insert(0, str(code_dir))
    import thermordf_common as common
    import thermordf_inference as inference

    manifest = json.loads((repository / "release_manifest.json").read_text(encoding="utf-8"))
    expected_data = manifest["source_netcdf"]
    if sha256_file(args.netcdf) != expected_data["sha256"]:
        raise RuntimeError("The NetCDF SHA-256 does not match the dataset used for training.")
    data = common.load_data(args.netcdf)
    subset = manifest["selection"]["subset_id"]
    t_min, t_max = float(np.min(data.temperature)), float(np.max(data.temperature))
    p_min, p_max = float(np.min(data.pressure)), float(np.max(data.pressure))
    if not (t_min <= args.temperature <= t_max):
        raise RuntimeError(f"Temperature lies outside [{t_min:g}, {t_max:g}] K.")
    if not (p_min <= args.pressure <= p_max):
        raise RuntimeError(f"Pressure lies outside [{p_min:g}, {p_max:g}] bar.")
    query = np.asarray([[
        (args.temperature - t_min) / (t_max - t_min),
        (args.pressure - p_min) / (p_max - p_min),
    ]], dtype=float)
    density = float(inference.predict_eos(repository, subset, 2.0 * query - 1.0)[0])
    predictions = {
        "IDW": inference.predict_idw(repository, subset, query),
        "Ridge": inference.predict_ridge(repository, subset, query),
        "MLP": inference.predict_mlp(repository, subset, query, args.device),
    }
    frame = pd.DataFrame({f"r_{data.units['radial']}": data.radial})
    for method, flat in predictions.items():
        curves = flat[0].reshape(len(common.CHANNELS), common.RADIAL_BINS)
        for channel_index, channel in enumerate(common.CHANNELS):
            frame[f"g_{method}_{channel}"] = curves[channel_index]
    metadata = {
        "status": "COMPLETE",
        "subset_id": subset,
        "temperature": args.temperature,
        "temperature_units": data.units["temperature"],
        "pressure": args.pressure,
        "pressure_units": data.units["pressure"],
        "density_prediction": density,
        "density_units": data.units["density"],
        "methods": list(predictions),
        "channels": list(common.CHANNELS),
    }
    exact = np.flatnonzero(
        np.isclose(data.temperature, args.temperature)
        & np.isclose(data.pressure, args.pressure)
    )
    if exact.size == 1:
        state = int(exact[0])
        metadata["matching_state_id"] = str(data.state_id[state])
        metadata["density_reference"] = float(data.density[state])
        for channel_index, channel in enumerate(common.CHANNELS):
            frame[f"g_reference_{channel}"] = data.rdf[state, channel_index]
    output = args.output.expanduser().resolve()
    metadata_output = output.with_suffix(".json")
    if output.exists() or metadata_output.exists():
        raise FileExistsError(f"Output already exists: {output} or {metadata_output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("STATUS: COMPLETE")
    print(f"Prediction: {output}")
    print(f"Metadata: {metadata_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("STATUS: FAILED", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

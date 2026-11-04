#!/usr/bin/env python3
"""Load saved ThermoRDF artifacts and perform inference without refitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

import thermordf_common as common


def require_artifact(root: Path, method: str, identifier: str) -> Path:
    artifact, metadata = common.artifact_paths(root, method, identifier)
    if not common.is_complete(root, method, identifier):
        raise common.CampaignError(f"Incomplete {method} artifact for subset {identifier}.")
    return artifact


def predict_eos(root: Path, identifier: str, coords: np.ndarray) -> np.ndarray:
    artifact = require_artifact(root, "EOS", identifier)
    with np.load(artifact, allow_pickle=False) as saved:
        model = {
            "coefficients": saved["coefficients"],
            "intercept": saved["intercept"],
            "pressure_knots": saved["pressure_knots"],
            "density_mean": float(saved["density_mean"]),
            "density_scale": float(saved["density_scale"]),
        }
        return common.predict_eos(model, coords)


def predict_idw(root: Path, identifier: str, coords: np.ndarray) -> np.ndarray:
    artifact = require_artifact(root, "IDW", identifier)
    with np.load(artifact, allow_pickle=False) as saved:
        prediction, _ = common.idw_predict(saved["train_inputs"], coords, saved["train_rdf_target"])
    return prediction


def predict_ridge(root: Path, identifier: str, coords: np.ndarray) -> np.ndarray:
    artifact = require_artifact(root, "Ridge", identifier)
    with np.load(artifact, allow_pickle=False) as saved:
        return np.asarray(common.linear_predict({
            "coefficients": saved["coefficients"], "intercept": saved["intercept"]
        }, coords), dtype=float)


def predict_mlp(root: Path, identifier: str, coords: np.ndarray, device_name: str) -> np.ndarray:
    import torch
    from thermordf_mlp import configure_device, predict_state

    artifact = require_artifact(root, "MLP", identifier)
    device = configure_device(device_name)
    tensors = torch.load(artifact, map_location="cpu", weights_only=True)
    return predict_state(tensors["state_dict"], coords, device).astype(float)


def selected_indices(root: Path, method: str, identifier: str) -> np.ndarray:
    artifact = require_artifact(root, method, identifier)
    if method.upper() == "MLP":
        import torch
        tensors = torch.load(artifact, map_location="cpu", weights_only=True)
        return tensors["selected_indices"].cpu().numpy().copy()
    with np.load(artifact, allow_pickle=False) as saved:
        return saved["selected_indices"].copy()


def metadata(root: Path, method: str, identifier: str) -> Mapping[str, object]:
    _, path = common.artifact_paths(root, method, identifier)
    return json.loads(path.read_text(encoding="utf-8"))

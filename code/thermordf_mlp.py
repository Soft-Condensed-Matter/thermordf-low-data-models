#!/usr/bin/env python3
"""Train every independent deterministic joint-output MLP."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Mapping, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

import thermordf_common as common


class JointRDFMLP(nn.Module):
    def __init__(self, output_mean: torch.Tensor) -> None:
        super().__init__()
        widths = (2, *common.MLP_HIDDEN_WIDTHS)
        self.hidden_layers = nn.ModuleList(
            nn.Linear(widths[index], widths[index + 1], bias=True)
            for index in range(len(widths) - 1)
        )
        self.output = nn.Linear(widths[-1], common.RDF_DIMENSION, bias=True)
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output.weight, gain=1.0)
        with torch.no_grad():
            self.output.bias.copy_(output_mean)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current = inputs
        for layer in self.hidden_layers:
            current = torch.tanh(layer(current))
        return self.output(current)

    def activations(self, inputs: torch.Tensor):
        result = []
        current = inputs
        for layer in self.hidden_layers:
            current = torch.tanh(layer(current))
            result.append(current)
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all fresh ThermoRDF MLP models.")
    parser.add_argument("netcdf", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("thermordf_low_data_campaign"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def configure_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise common.CampaignError("CUDA was requested but is unavailable. Use --device cpu explicitly.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(name)


def reset_seed() -> None:
    random.seed(common.MASTER_SEED)
    np.random.seed(common.MASTER_SEED)
    torch.manual_seed(common.MASTER_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(common.MASTER_SEED)


def loss_components(model: JointRDFMLP, prediction: torch.Tensor, target: torch.Tensor):
    error = (prediction - target).reshape(target.shape[0], len(common.CHANNELS), common.RADIAL_BINS)
    channel_terms = torch.mean(error.square(), dim=(0, 2))
    data_loss = torch.mean(channel_terms)
    weight_square = torch.sum(model.output.weight.square())
    for layer in model.hidden_layers:
        weight_square = weight_square + torch.sum(layer.weight.square())
    l2 = common.MLP_L2_ALPHA * weight_square
    return data_loss + l2, data_loss, l2


def snapshot(model: JointRDFMLP) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def activation_diagnostics(values: np.ndarray) -> np.ndarray:
    saturation = float(np.mean(np.abs(values) >= 0.95))
    mean_absolute = float(np.mean(np.abs(values)))
    centred = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centred, compute_uv=False)
    energy = singular**2
    if np.any(energy > 0.0):
        probabilities = energy[energy > 0.0] / np.sum(energy[energy > 0.0])
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    else:
        effective_rank = 0.0
    maximum_rank = float(max(1, min(values.shape[1], values.shape[0] - 1)))
    return np.asarray([saturation, mean_absolute, effective_rank, effective_rank / maximum_rank])


def channel_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    difference = (prediction - target).reshape(-1, len(common.CHANNELS), common.RADIAL_BINS)
    return np.mean(difference**2, axis=(0, 2))


def gradient_diagnostics(model: JointRDFMLP, inputs: torch.Tensor, target: torch.Tensor):
    model.train()
    model.zero_grad(set_to_none=True)
    total, _, _ = loss_components(model, model(inputs), target)
    total.backward()
    hidden_square = np.zeros(len(model.hidden_layers), dtype=float)
    output_square = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise common.CampaignError(f"Missing gradient for {name}.")
        value = float(torch.sum(parameter.grad.detach() ** 2).cpu())
        if name.startswith("hidden_layers."):
            hidden_square[int(name.split(".")[1])] += value
        else:
            output_square += value
    model.eval()
    return math.sqrt(float(np.sum(hidden_square)) + output_square), np.sqrt(hidden_square), math.sqrt(output_square)


def train(
    data: common.DataBundle,
    selected: np.ndarray,
    evaluation: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object], Dict[str, np.ndarray]]:
    reset_seed()
    x_train = torch.as_tensor(data.coords_range[selected], dtype=torch.float32, device=device)
    y_train = torch.as_tensor(data.rdf_target[selected], dtype=torch.float32, device=device)
    x_eval = torch.as_tensor(data.coords_range[evaluation], dtype=torch.float32, device=device)
    y_eval = data.rdf_target[evaluation]
    output_mean = torch.mean(y_train, dim=0).detach().cpu()
    model = JointRDFMLP(output_mean).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=common.MLP_LEARNING_RATE, betas=(0.9, 0.999),
        eps=1.0e-8, weight_decay=0.0, amsgrad=False,
    )
    history_total = np.full(common.MLP_MAX_EPOCHS + 1, np.nan, dtype=np.float32)
    history_data = np.full_like(history_total, np.nan)
    history_l2 = np.full_like(history_total, np.nan)
    model.train()
    with torch.no_grad():
        initial = loss_components(model, model(x_train), y_train)
    best_total, best_data, best_l2 = [float(value.detach().cpu()) for value in initial]
    history_total[0], history_data[0], history_l2[0] = best_total, best_data, best_l2
    best_epoch = 0
    best_state = snapshot(model)
    without_improvement = 0
    termination = "LIMIT_REACHED"
    epochs_run = 0
    for epoch in range(1, common.MLP_MAX_EPOCHS + 1):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = loss_components(model, model(x_train), y_train)
        if not bool(torch.isfinite(total).item()):
            raise common.CampaignError(f"Non-finite loss at epoch {epoch}.")
        total.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not bool(torch.all(torch.isfinite(parameter.grad)).item()):
                raise common.CampaignError(f"Invalid gradient for {name} at epoch {epoch}.")
        optimizer.step()
        with torch.no_grad():
            current_t = loss_components(model, model(x_train), y_train)
        current = [float(value.detach().cpu()) for value in current_t]
        history_total[epoch], history_data[epoch], history_l2[epoch] = current
        threshold = common.MLP_RELATIVE_MINIMUM_IMPROVEMENT * max(abs(best_total), 1.0e-12)
        if best_total - current[0] > threshold:
            best_total, best_data, best_l2 = current
            best_epoch = epoch
            best_state = snapshot(model)
            without_improvement = 0
        else:
            without_improvement += 1
        epochs_run = epoch
        if without_improvement >= common.MLP_PATIENCE:
            termination = "PLATEAU"
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_prediction = model(x_train).detach().cpu().numpy()
        evaluation_prediction = model(x_eval).detach().cpu().numpy()
        train_activations = [value.detach().cpu().numpy() for value in model.activations(x_train)]
        eval_activations = [value.detach().cpu().numpy() for value in model.activations(x_eval)]
    gradient_total, gradient_hidden, gradient_output = gradient_diagnostics(model, x_train, y_train)
    summary = {
        "termination_reason": termination,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_total_loss": best_total,
        "best_data_loss": best_data,
        "best_l2_contribution": best_l2,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "gradient_norm_total": gradient_total,
        "gradient_norm_output": gradient_output,
        "output_weight_frobenius": float(torch.linalg.vector_norm(model.output.weight.detach()).cpu()),
    }
    arrays = {
        "history_total": history_total[: epochs_run + 1],
        "history_data": history_data[: epochs_run + 1],
        "history_l2": history_l2[: epochs_run + 1],
        "train_channel_mse": channel_mse(train_prediction, data.rdf_target[selected]),
        "evaluation_channel_mse": channel_mse(evaluation_prediction, y_eval),
        "activation_train": np.asarray([activation_diagnostics(values) for values in train_activations]),
        "activation_evaluation": np.asarray([activation_diagnostics(values) for values in eval_activations]),
        "gradient_hidden": gradient_hidden,
        "hidden_weight_frobenius": np.asarray([
            float(torch.linalg.vector_norm(layer.weight.detach()).cpu()) for layer in model.hidden_layers
        ]),
    }
    return best_state, summary, arrays


def atomic_torch(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def predict_state(state: Mapping[str, torch.Tensor], inputs: np.ndarray, device: torch.device) -> np.ndarray:
    model = JointRDFMLP(torch.zeros(common.RDF_DIMENSION, dtype=torch.float32)).to(device=device, dtype=torch.float32)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        result = model(torch.as_tensor(inputs, dtype=torch.float32, device=device))
    return result.detach().cpu().numpy()


def fit_one(
    data: common.DataBundle,
    root: Path,
    device: torch.device,
    identifier: str,
    budget: int,
    text: str,
) -> None:
    selected, evaluation = common.selected_and_evaluation(data, text, budget)
    state, summary, arrays = train(data, selected, evaluation, device)
    artifact, metadata_path = common.artifact_paths(root, "MLP", identifier)
    tensor_payload = {
        "format_version": torch.tensor(common.FORMAT_VERSION, dtype=torch.int64),
        "selected_indices": torch.as_tensor(selected, dtype=torch.int64),
        "evaluation_indices": torch.as_tensor(evaluation, dtype=torch.int64),
        "state_dict": state,
        "diagnostics": {name: torch.as_tensor(value).clone() for name, value in arrays.items()},
    }
    atomic_torch(artifact, tensor_payload)
    restored = torch.load(artifact, map_location="cpu", weights_only=True)
    if int(restored["format_version"].item()) != common.FORMAT_VERSION:
        raise common.CampaignError("MLP checkpoint version validation failed.")
    test = predict_state(restored["state_dict"], data.coords_range[selected[:1]], device)
    if test.shape != (1, common.RDF_DIMENSION) or not np.all(np.isfinite(test)):
        raise common.CampaignError("MLP round-trip prediction validation failed.")
    common.atomic_json(metadata_path, {
        "status": "COMPLETE", "implementation_id": common.IMPLEMENTATION_ID,
        "format_version": common.FORMAT_VERSION, "method": "MLP",
        "subset_id": identifier, "budget": budget,
        "selected_state_ids": data.state_id[selected],
        "logical_aliases": common.aliases_for_subset(root, identifier),
        "input_coordinates": "global_minmax_temperature_pressure",
        "architecture": [2, *common.MLP_HIDDEN_WIDTHS, common.RDF_DIMENSION],
        "activation": "tanh", "dtype": "float32", "device": device.type,
        "learning_rate": common.MLP_LEARNING_RATE, "l2_alpha": common.MLP_L2_ALPHA,
        "maximum_epochs": common.MLP_MAX_EPOCHS, "patience": common.MLP_PATIENCE,
        "relative_minimum_improvement": common.MLP_RELATIVE_MINIMUM_IMPROVEMENT,
        "seed": common.MASTER_SEED, "summary": summary,
        "diagnostic_arrays": sorted(arrays),
        "joint_output_dimension": common.RDF_DIMENSION,
        "channel_order": common.CHANNELS,
        "artifact": common.file_identity(artifact),
    })
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    try:
        device = configure_device(args.device)
        data = common.load_data(args.netcdf)
        common.initialise_root(args.output_dir, args.netcdf, data, True)
        result = common.run_independent_jobs(
            args.output_dir, "MLP", Path(__file__).resolve().parent, args.resume,
            lambda identifier, budget, text: fit_one(
                data, args.output_dir, device, identifier, budget, text
            ),
        )
        print("STATUS: COMPLETE" if result == 0 else "STATUS: INCOMPLETE", flush=True)
        return result
    except Exception as exc:
        print("STATUS: FAILED", file=sys.stderr)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

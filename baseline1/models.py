from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn

from model9_NS_transformer.diffusion_models.diffuMTS import Model as D3UDiffusion
from baseline1.runtime import resolve_path, to_namespace


class ConditionOnlyBackbone(nn.Module):
    """Deterministic D3U condition network for target-day exogenous inputs."""

    def __init__(
        self,
        condition_channels: int,
        target_channels: int = 4,
        seq_len: int = 24,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        pv_channel_idx: int = 3,
        use_pv_year_head: bool = True,
        pv_year_film_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pv_channel_idx = pv_channel_idx
        self.use_pv_year_head = use_pv_year_head
        self.pv_year_film_scale = pv_year_film_scale
        self.input_projection = nn.Linear(condition_channels, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, target_channels)
        if use_pv_year_head:
            self.pv_year_film = nn.Linear(1, 2 * hidden_dim)
            self.pv_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.pv_year_film.weight)
            nn.init.zeros_(self.pv_year_film.bias)
        else:
            self.pv_year_film = None
            self.pv_head = None

    def forward(self, conditions: torch.Tensor, pv_year: torch.Tensor) -> torch.Tensor:
        if conditions.ndim != 3 or conditions.shape[-1] != self.seq_len:
            raise ValueError("conditions must be [B,C,24]")
        tokens = self.input_projection(conditions.transpose(1, 2)) + self.position
        tokens = self.encoder(tokens)
        output = self.output_head(tokens)
        if self.pv_year_film is not None:
            year = pv_year.transpose(1, 2).to(tokens.dtype)
            gamma_raw, beta = self.pv_year_film(year).chunk(2, dim=-1)
            gamma = 1.0 + self.pv_year_film_scale * torch.tanh(gamma_raw)
            pv = self.pv_head(gamma * tokens + beta)
            output = output.clone()
            output[:, :, self.pv_channel_idx : self.pv_channel_idx + 1] = pv
        return output


def build_condition_model(config: dict[str, Any], condition_channels: int) -> nn.Module:
    model = config["condition_model"]
    return ConditionOnlyBackbone(
        condition_channels=condition_channels,
        target_channels=4,
        seq_len=int(config["data"]["seq_len"]),
        hidden_dim=int(model["hidden_dim"]),
        num_layers=int(model["num_layers"]),
        num_heads=int(model["num_heads"]),
        dropout=float(model["dropout"]),
        use_pv_year_head=bool(model.get("use_pv_year_head", True)),
        pv_year_film_scale=float(model.get("pv_year_film_scale", 0.2)),
    )


def build_diffusion(config: dict[str, Any]) -> D3UDiffusion:
    model = config["diffusion_model"]
    diffusion = config["diffusion"]
    arguments = {
        "diffusion_config_dir": str(
            resolve_path("model9_NS_transformer/configs/toy_8gauss.yml")
        ),
        "timesteps": int(diffusion["timesteps"]),
        "sampling_timesteps": int(diffusion.get("sampling_timesteps", 50)),
        "parameterization": diffusion.get("parameterization", "x_start"),
        "denoise_model": "PatchDN",
        "enc_in": 4,
        "pred_len": int(config["data"]["seq_len"]),
        "depth": int(model["depth"]),
        "d_model_d": int(model["hidden_dim"]),
        "n_heads_d": int(model["num_heads"]),
        "patch_size": int(model["patch_size"]),
        "stride": int(model["stride"]),
        "padding_patch": model.get("padding_patch", "end"),
        "d_model_c": int(config["condition_model"]["hidden_dim"]),
        # D3U conditions PatchDN on the deterministic forecast itself.
        "use_pretraining_condition": True,
    }
    return D3UDiffusion(to_namespace(arguments))


def checkpoint_payload(
    config: dict[str, Any],
    condition_model: nn.Module,
    diffusion: nn.Module,
    epoch: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "method": "D3U-condition-only-baseline1",
        "epoch": epoch,
        "condition_model": deepcopy(condition_model.state_dict()),
        "diffusion_model": deepcopy(diffusion.state_dict()),
        "config": deepcopy(config),
        "metrics": deepcopy(metrics),
    }

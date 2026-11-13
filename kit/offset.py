import math

import torch
import torch.nn as nn


_CONTINUOUS_FACTORS = (4, 2, 1)


def supported_offset_factors(target_mode):
    if str(target_mode) != "continuous":
        raise ValueError("Only continuous TreeOffset V2 is supported")
    return _CONTINUOUS_FACTORS


def offset_bounds(factor, stride, target_mode="continuous", device=None, dtype=None):
    del stride
    factor = int(factor)
    if factor not in supported_offset_factors(target_mode):
        raise ValueError(
            f"Unsupported offset factor {factor} for target_mode={target_mode}"
        )
    lower = -0.5
    upper = float(factor) - 0.5
    if device is None and dtype is None:
        return lower, upper
    return (
        torch.as_tensor(lower, device=device, dtype=dtype),
        torch.as_tensor(upper, device=device, dtype=dtype),
    )


class OffsetPredictor(nn.Module):
    """Center-offset branch for zero-bit geometry refinement."""

    def __init__(
        self,
        channels,
        hidden_channels=32,
        leaf_hidden_channels=8,
        max_depth=16,
        target_mode="continuous",
        boe_mode="hierarchical_group_leaf",
        boe_count_groups=3,
        boe_leaves_per_group=4,
    ):
        super().__init__()
        self.channels = int(channels)
        self.hidden_channels = int(hidden_channels)
        self.leaf_hidden_channels = int(leaf_hidden_channels)
        self.max_depth = max(int(max_depth), 1)
        self.target_mode = str(target_mode)
        self.boe_mode = str(boe_mode)
        self.boe_count_groups = int(boe_count_groups)
        self.boe_leaves_per_group = int(boe_leaves_per_group)
        if self.boe_mode != "hierarchical_group_leaf":
            raise ValueError("TreeOffset V2 requires hierarchical_group_leaf")
        if self.boe_count_groups != 3 or self.boe_leaves_per_group != 4:
            raise ValueError("TreeOffset V2 requires 3 groups with 4 leaves each")
        if self.leaf_hidden_channels <= 0:
            raise ValueError("leaf_hidden_channels must be positive")

        self.supported_factors = supported_offset_factors(self.target_mode)
        self._factor_to_range_id = {
            factor: idx for idx, factor in enumerate(self.supported_factors)
        }

        # Keep the module names used by existing _offset_simple_mlp checkpoints.
        self.range_embedding = nn.Embedding(len(self.supported_factors), 4)
        input_channels = self.channels + 10
        self.group_trunks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_channels, self.hidden_channels),
                    nn.ReLU(True),
                    nn.Linear(self.hidden_channels, self.hidden_channels),
                    nn.ReLU(True),
                )
                for _ in range(self.boe_count_groups)
            ]
        )
        self.group_heads = nn.ModuleList(
            [nn.Linear(self.hidden_channels, 3) for _ in range(self.boe_count_groups)]
        )
        self.leaf_delta_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.hidden_channels, self.leaf_hidden_channels),
                            nn.ReLU(True),
                            nn.Linear(self.leaf_hidden_channels, 3),
                        )
                        for _ in range(self.boe_leaves_per_group)
                    ]
                )
                for _ in range(self.boe_count_groups)
            ]
        )
        for head in self.group_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        for group_heads in self.leaf_delta_heads:
            for head in group_heads:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def get_config(self):
        return {
            "channels": self.channels,
            "hidden_channels": self.hidden_channels,
            "leaf_hidden_channels": self.leaf_hidden_channels,
            "max_depth": self.max_depth,
            "target_mode": self.target_mode,
            "boe_mode": self.boe_mode,
            "boe_count_groups": self.boe_count_groups,
            "boe_leaves_per_group": self.boe_leaves_per_group,
        }

    def supports_factor(self, factor):
        return int(factor) in self._factor_to_range_id

    def _geometry_features(self, coords, stride, spherical_params, dtype):
        stride = float(stride)
        center = (stride - 1.0) * 0.5
        rho = coords[:, 1].float() * stride + center
        theta = coords[:, 2].float() * stride + center
        phi = coords[:, 3].float() * stride + center

        rho_value = rho * float(spherical_params.scale_rho)
        theta_deg = (
            theta - float(spherical_params.theta_offset)
        ) * float(spherical_params.scale_theta)
        phi_deg = (
            phi - float(spherical_params.phi_offset)
        ) * float(spherical_params.scale_phi)
        theta_rad = torch.deg2rad(theta_deg)
        phi_rad = torch.deg2rad(phi_deg)
        log_stride = math.log2(max(stride, 1.0)) / 16.0
        return torch.stack(
            (
                rho_value / 100.0,
                torch.sin(theta_rad),
                torch.cos(theta_rad),
                torch.sin(phi_rad),
                torch.cos(phi_rad),
                torch.full_like(rho_value, log_stride),
            ),
            dim=1,
        ).to(dtype=dtype)

    def _condition_features(self, coords, features, stride, factor, spherical_params):
        geometry = self._geometry_features(
            coords,
            stride,
            spherical_params,
            features.dtype,
        )
        factor = int(factor)
        if factor not in self._factor_to_range_id:
            raise ValueError(
                f"OffsetPredictor target_mode={self.target_mode} "
                f"does not support factor={factor}"
            )
        range_id = self._factor_to_range_id[factor]
        range_ids = torch.full(
            (coords.shape[0],),
            range_id,
            dtype=torch.long,
            device=coords.device,
        )
        return torch.cat(
            (
                features,
                geometry,
                self.range_embedding(range_ids).to(features.dtype),
            ),
            dim=1,
        )

    def _expanded_expert_ids(self, expert_idx, count, device):
        total_experts = self.boe_count_groups * self.boe_leaves_per_group
        if expert_idx is None:
            return torch.zeros(count, dtype=torch.long, device=device)
        if torch.is_tensor(expert_idx):
            ids = expert_idx.to(device=device, dtype=torch.long).reshape(-1)
            if ids.numel() == 1:
                ids = ids.expand(count)
            elif ids.numel() != count:
                raise ValueError(
                    f"expert_idx count {ids.numel()} does not match {count}"
                )
            return ids.clamp(0, total_experts - 1)
        value = max(0, min(int(expert_idx), total_experts - 1))
        return torch.full((count,), value, dtype=torch.long, device=device)

    def _hierarchical_boe_raw(self, model_input, expert_idx):
        expert_ids = self._expanded_expert_ids(
            expert_idx,
            model_input.shape[0],
            model_input.device,
        )
        group_ids = torch.div(
            expert_ids,
            self.boe_leaves_per_group,
            rounding_mode="floor",
        ).clamp(0, self.boe_count_groups - 1)
        leaf_ids = torch.remainder(
            expert_ids,
            self.boe_leaves_per_group,
        ).clamp(0, self.boe_leaves_per_group - 1)

        raw = model_input.new_zeros((model_input.shape[0], 3))
        for group_id in range(self.boe_count_groups):
            group_mask = group_ids == group_id
            if not bool(group_mask.any().item()):
                continue
            hidden = self.group_trunks[group_id](model_input[group_mask])
            group_raw = self.group_heads[group_id](hidden)
            group_leaf_ids = leaf_ids[group_mask]
            delta = torch.zeros_like(group_raw)
            for leaf_id in range(self.boe_leaves_per_group):
                leaf_mask = group_leaf_ids == leaf_id
                if bool(leaf_mask.any().item()):
                    delta[leaf_mask] = self.leaf_delta_heads[group_id][leaf_id](
                        hidden[leaf_mask]
                    )
            raw[group_mask] = group_raw + delta
        return raw

    def forward(
        self,
        coords,
        features,
        stride,
        factor,
        spherical_params,
        depth=None,
        expert_idx=None,
        confidence_threshold=0.0,
    ):
        if features.ndim != 2 or features.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} feature channels, got "
                f"{tuple(features.shape)}"
            )
        model_input = self._condition_features(
            coords,
            features,
            stride,
            factor,
            spherical_params,
        )
        raw = self._hierarchical_boe_raw(model_input, expert_idx)
        residual = torch.tanh(raw)
        lower, upper = offset_bounds(
            factor,
            stride,
            self.target_mode,
            device=residual.device,
            dtype=residual.dtype,
        )
        center = (lower + upper) * 0.5
        half = (upper - lower) * 0.5
        return center + half * residual

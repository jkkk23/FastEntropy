import math

import torch
import torch.nn as nn

from torchsparse import nn as spnn

class ResNet(torch.nn.Module): 
    """Residual Network
    """  
    def __init__(self, channels, k=3):
        super().__init__()
        self.conv0 = spnn.Conv3d(channels, channels, k)
        self.conv1 = spnn.Conv3d(channels, channels, k)
        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.conv0(x))
        out = self.conv1(out)
        out = self.relu(out+x)
        return out


class FCG(torch.nn.Module):
    def __init__(self):
        super(FCG, self).__init__()

        self.register_buffer("expand_coords_base", torch.tensor([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]), persistent=False)

        self.register_buffer("pos", torch.arange(0, 8).view(1, 8), persistent=False)

    def forward(self, x_C, x_O, x_F=None, return_info=False, return_parent_idx=False):
        occ = x_O.view(-1).long()  # (N,)
        device = x_C.device

        # active_mask has shape (N, 8) and marks each parent's active children.
        pos = self.pos.to(device=device).view(1, 8).long()
        active_mask = ((occ[:, None] >> pos) & 1).bool()

        # parent_idx maps each active child to its parent.
        # child_id is the active child's octant index in [0, 7].
        parent_idx, child_id = torch.nonzero(active_mask, as_tuple=True)

        base = self.expand_coords_base.to(device=device, dtype=x_C.dtype)

        x_up_C = x_C[parent_idx].clone()
        x_up_C[:, 1:] = x_up_C[:, 1:] * 2 + base[child_id]
        x_up_C = x_up_C.int()

        x_up_parent_O = occ[parent_idx]
        x_up_child_id = child_id.long()

        if x_F is None:
            if return_info:
                if return_parent_idx:
                    return x_up_C, x_up_parent_O, x_up_child_id, parent_idx
                return x_up_C, x_up_parent_O, x_up_child_id
            return x_up_C

        x_up_F = x_F[parent_idx]

        if return_info:
            if return_parent_idx:
                return x_up_C, x_up_F, x_up_parent_O, x_up_child_id, parent_idx
            return x_up_C, x_up_F, x_up_parent_O, x_up_child_id

        return x_up_C, x_up_F
        

class TargetEmbedding(torch.nn.Module): 
    """Target Embedding
    """  
    def __init__(
        self,
        channels,
        sph_alpha=1.0,
        scale_rho=200 / (2**16 - 1),
        scale_theta=0.02747 / 2,
        scale_phi=0.01373 / 2,
        theta_offset=None,
        phi_offset=None,
        use_log_radius=False,
    ):
        super(TargetEmbedding, self).__init__()
        self.target_res_embedding = nn.Embedding(8, channels)
        self.posi=nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU(True),
        )
        self.spherical_posi = nn.Sequential(
            nn.Linear(6, channels),
            nn.ReLU(True),
        )
        self.spherical_posi_gate = nn.Parameter(torch.ones(1) * 1e-3)
        self.set_spherical_params(
            sph_alpha=sph_alpha,
            scale_rho=scale_rho,
            scale_theta=scale_theta,
            scale_phi=scale_phi,
            theta_offset=theta_offset,
            phi_offset=phi_offset,
            use_log_radius=use_log_radius,
        )

    def set_spherical_params(
        self,
        sph_alpha=None,
        scale_rho=None,
        scale_theta=None,
        scale_phi=None,
        theta_offset=None,
        phi_offset=None,
        use_log_radius=None,
    ):
        if sph_alpha is not None:
            self.sph_alpha = float(sph_alpha)
        if scale_rho is not None:
            self.scale_rho = float(scale_rho)
        if scale_theta is not None:
            self.scale_theta = float(scale_theta)
        if scale_phi is not None:
            self.scale_phi = float(scale_phi)
        if theta_offset is not None:
            self.theta_offset = int(theta_offset)
        elif scale_theta is not None or not hasattr(self, "theta_offset"):
            self.theta_offset = int(math.ceil(180.0 / self.scale_theta)) + 1
        if phi_offset is not None:
            self.phi_offset = int(phi_offset)
        elif scale_phi is not None or not hasattr(self, "phi_offset"):
            self.phi_offset = int(math.ceil(90.0 / self.scale_phi)) + 1
        if use_log_radius is not None:
            self.use_log_radius = bool(use_log_radius)

    def _spherical_coord_features(self, x_up_C, target_stride, dtype):
        stride = torch.as_tensor(float(target_stride), dtype=torch.float32, device=x_up_C.device)
        center_offset = (stride - 1.0) * 0.5

        rho_center = x_up_C[:, 1].float() * stride + center_offset
        theta_center = x_up_C[:, 2].float() * stride + center_offset
        phi_center = x_up_C[:, 3].float() * stride + center_offset

        rho_value = rho_center * self.scale_rho
        if self.use_log_radius:
            rho_feature = rho_value / 8.0
        else:
            rho_feature = torch.log1p(rho_value.clamp_min(0.0)) / 8.0
        theta_rad = (theta_center - self.theta_offset) * self.scale_theta * (math.pi / 180.0)
        phi_rad = (phi_center - self.phi_offset) * self.scale_phi * (math.pi / 180.0)
        stride_log = torch.log2(stride).expand_as(theta_rad) / 16.0

        return torch.stack(
            [
                rho_feature,
                torch.sin(theta_rad),
                torch.cos(theta_rad),
                torch.sin(phi_rad),
                torch.cos(phi_rad),
                stride_log,
            ],
            dim=-1,
        ).to(dtype=dtype)

    def forward(self, x_up_F, x_up_C, target_stride=1):
        '''Embed x.F from x.C to x_up_C
        Input: x_up_F Feats (N_{d+1}, dim)
        Input: x_up_C Coords (N_{d+1}, 4)
        Return x_up SparseTensor (N_{d+1}, dim)
        ''' 
        
        r = x_up_C[:, 1].float()
        theta = x_up_C[:, 2].long()
        phi = x_up_C[:, 3].long()

        r_norm = (r - r.min()) / (r.max() - r.min()).clamp_min(1.0)
        theta_norm = (theta - theta.min()).float() / (theta.max() - theta.min()).clamp_min(1).float()
        phi_norm = (phi - phi.min()).float() / (phi.max() - phi.min()).clamp_min(1).float()

        # Reduce coordinate offsets modulo two to obtain binary octant bits.
        coords_delta = x_up_C[:, 1:] % 2

        # Convert the three offset bits into idx = x + 2*y + 4*z.
        coords_idx = coords_delta[:, 0] + coords_delta[:, 1]*2 + coords_delta[:, 2]*4
        # Use the integer octant index to look up the target-resolution embedding.
        norm_coords = torch.stack([r_norm, theta_norm, phi_norm], dim=-1).to(dtype=x_up_F.dtype)
        spherical_coords = self._spherical_coord_features(
            x_up_C,
            target_stride=target_stride,
            dtype=x_up_F.dtype,
        )
        x_up_F = (
            x_up_F
            + self.target_res_embedding(coords_idx.int())
            + self.posi(norm_coords)
            + self.spherical_posi_gate * self.spherical_posi(spherical_coords)
        ) # (B*Nt, C)
        return x_up_F

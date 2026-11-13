import torch
import torch.nn as nn
from torchsparse import nn as spnn
from torchsparse import SparseTensor
import kit.op as op
from kit.nn import ResNet, FCG, TargetEmbedding
from config import OPTIMAL_ARCHITECTURE
BIT_PERMUTATIONS = {'phi_first': (0, 1, 4, 5, 2, 3, 6, 7)}

class CoarseGridPrior(nn.Module):
    """A small angular lookup table used only by coarse octree levels."""

    def __init__(self, theta_bins, phi_bins, channels):
        super().__init__()
        self.theta_bins = int(theta_bins)
        self.phi_bins = int(phi_bins)
        self.embedding = nn.Embedding(self.theta_bins * self.phi_bins, channels)
        self.gate = nn.Parameter(torch.ones(1) * 0.001)

    def forward(self, coords, feats, target_stride, theta_offset, phi_offset):
        stride = float(target_stride)
        center = (stride - 1.0) * 0.5
        theta = coords[:, 2].float() * stride + center
        phi = coords[:, 3].float() * stride + center
        theta = (theta / max(2.0 * theta_offset, 1.0)).clamp(0.0, 1.0)
        phi = (phi / max(2.0 * phi_offset, 1.0)).clamp(0.0, 1.0)
        theta_idx = (theta * (self.theta_bins - 1)).long()
        phi_idx = (phi * (self.phi_bins - 1)).long()
        grid_idx = theta_idx * self.phi_bins + phi_idx
        grid_features = self.embedding(grid_idx).to(dtype=feats.dtype)
        return feats + self.gate * grid_features

class CodingCore(nn.Module):
    """One specialized entropy expert used at a complete coding level."""

    def __init__(self, channels, kernel_size, prior_res_blocks, target_res_blocks, coarse_k1_depths):
        super().__init__()
        self.prior_embedding = nn.Embedding(256, channels)
        self.prior_resnet = nn.Sequential(spnn.Conv3d(channels, channels, kernel_size), spnn.ReLU(True), *[ResNet(channels, k=kernel_size) for _ in range(prior_res_blocks)])
        if coarse_k1_depths > 0:
            self.coarse_prior_resnet = nn.Sequential(spnn.Conv3d(channels, channels, 1), spnn.ReLU(True), *[ResNet(channels, k=1) for _ in range(prior_res_blocks)])
        self.target_resnet = nn.Sequential(spnn.Conv3d(channels, channels, kernel_size), spnn.ReLU(True), *[ResNet(channels, k=kernel_size) for _ in range(target_res_blocks)])
        if coarse_k1_depths > 0:
            self.coarse_target_resnet = nn.Sequential(spnn.Conv3d(channels, channels, 1), spnn.ReLU(True), *[ResNet(channels, k=1) for _ in range(target_res_blocks)])
        self.pred_head_s0 = nn.Sequential(nn.Linear(channels, channels * 2), nn.ReLU(True), nn.Linear(channels * 2, 16), nn.Softmax(dim=-1))
        self.pred_head_s1_emb = nn.Embedding(16, channels)
        self.pred_head_s1_hat_emb = nn.Embedding(16, channels)
        self.pred_head_s1_xor_emb = nn.Embedding(16, channels)
        self.s1_s0_scale = nn.Parameter(torch.ones(1))
        self.s1_corr_scale = nn.Parameter(torch.ones(1) * 0.001)
        self.pred_head_s1 = nn.Sequential(nn.Linear(channels, channels * 2), nn.ReLU(True), nn.Linear(channels * 2, 16), nn.Softmax(dim=-1))
        self.pred_head_joint = nn.Sequential(nn.Linear(channels, channels), nn.ReLU(True), nn.Linear(channels, 256), nn.Softmax(dim=-1))
        self.blend_weights = nn.Parameter(torch.ones(2, channels))
        self.cross_s0_emb = nn.Embedding(16, channels)
        self.cross_s1_emb = nn.Embedding(16, channels)
        self.propagate_resnet = ResNet(channels, k=kernel_size)

class GroupBoundaryFeatureAlignment(nn.Module):
    """Group-boundary alignment for cross-expert context propagation."""

    def __init__(self, expert_count, channels, count_groups, leaves_per_group):
        super().__init__()
        self.expert_count = int(expert_count)
        self.channels = int(channels)
        self.count_groups = int(count_groups)
        self.leaves_per_group = int(leaves_per_group)
        self.depth_bins = 32
        self.alignment_scale = 0.1
        if self.expert_count <= 0:
            raise ValueError('expert_count must be positive')
        if self.count_groups <= 0 or self.leaves_per_group <= 0:
            raise ValueError('count_groups and leaves_per_group must be positive')
        if self.expert_count != self.count_groups * self.leaves_per_group:
            raise ValueError('GroupBoundaryFeatureAlignment expects expert_count=count_groups * leaves_per_group')
        pair_count = self.count_groups * self.count_groups
        self.gamma_pair = nn.Embedding(pair_count, self.channels)
        self.beta_pair = nn.Embedding(pair_count, self.channels)
        self.gamma_depth = nn.Embedding(self.depth_bins, self.channels)
        self.beta_depth = nn.Embedding(self.depth_bins, self.channels)
        self.blend_transition = nn.Embedding(3, 1)
        self.blend_pair = nn.Embedding(pair_count, 1)
        self.blend_depth = nn.Embedding(self.depth_bins, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for emb in (self.gamma_pair, self.beta_pair, self.gamma_depth, self.beta_depth, self.blend_transition, self.blend_pair, self.blend_depth):
            nn.init.zeros_(emb.weight)

    def _expert_id(self, value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = int(value.detach().reshape(-1)[0].item())
        else:
            value = int(value)
        if value < 0 or value >= self.expert_count:
            return None
        return value

    def forward(self, features, source_core_idx, target_core_idx, depth):
        source_id = self._expert_id(source_core_idx)
        target_id = self._expert_id(target_core_idx)
        if source_id is None or target_id is None:
            return (features, None)
        source_group = source_id // self.leaves_per_group
        target_group = target_id // self.leaves_per_group
        same_leaf = source_id == target_id
        same_group = source_group == target_group
        transition_id = 0 if same_leaf else 1 if same_group else 2
        depth_id = min(max(int(depth), 0), self.depth_bins - 1)
        device = features.device
        pair_id = torch.tensor(source_group * self.count_groups + target_group, dtype=torch.long, device=device)
        depth_tensor = torch.tensor(depth_id, dtype=torch.long, device=device)
        transition_tensor = torch.tensor(transition_id, dtype=torch.long, device=device)
        blend_bias = (self.blend_transition(transition_tensor) + self.blend_pair(pair_id) + self.blend_depth(depth_tensor)).view(())
        if same_group:
            return (features, blend_bias)
        gamma = (self.gamma_pair(pair_id) + self.gamma_depth(depth_tensor)).to(features.dtype)
        beta = (self.beta_pair(pair_id) + self.beta_depth(depth_tensor)).to(features.dtype)
        scale = self.alignment_scale
        gamma = scale * torch.tanh(gamma)
        beta = scale * torch.tanh(beta)
        aligned = features * (1.0 + gamma.view(1, -1)) + beta.view(1, -1)
        return (aligned, blend_bias)

class Network(nn.Module):

    def __init__(
        self,
        channels,
        kernel_size,
        sph_alpha=1.0,
        scale_rho=200 / (2 ** 16 - 1),
        scale_theta=0.02747 / 2,
        scale_phi=0.01373 / 2,
        theta_offset=None,
        phi_offset=None,
        use_log_radius=False,
        **architecture,
    ):
        super(Network, self).__init__()
        unknown = set(architecture) - set(OPTIMAL_ARCHITECTURE)
        if unknown:
            raise ValueError(f'Unknown architecture fields: {sorted(unknown)}')
        submitted = dict(OPTIMAL_ARCHITECTURE)
        submitted.update(architecture)
        mismatch = {name: (submitted[name], expected) for (name, expected) in OPTIMAL_ARCHITECTURE.items() if submitted[name] != expected}
        if mismatch:
            raise ValueError(f'Only the submitted FastEntropy architecture is supported: {mismatch}')
        if use_log_radius:
            raise ValueError('Only linear spherical radius is supported')
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        for name, value in submitted.items():
            setattr(self, name, value)
        expected_k = self.boe_count_groups * self.boe_leaves_per_group
        if self.boe_k != expected_k:
            raise ValueError(f'Expected {expected_k} hierarchical BoE experts')
        self._active_core_idx = 0
        permutation = torch.tensor(BIT_PERMUTATIONS[self.bit_permutation], dtype=torch.long)
        inverse_permutation = torch.empty_like(permutation)
        inverse_permutation[permutation] = torch.arange(8, dtype=torch.long)
        self.register_buffer('occupancy_permutation', permutation, persistent=False)
        self.register_buffer('occupancy_inverse_permutation', inverse_permutation, persistent=False)
        self.register_buffer('occupancy_popcount', torch.tensor([value.bit_count() for value in range(256)], dtype=torch.long), persistent=False)
        if self.uniform_prior_depths > 0:
            self.coarse_prior_logits = nn.Parameter(torch.zeros(self.uniform_prior_depths, 256))
        self.target_embedding = TargetEmbedding(channels, sph_alpha=sph_alpha, scale_rho=scale_rho, scale_theta=scale_theta, scale_phi=scale_phi, theta_offset=theta_offset, phi_offset=phi_offset, use_log_radius=use_log_radius)
        self.parent_occ_emb = nn.Embedding(256, channels)
        self.parent_child_emb = nn.Embedding(256 * 8, channels)
        self.lineage_child_emb = nn.Embedding(8, channels)
        self.lineage_s0_emb = nn.Embedding(16, channels)
        self.lineage_s1_emb = nn.Embedding(16, channels)
        self.lineage_depth_emb = nn.Embedding(64, channels)
        self.lineage_child_transition = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU(True), nn.Linear(channels, channels))
        self.lineage_parent_gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.Sigmoid())
        self.lineage_child_gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.Sigmoid())
        self.lineage_update = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU(True), nn.Linear(channels, channels), nn.Tanh())
        self.lineage_update_gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.Sigmoid())
        self.lineage_scale = nn.Parameter(torch.ones(1) * 0.001)
        self.register_buffer('cross_depth_mismatch_count', torch.zeros((), dtype=torch.long), persistent=False)
        self.coarse_grid_prior = CoarseGridPrior(self.coarse_grid_theta_bins, self.coarse_grid_phi_bins, channels)
        self.fcg = FCG()
        self.boe_cores = nn.ModuleList([
            CodingCore(
                channels,
                kernel_size,
                self.prior_res_blocks,
                self.target_res_blocks,
                self.coarse_k1_depths,
            )
            for _ in range(self.boe_k)
        ])
        self.group_boundary_feature_alignment = GroupBoundaryFeatureAlignment(
            self.boe_k,
            channels,
            self.boe_count_groups,
            self.boe_leaves_per_group,
        )
        self.register_buffer('boe_count_centers', torch.zeros(max(self.boe_count_groups, 1), 10), persistent=True)
        self.register_buffer('boe_symbol_centers', torch.zeros(max(self.boe_count_groups, 1), max(self.boe_leaves_per_group, 1), 32), persistent=True)

    def set_spherical_params(self, **kwargs):
        self.target_embedding.set_spherical_params(**kwargs)

    def get_model_config(self):
        return {
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            **OPTIMAL_ARCHITECTURE,
        }

    def uniform_prior_marker(self):
        return self.boe_k

    def core(self, idx=None):
        if idx is None:
            idx = self._active_core_idx
        return self.boe_cores[int(idx)]

    def set_active_core(self, idx):
        idx = int(idx)
        if idx == self.uniform_prior_marker():
            self._active_core_idx = 0
            return
        if idx < 0 or idx >= self.boe_k:
            raise ValueError(f'Invalid entropy expert index: {idx}')
        self._active_core_idx = idx

    def load_boe_centers(self, centers):
        if hasattr(centers, 'files'):
            if 'count_centers' not in centers.files or 'symbol_centers' not in centers.files:
                raise ValueError('Hierarchical BoE center file must contain count_centers and symbol_centers')
            count_centers = torch.as_tensor(centers['count_centers'], dtype=self.boe_count_centers.dtype, device=self.boe_count_centers.device)
            symbol_centers = torch.as_tensor(centers['symbol_centers'], dtype=self.boe_symbol_centers.dtype, device=self.boe_symbol_centers.device)
            if count_centers.shape != self.boe_count_centers.shape:
                raise ValueError(f'Expected hierarchical count centers with shape {tuple(self.boe_count_centers.shape)}, got {tuple(count_centers.shape)}')
            if symbol_centers.shape != self.boe_symbol_centers.shape:
                raise ValueError(f'Expected hierarchical symbol centers with shape {tuple(self.boe_symbol_centers.shape)}, got {tuple(symbol_centers.shape)}')
            self.boe_count_centers.copy_(count_centers)
            self.boe_symbol_centers.copy_(symbol_centers)
            return
        raise ValueError('Only hierarchical BoE center files are supported')

    def _symbol_descriptor(self, s0, s1, sqrt=False):
        hist_s0 = torch.bincount(s0.long().view(-1), minlength=16).float()
        hist_s1 = torch.bincount(s1.long().view(-1), minlength=16).float()
        hist = torch.cat((hist_s0, hist_s1), dim=0)
        hist = hist / hist.sum().clamp_min(1.0)
        if sqrt:
            hist = torch.sqrt(hist + 1e-08)
        return hist

    def _count_descriptor(self, occupancy):
        occupancy = occupancy.long().view(-1)
        counts = self.occupancy_popcount.to(occupancy.device)[occupancy]
        hist = torch.bincount(counts - 1, minlength=8).float()
        hist = hist / hist.sum().clamp_min(1.0)
        values = torch.arange(1, 9, dtype=torch.float32, device=hist.device)
        mean = (hist * values).sum()
        var = (hist * (values - mean).square()).sum()
        return torch.cat((torch.sqrt(hist + 1e-08), (mean / 8.0).view(1), (torch.sqrt(var) / 8.0).view(1)))

    def select_encoder_index(self, depth, s0=None, s1=None, occupancy=None):
        if depth < self.uniform_prior_depths or s0 is None or s1 is None:
            return self.uniform_prior_marker()
        if occupancy is None:
            return self.uniform_prior_marker()
        count_desc = self._count_descriptor(occupancy)
        count_dists = torch.norm(self.boe_count_centers.to(count_desc.device) - count_desc.unsqueeze(0), dim=1)
        group_idx = int(torch.argmin(count_dists).item())
        symbol_desc = self._symbol_descriptor(s0, s1, sqrt=True)
        symbol_centers = self.boe_symbol_centers[group_idx].to(symbol_desc.device)
        symbol_dists = torch.norm(symbol_centers - symbol_desc.unsqueeze(0), dim=1)
        leaf_idx = int(torch.argmin(symbol_dists).item())
        return group_idx * self.boe_leaves_per_group + leaf_idx

    def build_occupancy_pyramid(self, x, min_points=64, skipped_depths=0):
        return self._build_morton_pyramid(x, min_points, skipped_depths)

    def _build_morton_pyramid(self, x, min_points, skipped_depths):
        coords = op.morton_unique_sort_C(x.coords)
        data_ls = []
        discarded_depths = 0
        while True:
            (coords, occupancy) = op.morton_downsample(coords)
            if discarded_depths < skipped_depths:
                discarded_depths += 1
            else:
                data_ls.append((coords.clone(), occupancy.clone()))
            if coords.shape[0] < min_points:
                break
        return data_ls[::-1]

    def order_C(self, coords):
        return coords

    def order_CF(self, coords, feats):
        return (coords, feats)

    def order_CFs(self, coords, *features):
        if not features:
            return (self.order_C(coords),)
        return (coords, *features)

    def uses_uniform_prior(self, depth):
        return depth < self.uniform_prior_depths

    def uses_coarse_network(self, depth):
        return depth < self.coarse_k1_depths

    def uses_joint_head(self, num_points):
        return (
            self.joint_distribution_max_points > 0
            and int(num_points) <= self.joint_distribution_max_points
        )

    def run_prior_resnet(self, x, depth):
        core = self.core()
        if self.uses_coarse_network(depth):
            return core.coarse_prior_resnet(x)
        return core.prior_resnet(x)

    def fuse_cross_depth_parent(self, current_feats, propagated_feats, depth, encoded_depths, source_core_idx=None, target_core_idx=None):
        if propagated_feats is None:
            return current_feats
        if propagated_feats.shape != current_feats.shape:
            self.cross_depth_mismatch_count.add_(1)
            return current_feats
        if target_core_idx is None:
            target_core_idx = self._active_core_idx
        (propagated_feats, blend_bias) = self.group_boundary_feature_alignment(
            propagated_feats,
            source_core_idx,
            target_core_idx,
            depth,
        )
        blend_logits = self.core().blend_weights.to(current_feats.dtype)
        if blend_bias is not None:
            blend_bias = blend_bias.to(current_feats.dtype)
            blend_logits = torch.stack((blend_logits[0], blend_logits[1] + blend_bias), dim=0)
        weights = torch.softmax(blend_logits, dim=0)
        return weights[0].view(1, -1) * current_feats + weights[1].view(1, -1) * propagated_feats.to(current_feats.dtype)

    def fuse_lineage_parent(self, current_feats, lineage_feats):
        if lineage_feats is None:
            return current_feats
        if lineage_feats.shape != current_feats.shape:
            return current_feats
        lineage_feats = lineage_feats.to(current_feats.dtype)
        gate = self.lineage_parent_gate(torch.cat((current_feats, lineage_feats), dim=1))
        return current_feats + self.lineage_scale.to(current_feats.dtype) * gate * lineage_feats

    def expand_lineage_to_children(self, lineage_feats, parent_idx, child_id, dtype, device):
        child_emb = self.lineage_child_emb(child_id.long()).to(device=device, dtype=dtype)
        if lineage_feats is None:
            parent_state = torch.zeros_like(child_emb)
        else:
            parent_state = lineage_feats[parent_idx.long()].to(device=device, dtype=dtype)
        transition = self.lineage_child_transition(torch.cat((parent_state, child_emb), dim=1))
        return parent_state + child_emb + self.lineage_scale.to(dtype) * transition

    def fuse_lineage_child(self, child_feats, child_lineage):
        if child_lineage is None:
            return child_feats
        child_lineage = child_lineage.to(child_feats.dtype)
        gate = self.lineage_child_gate(torch.cat((child_feats, child_lineage), dim=1))
        return child_feats + self.lineage_scale.to(child_feats.dtype) * gate * child_lineage

    def update_lineage_memory(self, features, s0, s1, child_lineage, depth):
        if child_lineage is None:
            child_lineage = features.new_zeros(features.shape)
        child_lineage = child_lineage.to(features.dtype)
        depth_id = torch.full((features.shape[0],), min(max(int(depth), 0), self.lineage_depth_emb.num_embeddings - 1), dtype=torch.long, device=features.device)
        decoded_context = features + self.lineage_s0_emb(s0.long()).to(features.dtype) + self.lineage_s1_emb(s1.long()).to(features.dtype) + self.lineage_depth_emb(depth_id).to(features.dtype)
        update_input = torch.cat((child_lineage, decoded_context), dim=1)
        candidate = self.lineage_update(update_input)
        gate = self.lineage_update_gate(update_input).to(features.dtype)
        return (1.0 - gate) * child_lineage + gate * candidate

    def make_cross_depth_features(self, coords, features, s0, s1):
        if features is None:
            return None
        core = self.core()
        feats = features + core.cross_s0_emb(s0.long()).to(features.dtype) + core.cross_s1_emb(s1.long()).to(features.dtype)
        propagated = core.propagate_resnet(SparseTensor(coords=coords, feats=feats)).feats
        return feats + propagated

    def apply_coarse_grid_prior(self, coords, feats, depth, target_stride):
        if depth >= self.coarse_grid_depths:
            return feats
        return self.coarse_grid_prior(coords, feats, target_stride, self.target_embedding.theta_offset, self.target_embedding.phi_offset)

    def fuse_child_features(self, x_up_F, x_up_parent_O, x_up_child_id, child_lineage=None):
        parent_child_key = x_up_parent_O * 8 + x_up_child_id
        fused = x_up_F + self.parent_occ_emb(x_up_parent_O) + self.parent_child_emb(parent_child_key)
        return self.fuse_lineage_child(fused, child_lineage)

    def _permute_occupancy(self, occupancy):
        occupancy = occupancy.long()
        permuted = torch.zeros_like(occupancy)
        for (target_bit, source_bit) in enumerate(self.occupancy_permutation.tolist()):
            permuted = permuted | (occupancy >> source_bit & 1) << target_bit
        return permuted

    def _unpermute_occupancy(self, occupancy):
        occupancy = occupancy.long()
        restored = torch.zeros_like(occupancy)
        for (source_bit, target_bit) in enumerate(self.occupancy_inverse_permutation.tolist()):
            restored = restored | (occupancy >> target_bit & 1) << source_bit
        return restored

    def split_occupancy_two_stage(self, occupancy):
        permuted = self._permute_occupancy(occupancy)
        return (torch.remainder(permuted, 16), torch.div(permuted, 16, rounding_mode='floor'))

    def combine_occupancy_two_stage(self, s0, s1):
        permuted = s0.long() | s1.long() << 4
        return self._unpermute_occupancy(permuted)

    def predict_s0(self, x_up_feats):
        return (x_up_feats, self.core().pred_head_s0(x_up_feats))

    def predict_s1_from_s0(self, x_up_feats, prob_s0, decoded_s0):
        with torch.no_grad():
            decoded_s0 = decoded_s0.long()
            prob_s0_det = prob_s0.detach()
            fake_s0 = torch.argmax(prob_s0_det, dim=1).long()
            xor_s0 = torch.bitwise_xor(decoded_s0, fake_s0)
            p_true_s0 = prob_s0_det.gather(1, decoded_s0.view(-1, 1)).clamp(1e-06, 1.0)
            surprise_s0 = (1.0 - p_true_s0).to(dtype=x_up_feats.dtype)
            prob_s0_for_emb = prob_s0_det.to(dtype=x_up_feats.dtype)
        core = self.core()
        s0_emb = core.pred_head_s1_emb(decoded_s0)
        hat_weight = core.pred_head_s1_hat_emb.weight.to(dtype=x_up_feats.dtype)
        soft_hat_emb = torch.matmul(prob_s0_for_emb, hat_weight)
        xor_emb = core.pred_head_s1_xor_emb(xor_s0)
        corr_emb = surprise_s0 * (s0_emb - soft_hat_emb + xor_emb)
        s1_feats = x_up_feats + core.s1_s0_scale * s0_emb + core.s1_corr_scale * corr_emb
        return core.pred_head_s1(s1_feats)

    def predict_joint_two_stage(self, x_up_feats, decoded_s0, return_count_prob=False):
        """Predict the 8-bit symbol once, then expose two small 16-way CDFs."""
        (joint, count_prob) = self.predict_joint_distribution(x_up_feats, return_count_prob=True)
        prob_s0 = joint.sum(dim=1)
        prob_s1 = self.condition_joint_on_s0(joint, prob_s0, decoded_s0)
        if return_count_prob:
            return (prob_s0, prob_s1, count_prob)
        return (prob_s0, prob_s1)

    def predict_joint_distribution(self, x_up_feats, return_count_prob=False):
        joint = self.core().pred_head_joint(x_up_feats).view(-1, 16, 16)
        if return_count_prob:
            return (joint, None)
        return joint

    @staticmethod
    def condition_joint_on_s0(joint, prob_s0, decoded_s0):
        gather_idx = decoded_s0.long().view(-1, 1, 1).expand(-1, 16, 1)
        prob_s1 = joint.gather(2, gather_idx).squeeze(2)
        normalizer = prob_s0.gather(1, decoded_s0.long().view(-1, 1))
        return prob_s1 / normalizer.clamp_min(1e-10)

    def coarse_prior_distribution(self, depth, num_points, dtype):
        probability = torch.softmax(self.coarse_prior_logits[depth].float(), dim=0).to(dtype=dtype)
        return probability.unsqueeze(0).expand(num_points, -1)

    def coarse_prior_two_stage(self, depth, num_points, decoded_s0, dtype):
        joint = self.coarse_prior_distribution(depth, num_points, dtype).view(-1, 16, 16)
        prob_s0 = joint.sum(dim=1)
        prob_s1 = self.condition_joint_on_s0(joint, prob_s0, decoded_s0)
        return (prob_s0, prob_s1)

    def run_target_resnet(self, x, depth, encoded_depths, skipped_depths=0):
        target_resnet = self.core().coarse_target_resnet if self.uses_coarse_network(depth) else self.core().target_resnet
        del encoded_depths, skipped_depths
        x = target_resnet[0](x)
        x = target_resnet[1](x)
        return target_resnet[2](x)

    @staticmethod
    def separate_coding_depth(coords, feats, depth, encoded_depths, near_radius, far_radius):
        if near_radius is None or far_radius is None:
            return (coords, feats)
        if depth == encoded_depths - 1:
            (coords, _, feats, _) = op.separation(coords, feats, R=far_radius)
        elif depth == encoded_depths - 2:
            (coords, _, feats, _) = op.separation(coords, feats, R=near_radius / 2)
        return (coords, feats)

    def _rate_pass_from_pyramid(
        self,
        data_ls,
        original_points,
        skipped_depths=0,
        near_radius=None,
        far_radius=None,
    ):
        encoded_depths = max(len(data_ls) - 1, 0)
        total_bits = data_ls[0][1].new_zeros(())
        s0_bits = total_bits.clone()
        s1_bits = total_bits.clone()
        propagated_feats = None
        propagated_core_idx = None
        lineage_feats = None
        (x_C, x_O) = data_ls[0]
        for depth in range(encoded_depths):
            (gt_x_up_C, gt_x_up_O) = data_ls[depth + 1]
            (gt_x_up_C, gt_x_up_O) = self.separate_coding_depth(gt_x_up_C, gt_x_up_O, depth=depth, encoded_depths=encoded_depths, near_radius=near_radius, far_radius=far_radius)
            (gt_x_up_C, gt_x_up_O) = self.order_CF(gt_x_up_C, gt_x_up_O)
            (gt_s0, gt_s1) = self.split_occupancy_two_stage(gt_x_up_O)
            core_idx = self.select_encoder_index(depth, gt_s0[:, 0], gt_s1[:, 0], occupancy=gt_x_up_O[:, 0])
            self.set_active_core(core_idx)
            if self.uses_uniform_prior(depth):
                x_up_C = self.fcg(x_C, x_O)
                (x_up_C, _) = self.separate_coding_depth(x_up_C, x_up_C, depth=depth, encoded_depths=encoded_depths, near_radius=near_radius, far_radius=far_radius)
                (prob_s0, prob_s1) = self.coarse_prior_two_stage(depth, x_up_C.shape[0], gt_s0[:, 0], torch.float32)
                propagated_feats = None
                propagated_core_idx = None
                lineage_feats = None
            else:
                x_F = self.core().prior_embedding(x_O.int()).view(-1, self.channels)
                x_F = self.fuse_cross_depth_parent(x_F, propagated_feats, depth, encoded_depths, source_core_idx=propagated_core_idx, target_core_idx=core_idx)
                x_F = self.fuse_lineage_parent(x_F, lineage_feats)
                x = SparseTensor(coords=x_C, feats=x_F)
                x = self.run_prior_resnet(x, depth)
                (x_up_C, x_up_F, x_up_parent_O, x_up_child_id, x_up_parent_idx) = self.fcg(x_C, x_O, x.feats, return_info=True, return_parent_idx=True)
                x_up_lineage = self.expand_lineage_to_children(lineage_feats, x_up_parent_idx, x_up_child_id, x_up_F.dtype, x_up_F.device)
                x_up_F = self.fuse_child_features(x_up_F, x_up_parent_O, x_up_child_id, x_up_lineage)
                (x_up_C, x_up_F, x_up_lineage) = self.order_CFs(x_up_C, x_up_F, x_up_lineage)
                target_stride = 2 ** (len(data_ls) + skipped_depths - depth - 1)
                x_up_F = self.target_embedding(x_up_F, x_up_C, target_stride=target_stride)
                x_up_F = self.apply_coarse_grid_prior(x_up_C, x_up_F, depth, target_stride)
                x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
                x_up = self.run_target_resnet(x_up, depth=depth, encoded_depths=encoded_depths, skipped_depths=skipped_depths)
                x_scored_C = x_up.coords
                x_scored_F = x_up.feats
                x_scored_lineage = x_up_lineage
                if near_radius is not None and far_radius is not None:
                    if depth == encoded_depths - 1:
                        (x_up_C, _, x_up_F, _) = op.separation(x_scored_C, x_scored_F, R=far_radius)
                        if x_scored_lineage is not None:
                            (_, _, x_up_lineage, _) = op.separation(x_scored_C, x_scored_lineage, R=far_radius)
                    elif depth == encoded_depths - 2:
                        (x_up_C, _, x_up_F, _) = op.separation(x_scored_C, x_scored_F, R=near_radius / 2)
                        if x_scored_lineage is not None:
                            (_, _, x_up_lineage, _) = op.separation(x_scored_C, x_scored_lineage, R=near_radius / 2)
                    else:
                        (x_up_C, x_up_F) = (x_scored_C, x_scored_F)
                else:
                    (x_up_C, x_up_F) = (x_scored_C, x_scored_F)
                (x_up_C, x_up_F, x_up_lineage) = self.order_CFs(x_up_C, x_up_F, x_up_lineage)
                x_up = SparseTensor(coords=x_up_C, feats=x_up_F)
            if not self.uses_uniform_prior(depth):
                if self.uses_joint_head(x_up.feats.shape[0]):
                    (prob_s0, prob_s1) = self.predict_joint_two_stage(
                        x_up.feats,
                        gt_s0[:, 0],
                    )
                else:
                    (contextual_feats, prob_s0) = self.predict_s0(x_up.feats)
                    prob_s1 = self.predict_s1_from_s0(contextual_feats, prob_s0, gt_s0[:, 0])
                propagated_feats = self.make_cross_depth_features(x_up_C, x_up.feats, gt_s0[:, 0], gt_s1[:, 0])
                propagated_core_idx = int(core_idx) if propagated_feats is not None else None
                lineage_feats = self.update_lineage_memory(x_up.feats, gt_s0[:, 0], gt_s1[:, 0], x_up_lineage, depth)
            selected_s0 = prob_s0.float().gather(1, gt_s0.long())
            selected_s1 = prob_s1.float().gather(1, gt_s1.long())
            depth_s0_bits = torch.clamp(-torch.log2(selected_s0 + 1e-10), 0, 50).sum()
            depth_s1_bits = torch.clamp(-torch.log2(selected_s1 + 1e-10), 0, 50).sum()
            s0_bits = s0_bits + depth_s0_bits
            s1_bits = s1_bits + depth_s1_bits
            total_bits = total_bits + depth_s0_bits + depth_s1_bits
            (x_C, x_O) = (x_up_C, gt_x_up_O)
        rate_bpp = total_bits / max(int(original_points), 1)
        return {
            'loss': rate_bpp,
            'rate_bpp': rate_bpp,
            's0_bpp': s0_bits / max(int(original_points), 1),
            's1_bpp': s1_bits / max(int(original_points), 1),
        }

    def forward_rate(
        self,
        x,
        skipped_depths=0,
        near_radius=None,
        far_radius=None,
        return_details=False,
    ):
        original_points = x.coords.shape[0]
        data_ls = self.build_occupancy_pyramid(x, skipped_depths=skipped_depths)
        result = self._rate_pass_from_pyramid(
            data_ls,
            original_points=original_points,
            skipped_depths=skipped_depths,
            near_radius=near_radius,
            far_radius=far_radius,
        )
        return result if return_details else result['loss']

    def forward(self, x):
        return self.forward_rate(x)

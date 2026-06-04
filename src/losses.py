import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
        features = F.normalize(features, p=2, dim=1)
        labels = labels.view(-1, 1)
        batch_size = features.size(0)

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = torch.eq(labels, labels.T) & ~self_mask
        valid_anchor_mask = positive_mask.sum(dim=1) > 0
        valid_anchor_count = int(valid_anchor_mask.sum().item())

        if valid_anchor_count == 0:
            return features.new_tensor(0.0), 0

        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1)
        loss = -mean_log_prob_pos[valid_anchor_mask].mean()
        return loss, valid_anchor_count

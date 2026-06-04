from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class RobertaSupConClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        contrastive_dim: int = 128,
        pooling: str = "cls",
        use_location_priority_branch: bool = False,
        location_priority_embedding_dim: int = 8,
    ):
        super().__init__()
        self.pooling = pooling
        self.use_location_priority_branch = use_location_priority_branch
        self.encoder_config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.encoder_config)
        hidden_size = self.encoder_config.hidden_size
        classifier_input_size = hidden_size

        if self.use_location_priority_branch:
            self.location_priority_embedding = nn.Embedding(3, location_priority_embedding_dim)
            classifier_input_size += location_priority_embedding_dim
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_input_size, num_labels)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, contrastive_dim),
        )

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-12)
        return last_hidden_state[:, 0, :]

    def forward(self, input_ids, attention_mask, token_type_ids=None, location_priority_ids=None):
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**model_inputs)
        sentence_embedding = self._pool(outputs.last_hidden_state, attention_mask)
        classifier_embedding = sentence_embedding
        if self.use_location_priority_branch:
            if location_priority_ids is None:
                location_priority_ids = torch.ones(input_ids.size(0), dtype=torch.long, device=input_ids.device)
            location_embedding = self.location_priority_embedding(location_priority_ids)
            classifier_embedding = torch.cat([sentence_embedding, location_embedding], dim=1)
        logits = self.classifier(self.dropout(classifier_embedding))
        contrastive_embedding = F.normalize(self.projection_head(sentence_embedding), p=2, dim=1)
        return {
            "logits": logits,
            "sentence_embedding": sentence_embedding,
            "contrastive_embedding": contrastive_embedding,
        }

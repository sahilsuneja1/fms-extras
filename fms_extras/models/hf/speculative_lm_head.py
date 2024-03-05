from typing import Optional

import torch
import torch.nn as nn
from fms.models.hf.lm_head_mixins import LMHeadMixin
from torch.nn.modules.loss import _Loss
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import ModelOutput


class SpeculativeLMHead(LMHeadMixin):
    def _get_empty_lm_head(self, **kwargs) -> nn.Module:
        pass

    def _compute_loss(self, prediction: torch.Tensor, labels: torch.Tensor) -> _Loss:
        raise NotImplementedError("")

    def _lm_head(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        pass

    def _produce_lm_output(self, logits: torch.FloatTensor, loss: _Loss,
                           encoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions],
                           decoder_outputs: Optional[BaseModelOutputWithPastAndCrossAttentions]) -> ModelOutput:
        pass
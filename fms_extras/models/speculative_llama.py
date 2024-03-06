import re
from dataclasses import dataclass, field
from typing import List, Optional

from fms.distributed.strategy import DistributedStrategy, NoOpStrategy

from fms import models
from fms.utils import serialization

from fms_extras.models.paged_llama import PagedLLaMAConfig, PagedLLaMAHeadless, PagedLLaMA
import torch.nn as nn
import torch
from fms_extras.models.speculator import MLPSpeculator, select_inflate_dim, flatten_batch
from fms_extras.utils.cache.paged import PagedAttentionCacheData


@dataclass
class SpeculativeLLaMAConfig(PagedLLaMAConfig):

    n_predict: int = 3
    inner_dim: int = 4096
    top_k: int = 5
    threshes: List[int] = field(default_factory=lambda: [5, 3, 2])


class SpeculativeLLaMA(nn.Module):

    def __init__(
        self,
        model: PagedLLaMA,
        speculator: MLPSpeculator,
        config: Optional[SpeculativeLLaMAConfig] = None,
        distributed_strategy: DistributedStrategy = NoOpStrategy,
        **kwargs,
    ):
        super(SpeculativeLLaMA, self).__init__()
        if config is not None:
            self.config = config
        else:
            self.config = SpeculativeLLaMAConfig()
        self.config = self.config.updated(**model.config.as_dict())
        self.config = self.config.updated(**kwargs)
        self.distributed_strategy = distributed_strategy

        self.headless_model = model.headless_model
        self.speculator_head = speculator
        self.lm_head = model.head

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_data: PagedAttentionCacheData,
        embeds: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        attn_algorithm: Optional[str] = None,
        flatting: bool = True,
    ):
        # assume this is always used for inference
        if not cache_data.is_filled():
            return self.headless_model(
                x_in=input_ids,
                position_ids=position_ids,
                mask=mask,
                cache_data=cache_data,
                use_cache=use_cache,
                attn_algorithm=attn_algorithm
            )
        else:
            if embeds is None:
                raise ValueError("once the cache is filled we must provide embeds")

            bsize = input_ids.size(0)
            n_adds = self.config.n_predict + 1
            adds = self.speculator_head.generate_suffixes(embeds, input_ids, self.config.threshes, self.config.top_k)  # b k h
            input_ids = torch.cat(
                [input_ids.unsqueeze(1).expand(bsize, self.config.top_k, 1), adds], dim=-1
            ).int()  # b k 1+h

            this_flatting = False
            if flatting:
                flat_inputs, unflat_indices, flat_indices = flatten_batch(
                    input_ids
                )  # b', b k 1+h, b'
                compression = flat_inputs.numel() / input_ids.numel()
                if compression < 0.75:
                    this_flatting = True
                    flat_inputs = flat_inputs[None,]  # 1 b'
                    cache_data.unflatten_indices = unflat_indices
                    cache_data.flatten_indices = flat_indices
                    position_ids = select_inflate_dim(position_ids.view(-1), flat_indices)[
                        None,
                    ]
            input_ids = input_ids.view(-1, n_adds)  # bk 1+h

            context_lengths = cache_data.context_lengths  # bk
            inflate_factor = (
                cache_data.query_length
                if cache_data.unflatten_indices is None
                else cache_data.unflatten_indices.size(-1)
            )
            # no reason to check type here as generation allocation always returns context_lengths
            context_lengths = context_lengths.unsqueeze(1).expand(  # type: ignore
                -1, inflate_factor
            )  # bk n
            context_lengths = (
                context_lengths.sub(context_lengths.sign().cumsum(1).flip([1]).sub(1))
                .int()
                .view(-1)
            )  # bkn
            block_mappings = cache_data.block_mapping.repeat_interleave(
                inflate_factor, dim=0
            )  # bkn n_blocks
            if cache_data.flatten_indices is not None:
                context_lengths = select_inflate_dim(
                    context_lengths, cache_data.flatten_indices
                )  # n'
                block_mappings = select_inflate_dim(
                    block_mappings, cache_data.flatten_indices
                )  # n' n_blocks

            cache_data.block_mapping = block_mappings
            cache_data.context_lengths = context_lengths

            input_ids_unflat = input_ids
            if this_flatting:
                input_ids = flat_inputs

            embeds, cache = self.headless_model(
                x_in=input_ids,
                position_ids=position_ids,
                cache_data=cache_data,
                use_cache=use_cache,
                attn_algorithm=attn_algorithm
            )

            logits = self.lm_head(embeds, reverse=True)
            next_vals = torch.argmax(logits, dim=-1)  # 1 n'

            if this_flatting:
                unflat_indices = unflat_indices.view(-1, unflat_indices.size(2))
                next_vals = select_inflate_dim(next_vals[0], unflat_indices)  # bk 1+h
                embeds = select_inflate_dim(embeds[0], unflat_indices)  # bk 1+h d
                # TODO: make more efficient by best guessing out of unflat indices rather than from here directly
            else:
                next_vals = next_vals.view(-1, n_adds)
                embeds = embeds.view(next_vals.size(0), n_adds, -1)

            # Check correctness of speculator predictions
            test = input_ids_unflat.roll(-1, 1).eq(next_vals).cumprod(1)
            n_correct = (
                test.sum(1).clamp(0, n_adds - 1).view(bsize, self.config.top_k)
            )  # clamp in case pred[0]==targ[-1]
            best_guess = n_correct.argmax(1)  # b
            best_guess_unflat = (
                best_guess.unsqueeze(1).expand(bsize, n_adds).unsqueeze(1)
            )  # b 1 1+h

            # Set global values to those of best guess
            next_vals = (
                next_vals.view(bsize, self.config.top_k, n_adds).gather(1, best_guess_unflat).squeeze(1)
            )  # b 1+h
            n_correct = n_correct.gather(1, best_guess.unsqueeze(1)).squeeze(1)  # b
            embeds = (
                embeds.view(bsize, self.config.top_k, *embeds.size()[1:])
                .gather(
                    1, best_guess_unflat.unsqueeze(3).expand(-1, -1, -1, embeds.size(2))
                )
                .squeeze(1)
            )  # b 1+h d

            # Toss any wrong speculator tokens
            next_vals_split = list(next_vals)
            next_vals_split = [
                next_vals_split[i][: n_correct[i] + 1] for i in range(len(next_vals_split))
            ]  # [b] h'
            embeds = embeds.gather(
                1, n_correct.view(-1, 1, 1).expand(-1, -1, embeds.size(2))
            )  # Grab last correct embed

            return next_vals_split, cache, embeds, best_guess, n_correct

_7b_config = SpeculativeLLaMAConfig()
_13b_config = SpeculativeLLaMAConfig(emb_dim=5120, nheads=40, nlayers=40)


_architecture_name = "speculative_llama"


def _llama_factory_factory(config):
    def factory(**kwargs):
        return SpeculativeLLaMA(config, **kwargs)

    return factory


models.register_model(_architecture_name, "7b", _llama_factory_factory(_7b_config))
models.register_model(_architecture_name, "13b", _llama_factory_factory(_13b_config))

def _rename_weights_to_fms(orig_sd):
    replacements = [
        (r"^tok_embeddings", "headless_model.shared.emb"),
        (r"^norm", "headless_model.dec_norm"),
        (r"^output", "lm_head.head"),
        (r"^layers", "headless_model.layers"),
        (r"\.attention\.", ".attn."),
        (r"attn\.wq", "attn.query"),
        (r"attn\.wk", "attn.key"),
        (r"attn\.wv", "attn.value"),
        (r"attn\.wo", "attn.dense"),
        (r"attention_norm", "ln"),
        (r"feed_forward\.w1", "ff_sub_layer.wg"),
        (r"feed_forward\.w2", "ff_sub_layer.w2"),
        (r"feed_forward\.w3", "ff_sub_layer.w1"),
        (r"ffn_norm", "ff_ln"),
        (r"^emb", "speculator_head.emb"),
        (r"^proj", "speculator_head.proj"),
        (r"^head", "speculator_head.head"),
        (r"^ln", "speculator_head.ln"),
    ]
    new_sd = {}
    for name, param in orig_sd.items():
        new_name = name
        for pattern, repl in replacements:
            new_name = re.sub(pattern, repl, new_name)
        new_sd[new_name] = param

        # llama in meta has unfused qkv attn weights, so these weights must be converted to fused weights in fms
        if (
            "attn.query" in new_name
            or "attn.key" in new_name
            or "attn.value" in new_name
        ):
            unfused_weights = [
                re.sub(r"w[qkv]", "wq", name),
                re.sub(r"w[qkv]", "wk", name),
                re.sub(r"w[qkv]", "wv", name),
            ]
            missing_weights = [w for w in unfused_weights if w not in orig_sd.keys()]
            if len(missing_weights) != 0:
                raise serialization.FusableWeightsMissingError(missing_weights)

            new_sd[
                re.sub(r"attn.(query|key|value)", "attn.qkv_fused", new_name)
            ] = torch.cat([orig_sd[w] for w in unfused_weights], dim=0)

    return new_sd

serialization.register_adapter(_architecture_name, "meta", _rename_weights_to_fms)
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit initialization, readout scaling, and depth residual variants.

CompleteP's Table 1 supplies the width initialization and depth exponents.
Muon uses spectral updates, so its optimizer scaling is specified separately.
Full AttnRes implements arXiv:2603.15031 equations 2--4, with HybridNorm
applied to each stored branch output. It is an architectural comparison,
not a claimed CompleteP theorem for attention over depth.
"""

import dataclasses
from enum import StrEnum

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
from haliax.jax_utils import maybe_rng_split, shaped_rng_split
from haliax.nn.scan import Stacked
from levanter.models.llama import LlamaDecoderLayer, LlamaEmbedding, LlamaLMHeadModel
from levanter.models.lm_model import LmConfig
from levanter.models.qwen import Qwen3Config


class Residual(StrEnum):
    STANDARD = "standard"
    COMPLETEP = "completep"
    DEPTH_MUP = "depth_mup"
    FULL_ATTNRES = "full_attnres"


@LmConfig.register_subclass("hybrid_transfer")
@dataclasses.dataclass(frozen=True)
class TransferConfig(Qwen3Config):
    implementation_digest: str = ""
    base_width: int = 256
    base_depth: int = 4
    residual: Residual = Residual.STANDARD
    hybrid_norm: bool = True
    use_bias: bool = False
    tie_word_embeddings: bool = False

    @property
    def width_multiplier(self):
        return self.hidden_dim / self.base_width

    @property
    def depth_multiplier(self):
        return self.num_layers / self.base_depth

    @property
    def depth_alpha(self):
        return {Residual.COMPLETEP: 1.0, Residual.DEPTH_MUP: 0.5}.get(self.residual, 0.0)

    @property
    def branch_scale(self):
        return self.depth_multiplier ** (-self.depth_alpha)

    @property
    def hidden_lr_multiplier(self):
        if self.residual in (Residual.COMPLETEP, Residual.DEPTH_MUP):
            return self.depth_multiplier ** (self.depth_alpha - 1)
        return 1.0

    def attention_config(self):
        cfg = super().attention_config()
        return dataclasses.replace(cfg, scaling_factor=1.0 / cfg.HeadSize.size)

    @property
    def model_type(self):
        return TransferLM

    def build(self, Vocab, *, key):
        if not self.hybrid_norm or self.tie_word_embeddings or self.use_bias:
            raise ValueError("This experiment requires HybridNorm, untied readout, and no biases")
        return TransferLM.init(Vocab, self, key=key)


class TransferLayer(LlamaDecoderLayer):
    @staticmethod
    def init(config, *, key):
        layer = LlamaDecoderLayer.init(config, key=key)
        return TransferLayer(**{f.name: getattr(layer, f.name) for f in dataclasses.fields(layer)})

    def attention_branch(self, x, mask, *, key=None, pos_ids=None):
        return self.post_attn_layernorm(self.self_attn(self.input_layernorm(x), mask=mask, key=key, pos_ids=pos_ids))

    def mlp_branch(self, x, *, key=None):
        return self.post_mlp_layernorm(self.mlp(self.post_attention_layernorm(x), key=key))

    def __call__(self, x, mask, *, key=None, pos_ids=None):
        ka, km = maybe_rng_split(key, 2)
        x = x + self.config.branch_scale * self.attention_branch(x, mask, key=ka, pos_ids=pos_ids)
        return x + self.config.branch_scale * self.mlp_branch(x, key=km)


class ResidualAttention(eqx.Module):
    query: hax.NamedArray
    norm: hnn.RmsNorm

    @staticmethod
    def init(config):
        return ResidualAttention(hax.zeros(config.Embed), config.mk_LayerNorm(config.Embed))

    def __call__(self, history):
        History = hax.Axis("residual_history", len(history))
        values = hax.stack(History, history)
        # Accumulate scores and softmax in float32; preserve the value dtype.
        keys = self.norm(values).astype(jnp.float32)
        logits = hax.dot(keys, self.query.astype(jnp.float32), axis=self.query.axes)
        weights = hnn.softmax(logits, axis=History)
        return hax.sum(weights * values, axis=History).astype(values.dtype)


class TransferTransformer(eqx.Module):
    config: TransferConfig = eqx.field(static=True)
    layers: Stacked | tuple
    norm: hnn.RmsNorm
    residual_attention: tuple[ResidualAttention, ...]

    @staticmethod
    def init(config, *, key):
        keys = shaped_rng_split(key, config.num_layers)
        if config.residual == Residual.FULL_ATTNRES:
            layers = tuple(TransferLayer.init(config, key=k) for k in keys)
            # One read for each branch input and one final read for the LM head.
            readers = tuple(ResidualAttention.init(config) for _ in range(2 * config.num_layers + 1))
        else:
            layers = Stacked.init(config.Layers, TransferLayer, gradient_checkpointing=config.gradient_checkpointing)(
                config, key=keys
            )
            readers = ()
        return TransferTransformer(config, layers, config.mk_LayerNorm(config.Embed), readers)

    def __call__(self, x, attn_mask, *, key=None, pos_ids=None):
        keys = maybe_rng_split(key, self.config.num_layers) if key is not None else None
        if self.config.residual != Residual.FULL_ATTNRES:
            return self.norm(self.layers.fold(x, mask=attn_mask, key=keys, pos_ids=pos_ids))
        history = [x]
        for i, layer in enumerate(self.layers):
            ka, km = maybe_rng_split(keys[i] if keys is not None else None, 2)
            x = self.residual_attention[2 * i](history)
            attn = (
                eqx.filter_checkpoint(layer.attention_branch)
                if self.config.gradient_checkpointing
                else layer.attention_branch
            )
            history.append(attn(x, attn_mask, key=ka, pos_ids=pos_ids))
            x = self.residual_attention[2 * i + 1](history)
            mlp = eqx.filter_checkpoint(layer.mlp_branch) if self.config.gradient_checkpointing else layer.mlp_branch
            history.append(mlp(x, key=km))
        return self.norm(self.residual_attention[-1](history))


class TransferLM(LlamaLMHeadModel):
    transformer: TransferTransformer

    @classmethod
    def init(cls, Vocab, config, *, key):
        kt, ke, kh, ki = jax.random.split(key, 4)
        transformer = TransferTransformer.init(config, key=kt)
        embeddings = LlamaEmbedding.init(Vocab, config, key=ke)
        head = hnn.Linear.init(config.Embed, Vocab, key=kh, use_bias=False, out_first=True)
        model = cls(transformer, embeddings, head)
        # Draw actual normal weights explicitly: initializer_range in native
        # Qwen3 does not set the hidden Linear initialization in this revision.
        leaves, treedef = jax.tree.flatten(model, is_leaf=lambda x: isinstance(x, (hnn.Linear, hnn.Embedding)))
        rngs = iter(jax.random.split(ki, len(leaves)))
        result = []
        for leaf in leaves:
            rng = next(rngs)
            if isinstance(leaf, (hnn.Linear, hnn.Embedding)):
                is_readout = isinstance(leaf, hnn.Linear) and leaf.Out == Vocab
                std = config.initializer_range
                if isinstance(leaf, hnn.Linear) and not is_readout:
                    std /= config.width_multiplier**0.5
                leaf = dataclasses.replace(leaf, weight=hax.random.normal(rng, leaf.weight.axes) * std)
            result.append(leaf)
        return jax.tree.unflatten(treedef, result)

    def get_lm_head(self):
        return super().get_lm_head() / self.config.width_multiplier

    def __call__(self, *args, **kwargs):
        return super().__call__(*args, **kwargs) / self.config.width_multiplier

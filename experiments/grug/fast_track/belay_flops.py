# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Useful matrix FLOPs for the frozen base16 d512 recipe, not generic Grug.

Count two FLOPs per MAC and three times forward work for training. Exclude
elementwise operations, lookups, optimizer/EMA, padding, rematerialization and
the extra work introduced by the parallel KDA algorithm. Attention is causal
and document-local. This convention is separate from the legacy trainer metric.
"""

from collections.abc import Sequence


def training_flops_per_token(
    document_lengths: Sequence[int],
    *,
    batch_size: int,
    dropped_assignments: Sequence[int] = (0, 0, 0, 0, 0, 0),
) -> dict[str, float]:
    """Return the matrix-FLOP breakdown for equally packed batch rows.

    Dropped assignments are global counts in layer order, after both sender
    and receiver capacity limits. Omitting them gives nominal top-eight work.
    """
    if not document_lengths or any(n <= 0 for n in document_lengths) or batch_size <= 0:
        raise ValueError("Document lengths and batch size must be positive")
    tokens = batch_size * sum(document_lengths)
    if len(dropped_assignments) != 6 or any(n < 0 or n > 8 * tokens for n in dropped_assignments):
        raise ValueError("Expected six global per-layer dropped-route counts")
    # Each entry counts active weight uses per token, before the 6x training factor.
    uses = {
        "lm_head": 512 * 16384,
        "kda_projections": 4 * (5 * 512 * 512 + 2 * 512 * 128 + 512 * 4),
        "mla_projections": 2 * (4 * 512 * 768 + 512 * 512 + 2 * 512 * 6),
        "gated_norm_projections": 6 * 2 * 2 * 512 * 128,
        "routers": 6 * 512 * 512,
        "latent_projections": 6 * 512 * 256 + 4 * 256 * 512,
        "shared_experts": 6 * 2 * 512 * 768,
        "bigram_gate": 2 * 512 * 16,
        "inkling_projection": 2 * 512 * 96,
        "inkling_bias_bank": 2 * 6 * 16 * 1024,
    }
    result = {name: float(6 * value) for name, value in uses.items()}
    result["routed_experts"] = sum(
        6 * 384 * (256 + write) * (8 - dropped / tokens)
        for write, dropped in zip((256, 256, 256, 256, 512, 512), dropped_assignments, strict=True)
    )
    length = sum(document_lengths)
    valid_taps = sum(max(n - offset, 0) for n in document_lengths for offset in range(4))
    result["short_convolutions"] = 6 * 20 * 512 * valid_taps / length
    result["kda_recurrent_contractions"] = float(3 * 6 * 4 * 4 * 128 * 128)
    pairs = sum(n * (n + 1) // 2 for n in document_lengths)
    result["causal_attention_contractions"] = 3 * 4 * 2 * 6 * 128 * pairs / length
    return result

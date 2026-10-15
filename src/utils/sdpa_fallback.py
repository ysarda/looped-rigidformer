"""Flash-Attention stand-in built on PyTorch SDPA.

`src/models/decoder.py` calls `flash_attn_func` on its unmasked attention path. The
flash-attn wheel has no Windows build and compiling it needs the CUDA toolkit, so this
supplies the same function through `torch.nn.functional.scaled_dot_product_attention`,
which is exact attention and differs only in kernel and memory profile. The decoder
imports this only when the real package is missing, so an environment that has
flash-attn installed keeps using it.

The two differ in layout: flash-attn takes and returns (batch, seq, heads, dim), SDPA
works in (batch, heads, seq, dim). The decoder's own masked branch does the same
transpose inline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attn_func(
    q: torch.Tensor,            # (B, N, H, D)
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    **_unused,
) -> torch.Tensor:
    """Signature-compatible with `flash_attn.flash_attn_func` for the decoder's usage."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=dropout_p,
        is_causal=causal,
        scale=softmax_scale,
    )                            # (B, H, N, D)
    return out.transpose(1, 2)   # back to (B, N, H, D)

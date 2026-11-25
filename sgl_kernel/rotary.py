# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding kernel mirroring sgl_kernel.rotary_embedding (pos_enc.cu).

    rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)

In-place on query/key, rows indexed by token:
    for token i at position p = positions[i], every head vector is rotated by
    cos/sin of p. query/key are (num_tokens, num_heads * head_size), cache is
    (max_pos, rotary_dim) fp32 as built by sglang's RotaryEmbedding layer.

Layouts (both must match the CUDA kernel exactly):
    is_neox=True  — NeoX/rotate_half: pair x[i] with x[i + rotary_dim/2];
                    cache row = [cos(rotary_dim/2) | sin(rotary_dim/2)].
    is_neox=False — GPT-J interleaved: pair even lane 2j with odd lane 2j+1;
                    cache row = [cos0, sin0, cos1, sin1, ...] interleaved.
"""

from __future__ import annotations

import torch

from .compat import HAS_TRITON, check_cuda_tensor

if HAS_TRITON:
    import triton
    import triton.language as tl

    @triton.jit
    def _rope_neox_kernel(
        POS,  # (num_tokens,) int64
        T,  # (num_tokens, num_heads * head_size) row-major, updated in place
        t_stride,
        COS_SIN,  # (max_pos, rotary_dim) fp32
        num_heads,
        HEAD_SIZE: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        BLOCK_ROTARY_HALF: tl.constexpr,
        BLOCK_HEADS: tl.constexpr,
    ):
        token = tl.program_id(0).to(tl.int64)
        head_blk = tl.program_id(1)
        half: tl.constexpr = ROTARY_DIM // 2
        half_idx = tl.arange(0, BLOCK_ROTARY_HALF)
        rotary_mask = half_idx < half

        pos = tl.load(POS + token)
        c = tl.load(COS_SIN + pos * ROTARY_DIM + half_idx, mask=rotary_mask, other=0.0)
        s = tl.load(COS_SIN + pos * ROTARY_DIM + half + half_idx, mask=rotary_mask, other=0.0)

        heads = head_blk * BLOCK_HEADS + tl.arange(0, BLOCK_HEADS)
        hmask = heads < num_heads
        mask = hmask[:, None] & rotary_mask[None, :]
        off = tl.where(hmask, heads, 0) * HEAD_SIZE

        base = T + token * t_stride
        x1 = tl.load(base + off[:, None] + half_idx[None, :],
                     mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(base + off[:, None] + (half + half_idx)[None, :],
                     mask=mask, other=0.0).to(tl.float32)
        y1 = x1 * c[None, :] - x2 * s[None, :]
        y2 = x1 * s[None, :] + x2 * c[None, :]
        tl.store(base + off[:, None] + half_idx[None, :],
                 y1.to(base.dtype.element_ty), mask=mask)
        tl.store(base + off[:, None] + (half + half_idx)[None, :],
                 y2.to(base.dtype.element_ty), mask=mask)


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool = True,
) -> None:
    """Rotate query (and key) rows in place. See module docstring."""
    if positions.numel() == 0:
        return
    if positions.dtype != torch.int64:
        # The triton kernel indexes the cache with 64-bit math; an int32
        # positions tensor would overflow pos * rotary_dim on long
        # contexts (upstream pos_enc.cu loads positions as int64_t).
        positions = positions.long()
    use_triton = HAS_TRITON and query.is_cuda and cos_sin_cache.is_cuda and is_neox
    if key is not None and use_triton:
        use_triton = key.is_cuda
    if not use_triton:
        impl = (
            _rotary_embedding_neox_torch
            if is_neox
            else _rotary_embedding_interleaved_torch
        )
        impl(positions, query, key, head_size, cos_sin_cache)
        return

    check_cuda_tensor(positions, query, cos_sin_cache)
    if key is not None:
        check_cuda_tensor(key)

    rotary_dim = cos_sin_cache.shape[1]
    num_tokens = positions.shape[0]
    contiguous_in = query.is_contiguous() and (key is None or key.is_contiguous())
    q = query if query.is_contiguous() else query.contiguous()
    k = None
    if key is not None:
        k = key if key.is_contiguous() else key.contiguous()

    BLOCK_HEADS = 8
    q_heads = q.shape[-1] // head_size
    grid = (num_tokens, triton.cdiv(q_heads, BLOCK_HEADS))
    _rope_neox_kernel[grid](
        positions,
        q,
        q.stride(0),
        cos_sin_cache,
        q_heads,
        HEAD_SIZE=head_size,
        ROTARY_DIM=rotary_dim,
        BLOCK_ROTARY_HALF=triton.next_power_of_2(rotary_dim // 2),
        BLOCK_HEADS=BLOCK_HEADS,
        num_warps=4,
    )
    if k is not None and k.data_ptr() != q.data_ptr():
        k_heads = k.shape[-1] // head_size
        grid_k = (num_tokens, triton.cdiv(k_heads, BLOCK_HEADS))
        _rope_neox_kernel[grid_k](
            positions,
            k,
            k.stride(0),
            cos_sin_cache,
            k_heads,
            HEAD_SIZE=head_size,
            ROTARY_DIM=rotary_dim,
            BLOCK_ROTARY_HALF=triton.next_power_of_2(rotary_dim // 2),
            BLOCK_HEADS=BLOCK_HEADS,
            num_warps=4,
        )

    if not contiguous_in:
        if q.data_ptr() != query.data_ptr():
            query.copy_(q.view(query.shape))
        if k is not None and k.data_ptr() != key.data_ptr():
            key.copy_(k.view(key.shape))


def _rotary_embedding_neox_torch(positions, query, key, head_size, cos_sin_cache):
    """Pure-torch NeoX reference (also the CPU path)."""
    rotary_dim = cos_sin_cache.shape[1]
    pos = positions.long()
    cos = cos_sin_cache[pos, : rotary_dim // 2].float()
    sin = cos_sin_cache[pos, rotary_dim // 2:].float()
    num_tokens = positions.shape[0]

    def _rope(t):
        if t is None:
            return None
        orig_shape = t.shape
        t = t.view(num_tokens, -1, head_size)
        x1 = t[..., : rotary_dim // 2].float()
        x2 = t[..., rotary_dim // 2 : rotary_dim].float()
        y1 = x1 * cos.unsqueeze(1) - x2 * sin.unsqueeze(1)
        y2 = x2 * cos.unsqueeze(1) + x1 * sin.unsqueeze(1)
        out = t.clone()
        out[..., : rotary_dim // 2] = y1.to(t.dtype)
        out[..., rotary_dim // 2 : rotary_dim] = y2.to(t.dtype)
        return out.view(orig_shape)

    out_q = _rope(query)
    if out_q is not None:
        query.copy_(out_q)
    out_k = _rope(key)
    if out_k is not None:
        key.copy_(out_k)


def _rotary_embedding_interleaved_torch(positions, query, key, head_size, cos_sin_cache):
    """GPT-J interleaved layout torch reference.

    sglang's RotaryEmbedding always stores the cache half-split
    ([cos(rotary_dim/2) | sin(rotary_dim/2)]), independent of
    is_neox_style; only the lane pairing of the rotated vector differs.
    Pair even lane 2j with odd lane 2j+1.
    """
    rotary_dim = cos_sin_cache.shape[1]
    pos = positions.long()
    cache = cos_sin_cache[pos].float()  # (T, rotary_dim)
    cos_i = cache[:, : rotary_dim // 2]
    sin_i = cache[:, rotary_dim // 2:]
    num_tokens = positions.shape[0]

    def _rope(t):
        if t is None:
            return None
        orig_shape = t.shape
        t = t.view(num_tokens, -1, head_size)
        # The rotary window is the FIRST rotary_dim lanes of each head
        # vector (last dim); inside it, even lane 2j pairs with odd 2j+1.
        # (The old `t[..., :rotary_dim, 0::2]` sliced the HEAD dim by
        # rotary_dim and strided head_size — correct only by accident for
        # single-head tensors with rotary_dim == head_size.)
        # Index tensors must live on t's device: CUDA self + CPU indices
        # is a hard RuntimeError in advanced indexing (review #80 P1).
        even = torch.arange(0, rotary_dim, 2, device=t.device)
        odd = torch.arange(1, rotary_dim, 2, device=t.device)
        x1 = t[..., even].float()  # even lanes of the rotary window
        x2 = t[..., odd].float()   # odd lanes
        y1 = x1 * cos_i.unsqueeze(1) - x2 * sin_i.unsqueeze(1)
        y2 = x1 * sin_i.unsqueeze(1) + x2 * cos_i.unsqueeze(1)
        out = t.clone()
        out[..., even] = y1.to(t.dtype)
        out[..., odd] = y2.to(t.dtype)
        return out.view(orig_shape)

    out_q = _rope(query)
    if out_q is not None:
        query.copy_(out_q)
    out_k = _rope(key)
    if out_k is not None:
        key.copy_(out_k)


# ---------------------------------------------------------------------------
# MiniMax-H3 image path: minimax_h3_rotary_embedding
#
# Contract source (upstream v0.5.22, vendored in sglang-das):
#   kernels/aot/csrc/sgl_diffusion/minimax_h3_rope.cu (schema in
#   kernels/aot/csrc/common_extension_rocm.cc, CUDA-only impl).
#
# Community multimodal_gen probes
# getattr(torch.ops.sgl_kernel, "minimax_h3_rotary_embedding", None) on HCU
# (DiT dits/minimax_h3.py _apply_rope_qk, VAE vit_utils.py
# apply_rotary_pos_emb_qk) and hard-fails when the op is absent, so the
# shadow package must mirror the namespace (kvcacheio torch-op precedent:
# no routing decision, no env kill-switch, no catalog entry).
#
# Torch-composition tier only: rounded_product rounds every product to
# the input dtype BEFORE the fp32 add/subtract and the final rounding —
# bit-exact with the eager chain x_rot * cos + rotate_half(x_rot) * sin
# the kernel was written to match (negation commutes with
# round-to-nearest-even), so a fused tier needs a node to justify it.


def _minimax_h3_rope_torch(positions, query, key, head_size, cos_sin_cache):
    """Bit-exact torch mirror of the minimax_h3_rope.cu kernel math."""
    rotary_dim = cos_sin_cache.shape[1]
    half = rotary_dim // 2
    tokens, width = query.shape
    heads = width // head_size
    # One cache row per token; row = [cos(rd/2) | sin(rd/2)] (cos first).
    rows = cos_sin_cache.index_select(0, positions).float()  # (T, rotary_dim)
    cos = rows[:, :half].unsqueeze(1)  # (T, 1, half)
    sin = rows[:, half:].unsqueeze(1)

    def _rotate(t):
        packed = t.view(tokens, heads, head_size)  # split of the packed row
        x = packed[..., :half]
        y = packed[..., half:rotary_dim]
        xc = (x.float() * cos).to(t.dtype)  # rounded_product(x, cos)
        xs = (x.float() * sin).to(t.dtype)
        yc = (y.float() * cos).to(t.dtype)
        ys = (y.float() * sin).to(t.dtype)
        packed[..., :half] = (xc.float() - ys.float()).to(t.dtype)
        packed[..., half:rotary_dim] = (yc.float() + xs.float()).to(t.dtype)

    _rotate(query)
    if key is not None:
        _rotate(key)


def minimax_h3_rotary_embedding(
    positions, query, key, head_size, cos_sin_cache, is_neox=True
):
    """Rotate packed MiniMax-H3 Q/K rows in place.

    Mirrors the upstream AOT op: in-place on query (and key when given),
    one row per token at cache row positions[i], NeoX half-split inside
    each head's first rotary_dim lanes; lanes [rotary_dim, head_size)
    pass through untouched.
    """
    # Guard order and messages mirror minimax_h3_rope.cu verbatim. The only
    # dropped clause is the .cu's `query.is_cuda()` device guard: this torch
    # tier is device-agnostic (the community never calls it with CPU
    # tensors; HCU device tensors reach it via the CUDA dispatch key).
    if not is_neox:
        raise RuntimeError("MiniMax-H3 RoPE requires the NeoX layout")
    if query.dim() != 2 or query.stride(1) != 1:
        raise RuntimeError(
            "MiniMax-H3 RoPE requires packed GPU Q/K with contiguous heads"
        )
    if (
        positions.device != query.device
        or positions.dtype != torch.int64
        or positions.dim() != 1
        or not positions.is_contiguous()
        or positions.numel() != query.size(0)
    ):
        raise RuntimeError(
            "MiniMax-H3 RoPE requires one GPU int64 position per token"
        )
    if head_size <= 0 or query.size(1) <= 0 or query.size(1) % head_size != 0:
        raise RuntimeError("Invalid MiniMax-H3 attention head size")
    if (
        cos_sin_cache.device != query.device
        or cos_sin_cache.dtype != query.dtype
        or cos_sin_cache.dim() != 2
        or not cos_sin_cache.is_contiguous()
    ):
        raise RuntimeError(
            "MiniMax-H3 RoPE cache must have the Q/K device and dtype"
        )
    rotary_dim = cos_sin_cache.size(1)
    if rotary_dim <= 0 or rotary_dim % 2 != 0 or rotary_dim > head_size:
        raise RuntimeError("Invalid MiniMax-H3 rotary dimension")
    if key is not None:
        if (
            key.device != query.device
            or key.dtype != query.dtype
            or key.sizes() != query.sizes()
            or key.stride(1) != 1
        ):
            raise RuntimeError("MiniMax-H3 RoPE requires matching packed Q/K")
    if query.size(0) == 0:
        # .cu: the empty-batch return sits AFTER the shape guards and
        # BEFORE the dtype dispatch (an empty float64 batch is accepted).
        return
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        # DISPATCH_FLOAT_TYPES is redefined by the vendored aot utils.h to
        # {Float, Half, BFloat16}: float64 is not dispatchable upstream.
        raise RuntimeError(
            "minimax_h3_rotary_embedding not implemented for %s" % query.dtype
        )
    _minimax_h3_rope_torch(positions, query, key, head_size, cos_sin_cache)


_MINIMAX_H3_ROPE_LIB = None  # keep the Library handle alive: GC unregisters its ops


def _register_minimax_h3_rope_torch_op() -> None:
    global _MINIMAX_H3_ROPE_LIB
    if hasattr(torch.ops.sgl_kernel, "minimax_h3_rotary_embedding"):
        return
    lib = torch.library.Library("sgl_kernel", "FRAGMENT")
    lib.define(
        "minimax_h3_rotary_embedding(Tensor positions, Tensor! query, "
        "Tensor!? key, int head_size, Tensor cos_sin_cache, bool is_neox) -> ()"
    )
    # CUDA is the key the community's HCU calls dispatch on; the CPU key
    # keeps the torch.ops seam exercisable on any host (kvcacheio pattern).
    lib.impl("minimax_h3_rotary_embedding", minimax_h3_rotary_embedding, "CPU")
    lib.impl("minimax_h3_rotary_embedding", minimax_h3_rotary_embedding, "CUDA")
    _MINIMAX_H3_ROPE_LIB = lib


_register_minimax_h3_rope_torch_op()

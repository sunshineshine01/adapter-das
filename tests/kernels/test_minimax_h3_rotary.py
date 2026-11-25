# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""AOT contract anchors: minimax_h3_rotary_embedding (minimax_h3_rope.cu).

Contract sources (upstream v0.5.22, vendored in sglang-das):
  - kernels/aot/csrc/sgl_diffusion/minimax_h3_rope.cu
      * in-place on query/key, one row per token, cache row = positions[i]
      * cache row = [cos(rd/2) | sin(rd/2)] in the ACTIVATION dtype (the DiT
        builds it with dtype=q.dtype; the .cu guards cache dtype == Q dtype)
      * NeoX half-split INSIDE each head: rotary-window lane j pairs with
        lane j + rd/2; lanes [rd, head_size) pass through untouched
      * rounded_product: every product computed in fp32 then rounded to
        the input dtype BEFORE the fp32 add/subtract and the final
        rounding — bit-exact with the eager chain
        x_rot * cos + rotate_half(x_rot) * sin (negation commutes with
        round-to-nearest-even)
      * guards, in order: is_neox must be True; query 2-D with
        stride(1)==1; positions int64 1-D contiguous with
        numel == query.size(0); head_size divides width; cache 2-D
        contiguous, same device+dtype as query; rot_dim positive, even,
        <= head_size; key (when given) matches query device/dtype/sizes
        and has stride(1)==1; empty batch returns BEFORE the dtype
        dispatch
      * the vendored aot utils.h redefines DISPATCH_FLOAT_TYPES to
        {Float, Half, BFloat16} — float64 is NOT dispatchable
  - kernels/aot/csrc/common_extension_rocm.cc: schema
    "minimax_h3_rotary_embedding(Tensor positions, Tensor! query, Tensor!?
    key, int head_size, Tensor cos_sin_cache, bool is_neox) -> ()",
    CUDA-only impl — the shadow package mirrors it with a torch-composition
    tier registered for CPU+CUDA keys (kvcacheio precedent).
  - callers (HCU hard dependency): multimodal_gen
    dits/minimax_h3.py _apply_rope_qk, vaes/minimax_h3_video_vae/
    vit_utils.py apply_rotary_pos_emb_qk, sglang-plugin-das
    around_apply_rope_qk hook — all probe
    torch.ops.sgl_kernel.minimax_h3_rotary_embedding and fail loudly.

Known deliberate divergences:
  - the .cu's `query.is_cuda()` clause is dropped: the torch tier is
    device-agnostic; error messages are kept verbatim.

Run (CPU face, any host):  pytest tests/kernels/test_minimax_h3_rotary.py
Run (GPU face):            same on an HCU node (对拍待容器 without one).
"""

import pytest
import torch

import sgl_kernel

requires_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.device_count() > 0),
    reason="requires HCU device",
)

DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _make_cache(max_pos, rotary_dim, dtype):
    """DiT cache layout: [cos(rd/2) | sin(rd/2)] in the activation dtype."""
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_pos, rd/2)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1).to(dtype).contiguous()


def _eager_chain_ref(q, k, positions, head_size, cache):
    """The community eager fallback (_apply_rope_cos_sin), per-op rounding:
    (x_rot * cos) + (rotate_half(x_rot) * sin) — round(x*cos) and
    round(rotate_half(x)*sin) land in the activation dtype, the add rounds
    once more. This is precisely rounded_product, so the op must match it
    BIT-for-bit, not merely closely."""
    rd = cache.shape[1]
    half = rd // 2
    cos = cache[positions.long(), :half].unsqueeze(1)  # (T, 1, half)
    sin = cache[positions.long(), half:].unsqueeze(1)

    def rot(t):
        shape = t.shape
        v = t.view(shape[0], -1, head_size)
        out = v.clone()
        x1 = v[..., :half]
        x2 = v[..., half:rd]
        out[..., :half] = (x1 * cos) + ((-x2) * sin)
        out[..., half:rd] = (x2 * cos) + (x1 * sin)
        return out.view(shape)

    return rot(q), (rot(k) if k is not None else None)


# ------------------------------------------------ cpu face (torch tier) --

def test_torch_op_registered_at_package_import():
    # The community HCU probe: getattr on the torch.ops namespace must find
    # the op as soon as the shadow package is imported (the DiT/VAE code
    # imports `from sgl_kernel import rotary_embedding` right before it).
    assert getattr(torch.ops.sgl_kernel,
                   "minimax_h3_rotary_embedding", None) is not None


@pytest.mark.parametrize("dtype", DTYPES)
def test_bitexact_vs_eager_chain(dtype):
    # rounded_product == the eager chain's per-op rounding: BIT-identical.
    torch.manual_seed(0)
    T, H, hs, rd = 7, 4, 64, 32
    cache = _make_cache(64, rd, dtype)
    q = torch.randn(T, H * hs).to(dtype)
    k = torch.randn(T, H * hs).to(dtype)  # key sizes MUST equal query sizes
    positions = torch.tensor([0, 3, 7, 15, 33, 63, 63], dtype=torch.int64)
    q_ref, k_ref = _eager_chain_ref(q.clone(), k.clone(), positions, hs, cache)
    ret = torch.ops.sgl_kernel.minimax_h3_rotary_embedding(
        positions, q, k, hs, cache, True)
    assert ret is None  # in-place contract (schema returns nothing)
    assert torch.equal(q, q_ref)
    assert torch.equal(k, k_ref)


def test_hand_computed_pairing_and_cache_layout():
    # One hand-checked head pins pairing AND layout: rotary-window lane j
    # pairs with lane j + rd/2 (NeoX inside the head), cache row is
    # [cos | sin] with cos FIRST. All values are exact binary fractions,
    # so f32 rounding is invisible. Swapped halves or GPT-J pairing fail.
    cache = torch.tensor([[0.5, 0.25, 0.75, 0.125]])  # [c0, c1 | s0, s1]
    q = torch.tensor([[1.0, 2.0, 4.0, 8.0]])
    positions = torch.zeros(1, dtype=torch.int64)
    torch.ops.sgl_kernel.minimax_h3_rotary_embedding(
        positions, q, None, 4, cache, True)
    assert q.tolist() == [
        [1.0 * 0.5 - 4.0 * 0.75,    # y0 = x0*c0 - x2*s0
         2.0 * 0.25 - 8.0 * 0.125,  # y1 = x1*c1 - x3*s1
         4.0 * 0.5 + 1.0 * 0.75,    # y2 = x2*c0 + x0*s0
         8.0 * 0.25 + 2.0 * 0.125]  # y3 = x3*c1 + x1*s1
    ]


@pytest.mark.parametrize("dtype", DTYPES)
def test_partial_rotary_passthrough(dtype):
    # Only the first rotary_dim lanes of each head rotate; the tail passes
    # through bit-exact (the .cu writes only the rotary window).
    torch.manual_seed(1)
    T, H, hs, rd = 4, 3, 64, 16
    cache = _make_cache(64, rd, dtype)
    q = torch.randn(T, H * hs).to(dtype)
    k = torch.randn(T, H * hs).to(dtype)
    q_tail = q.view(T, H, hs)[:, :, rd:].clone()
    k_tail = k.view(T, H, hs)[:, :, rd:].clone()
    positions = torch.tensor([2, 9, 40, 63], dtype=torch.int64)
    torch.ops.sgl_kernel.minimax_h3_rotary_embedding(
        positions, q, k, hs, cache, True)
    assert torch.equal(q.view(T, H, hs)[:, :, rd:], q_tail)
    assert torch.equal(k.view(T, H, hs)[:, :, rd:], k_tail)


def test_key_none_python_symbol_and_repeated_positions():
    # key is optional (Tensor!?); the python symbol mirrors the torch-op
    # face; repeated positions must reuse the same cache row.
    cache = _make_cache(8, 8, torch.float32)
    q = torch.randn(3, 16, dtype=torch.float32)
    positions = torch.tensor([1, 6, 1], dtype=torch.int64)
    ret = sgl_kernel.minimax_h3_rotary_embedding(
        positions, q, None, 8, cache, True)
    assert ret is None
    q_ref, _ = _eager_chain_ref(q.clone(), None, positions, 8, cache)
    assert torch.equal(q, q_ref)
    assert torch.equal(q[0], q[2])  # same position -> identical rows


def test_row_stride_respected():
    # The .cu indexes query/key rows by stride(0): packed buffers with
    # padded row pitch are legal, and the padding must stay untouched.
    torch.manual_seed(2)
    T, H, hs, rd = 3, 2, 16, 16
    W = H * hs
    cache = _make_cache(16, rd, torch.float32)
    positions = torch.tensor([1, 5, 15], dtype=torch.int64)

    q_pad = torch.zeros(T, W + 8)
    q_pad[:, :W] = torch.randn(T, W)
    q_view = q_pad[:, :W]  # row pitch W + 8
    k_pad = torch.zeros(T, W + 4)
    k_pad[:, :W] = torch.randn(T, W)
    k_view = k_pad[:, :W]  # different row pitch

    q_dense, k_dense = q_view.clone(), k_view.clone()
    torch.ops.sgl_kernel.minimax_h3_rotary_embedding(
        positions, q_view, k_view, hs, cache, True)
    torch.ops.sgl_kernel.minimax_h3_rotary_embedding(
        positions, q_dense, k_dense, hs, cache, True)
    assert torch.equal(q_view, q_dense)
    assert torch.equal(k_view, k_dense)
    assert torch.equal(q_pad[:, W:], torch.zeros(T, 8))  # padding intact
    assert torch.equal(k_pad[:, W:], torch.zeros(T, 4))


def test_empty_batch_is_a_silent_noop():
    # .cu: the empty return sits AFTER the shape guards and BEFORE the
    # dtype dispatch — an empty float64 batch with a matching float64
    # cache is accepted (no dispatchable-dtype error).
    q = torch.zeros(0, 8, dtype=torch.float64)
    k = torch.zeros(0, 8, dtype=torch.float64)
    cache = torch.zeros(4, 8, dtype=torch.float64)
    positions = torch.zeros(0, dtype=torch.int64)
    ret = torch.ops.sgl_kernel.minimax_h3_rotary_embedding(
        positions, q, k, 8, cache, True)
    assert ret is None


def _valid_call():
    """Minimal valid fixture: (positions, q, k, head_size, cache)."""
    cache = _make_cache(8, 8, torch.float32)
    q = torch.randn(2, 3 * 8)
    k = torch.randn(2, 3 * 8)
    positions = torch.tensor([0, 7], dtype=torch.int64)
    return positions, q, k, 8, cache


def test_guards_mirror_the_cu_checks():
    op = torch.ops.sgl_kernel.minimax_h3_rotary_embedding
    positions, q, k, hs, cache = _valid_call()

    # is_neox is checked FIRST, even when shapes are also wrong
    with pytest.raises(RuntimeError, match="NeoX layout"):
        op(positions, q.unsqueeze(0), None, hs, cache, False)

    with pytest.raises(RuntimeError, match="packed GPU Q/K"):
        op(positions, q.unsqueeze(0), None, hs, cache, True)  # 3-D query
    q_strided = torch.randn(2, 3 * hs, 2)[:, :, 0]  # stride(1) == 2
    with pytest.raises(RuntimeError, match="packed GPU Q/K"):
        op(positions, q_strided, None, hs, cache, True)

    with pytest.raises(RuntimeError, match="position per token"):
        op(positions[:-1], q.clone(), None, hs, cache, True)  # numel != T
    with pytest.raises(RuntimeError, match="position per token"):
        op(positions.to(torch.int32), q.clone(), None, hs, cache, True)
    with pytest.raises(RuntimeError, match="position per token"):
        op(positions.unsqueeze(1), q.clone(), None, hs, cache, True)  # 2-D
    with pytest.raises(RuntimeError, match="position per token"):
        nc_pos = torch.arange(0, 4, dtype=torch.int64)[::2]  # non-contig
        op(nc_pos, q.clone(), None, hs, cache, True)

    with pytest.raises(RuntimeError, match="head size"):
        op(positions, q.clone(), None, hs + 1, cache, True)  # width % hs

    with pytest.raises(RuntimeError, match="device and dtype"):
        op(positions, q.clone().to(torch.float16), None, hs, cache, True)
    nc_cache = torch.randn(8, 16)[:, :8]  # same shape, row pitch 16
    with pytest.raises(RuntimeError, match="device and dtype"):
        op(positions, q.clone(), None, hs, nc_cache, True)

    with pytest.raises(RuntimeError, match="rotary dimension"):
        op(positions, q.clone(), None, hs, torch.zeros(8, 7), True)  # odd
    with pytest.raises(RuntimeError, match="rotary dimension"):
        wide = torch.zeros(8, 2 * hs + 2)  # even but > head_size
        op(positions, q.clone(), None, hs, wide, True)

    with pytest.raises(RuntimeError, match="matching packed Q/K"):
        op(positions, q.clone(), k.clone()[:, :8], hs, cache, True)
    k_strided = torch.randn(2, 3 * hs, 2)[:, :, 0]  # sizes ok, stride(1)==2
    with pytest.raises(RuntimeError, match="matching packed Q/K"):
        op(positions, q.clone(), k_strided, hs, cache, True)

    # dispatch set {f32, f16, bf16} (vendored utils.h DISPATCH_FLOAT_TYPES)
    with pytest.raises(RuntimeError, match="not implemented"):
        op(positions, q.clone().to(torch.float64), None, hs,
           cache.to(torch.float64), True)

    # control: the fixture itself is valid
    op(positions, q, k, hs, cache, True)
    assert not torch.isnan(q).any()


# ------------------------------------------------ gpu face (torch tier) --

@requires_gpu
def test_gpu_probe_and_community_call_shape():
    # Exactly what the DiT does on HCU: getattr probe, then
    # (positions, q.view(T, -1), k.view(T, -1), q.shape[-1], cache, True).
    rotary = getattr(torch.ops.sgl_kernel, "minimax_h3_rotary_embedding", None)
    assert rotary is not None
    q3 = torch.randn(2, 3, 8, device="cuda").to(torch.float16)
    k3 = torch.randn(2, 3, 8, device="cuda").to(torch.float16)
    cache = _make_cache(8, 8, torch.float16).cuda()
    positions = torch.tensor([0, 3], device="cuda", dtype=torch.int64)
    rotary(positions, q3.view(q3.shape[0], -1), k3.view(k3.shape[0], -1),
           q3.shape[-1], cache, True)
    assert not torch.isnan(q3.float()).any()


@requires_gpu
@pytest.mark.parametrize("dtype", DTYPES)
def test_gpu_bitexact_vs_cpu_eager_chain(dtype):
    # The torch tier must stay bit-exact on device tensors: rounded_product
    # exists precisely to stop the gfx938 compiler from contracting the
    # first product into the add; eager device elementwise ops carry no
    # contraction, and their bits are device-independent.
    torch.manual_seed(3)
    T, H, hs, rd = 13, 5, 64, 32
    cache_cpu = _make_cache(64, rd, dtype)
    q = torch.randn(T, H * hs, device="cuda").to(dtype)
    k = torch.randn(T, H * hs, device="cuda").to(dtype)
    positions = torch.randint(0, 64, (T,), device="cuda", dtype=torch.int64)
    q_ref, k_ref = _eager_chain_ref(
        q.clone().cpu(), k.clone().cpu(), positions.cpu(), hs, cache_cpu)
    torch.ops.sgl_kernel.minimax_h3_rotary_embedding(
        positions, q, k, hs, cache_cpu.cuda(), True)
    assert torch.equal(q, q_ref.cuda())
    assert torch.equal(k, k_ref.cuda())

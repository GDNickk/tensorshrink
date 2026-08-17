"""
tensorshrink.triton_kernels
============================

Fused CUDA kernels (via Triton) for QuantLinear.forward(). The pure-PyTorch
dequant path in modules.py runs unpack -> cast -> multiply -> add -> reshape
as separate kernel launches over a dense-weight-sized buffer before the
matmul even starts. These kernels fuse the group-affine reconstruction
directly into the GEMM itself (x @ W^T), so the dense fp16 weight is never
materialized in global memory -- only per-tile fragments live in registers.

Three GEMM kernels, one per packed bit-width (2/4/8-bit), all
@triton.autotune-d over tile width / warp count / pipelining depth (first
call per shape pays a one-time benchmark, cached after that). Each kernel
processes one group (GROUP_SIZE columns) per loop iteration, so the group
width is a runtime kwarg rather than a compile-time tile dimension --
group_size in {64, 128} (see _SUPPORTED_GROUP_SIZES) both work with no
extra kernel specialization. fused_quantized_linear is the dispatch point
and returns None for any shape/dtype/group_size it doesn't cover, so the
caller falls back to the pure-PyTorch path.

Compute dtype: each kernel takes COMPUTE_DTYPE (fp16 or bf16); the
accumulator stays fp32. bf16 needs Ampere+ (checked on the host in
fused_quantized_linear).

Fused outlier correction: instead of a separate atomic-scatter kernel pass,
each GEMM tile adds the outliers landing in its own column range as a small
matmul against a 0/1 selection matrix -- no atomics, no second output pass.
This relies on QuantLinear's outlier tables being sorted by row (see
modules.py), so each tile can binary-search its own slice instead of
scanning the whole table. Fusing wins for small outlier tables/row tiles and
loses for large ones (see MAX_FUSED_OUTLIERS); the original standalone
atomic-add kernel (_outlier_correction_kernel) is kept for that regime and
for the public apply_outlier_correction API.

Note for very VRAM-constrained streaming: the first call per distinct layer
shape compiles and benchmarks every _AUTOTUNE_CONFIGS entry, which
transiently uses more memory than steady state. Run one warmup forward pass
per shape before entering a tight-budget loop to avoid paying that during
the real run.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only where triton is absent
    TRITON_AVAILABLE = False

SUPPORTED_GROUP_SIZE = 128  # legacy alias, still exported
_SUPPORTED_GROUP_SIZES = {64, 128}

# Bit-widths with a fused GEMM kernel below. Anything else (e.g. the codec's
# 6-bit RVQ mode) is not an error — `fused_quantized_linear` returns None and
# the caller uses the pure-PyTorch reconstruction path.
SUPPORTED_BITS = (2, 4, 8)

# Guards on folding the outlier correction into the GEMM vs. running the
# standalone atomic pass — pure performance knobs, numerically equivalent.
# Fusion wins in the decode-shaped corner (small batch, moderate outlier
# table) this kernel set targets, and loses once either M or the outlier
# table gets large (measured on an RTX 4060, 4-bit fp16). MAX_FUSED_BLOCK_M
# lines up with `_select_block_m`'s first tier.
MAX_FUSED_OUTLIERS = 10000
MAX_FUSED_BLOCK_M = 16


if TRITON_AVAILABLE:

    # torch dtype -> Triton tile dtype for the `COMPUTE_DTYPE` constexpr.
    # fp32 is deliberately absent: `force_fp32_compute` layers take the
    # pure-PyTorch path (see `QuantLinear.forward`), and an fp32 `tl.dot`
    # here would be both slower and pointless given the accumulator is
    # already fp32.
    _TL_COMPUTE_DTYPE = {
        torch.float16: tl.float16,
        torch.bfloat16: tl.bfloat16,
    }

    # `NUM_STAGES` feeds `tl.range(..., num_stages=NUM_STAGES)` for the group
    # loop below (explicit software pipelining), separate from Triton's own
    # `num_stages=` config field which targets a different pipelining path.
    #
    # Curated subset (not a full cross product) to keep autotune's one-time
    # per-shape warmup cheap. Covers decode-shaped calls (small M) and wide
    # FLUX-scale layers, where a bigger BLOCK_N or deeper pipelining helps.
    #
    # IMPORTANT, found the hard way on a companded (HAS_CODEBOOK=True, see
    # codec.py's Lloyd-Max quantizer) SDXL UNet run: the docstring claim that
    # "configs that don't fit are pruned automatically" does NOT hold here.
    # Triton's autotune pruning estimates shared-memory footprint from its
    # own `num_stages=` config field; the `NUM_STAGES` above is a *manual*
    # meta-parameter driving this kernel's own hand-rolled software
    # pipelining (`tl.range(..., num_stages=NUM_STAGES)`), which Triton's
    # pruning heuristic has no visibility into at all. So a config whose
    # real (manual-pipelining) shared-memory need exceeds the card's limit
    # is never scored +inf and skipped — it's launched, and
    # `_init_handles()` raises `OutOfResources` at that point instead of
    # during the safe autotune search. Confirmed on an RTX 4060 (99KB
    # shared memory/SM): companded's extra codebook-gather load pushes the
    # widest tiers (BLOCK_N=128 at NUM_STAGES>=3, and both BLOCK_N=256
    # tiers) from "fits" (plain kernel) to "224KB required, 99KB available"
    # (companded kernel) at a mid-size M (1152) that's common in a UNet's
    # attention blocks — this is what produced a hard crash in isolation,
    # and something far worse (~30s/step, VRAM ballooning past the physical
    # 8GB card into Windows' WDDM shared-memory fallback) inside a full
    # diffusers pipeline call, where the exception was somehow being
    # swallowed and retried rather than propagating. Trimmed to the tiers
    # verified safe for the companded kernel too (see
    # `fused_quantized_linear`'s `OutOfResources` catch below for the
    # second, defense-in-depth layer). This costs plain (non-companded)
    # quantization a small amount of peak throughput on the very widest
    # FLUX-scale layers, which the removed tiers targeted — worth it for
    # not being able to blow up the same way on a card/shape/kernel
    # combination this session didn't happen to test.
    _AUTOTUNE_CONFIGS = [
        # -- narrow tiles: decode-shaped calls (M=1-16), latency-bound ------
        triton.Config({"BLOCK_N": 32, "NUM_STAGES": 1}, num_warps=2),
        triton.Config({"BLOCK_N": 32, "NUM_STAGES": 2}, num_warps=4),
        triton.Config({"BLOCK_N": 32, "NUM_STAGES": 3}, num_warps=8),
        triton.Config({"BLOCK_N": 32, "NUM_STAGES": 4}, num_warps=2),
        triton.Config({"BLOCK_N": 64, "NUM_STAGES": 4}, num_warps=2),
        # -- mid tiles (widest tier verified safe for the companded kernel) -
        triton.Config({"BLOCK_N": 64, "NUM_STAGES": 1}, num_warps=4),
        triton.Config({"BLOCK_N": 64, "NUM_STAGES": 2}, num_warps=8),
        triton.Config({"BLOCK_N": 64, "NUM_STAGES": 3}, num_warps=4),
        triton.Config({"BLOCK_N": 128, "NUM_STAGES": 1}, num_warps=4),
        triton.Config({"BLOCK_N": 128, "NUM_STAGES": 2}, num_warps=8),
    ]

    # `BLOCK_M` is host-selected from M (see `_select_block_m`) rather than
    # autotuned, but it *is* part of the autotune key: the best
    # BLOCK_N/warps/stages for a 16-row tile is not generally the best for a
    # 64-row one, and keying on it lets each tier tune independently instead
    # of inheriting whichever tier happened to be benchmarked first.
    # `COMPUTE_DTYPE` is in the key for the same reason (bf16 and fp16 have
    # different `tl.dot` shapes/throughput on the same hardware).
    _AUTOTUNE_KEY = ["BLOCK_M", "COMPUTE_DTYPE", "N", "COLS", "N_GROUPS"]

    # Outliers are processed `BLOCK_O` at a time inside the GEMM kernels.
    # 16 is the minimum K extent `tl.dot` accepts; at the default 0.3%
    # `outlier_frac` a single (BLOCK_M, BLOCK_N=64) tile of a 4096-wide
    # layer owns ~50 outliers, so a small chunk keeps the tail waste low.
    _BLOCK_O = 16

    @triton.jit
    def _lower_bound(ptr, n, value):
        """Index of the first entry of the ascending-sorted array `ptr[:n]`
        that is `>= value` (i.e. C++'s `std::lower_bound`). Scalar binary
        search, ~log2(n) dependent loads — used by the GEMM kernels to find
        the slice of the row-sorted outlier table belonging to their own
        column tile, instead of every program scanning the whole table.

        `n - n` rather than a literal `0` for the initial `lo` so the
        loop-carried variables start out as runtime scalars of the same
        dtype as `n`, instead of Triton having to promote a constexpr int
        into the loop.
        """
        lo = n - n
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            v = tl.load(ptr + mid)
            if v < value:
                lo = mid + 1
            else:
                hi = mid
        return lo

    @triton.jit
    def _add_outlier_tile(
        acc, x_ptr, offs_m, mask_m, offs_n, n_lo, n_hi,
        row_ptr, col_ptr, delta_ptr, N_OUTLIERS,
        stride_xm, stride_xk,
        BLOCK_O: tl.constexpr, OUTLIERS_SORTED: tl.constexpr,
    ):
        """Add this program's share of the sparse outlier correction
        directly into the GEMM accumulator `acc` (shape `(BLOCK_M,
        BLOCK_N)`, fp32), returning the updated accumulator.

        Outlier k replaces a clipped weight at (weight row `row[k]`, input
        column `col[k]`) with its true value, which changes output element
        `[m, row[k]]` by `x[m, col[k]] * delta[k]`. Only outliers whose
        `row[k]` lands in this tile's column range `[n_lo, n_hi)` matter.
        The per-k scatter into the n axis is done as a 0/1 selection matrix
        fed to `tl.dot`, which keeps everything in registers — the
        alternative (`tl.atomic_add` into global memory, as the standalone
        kernel does) would force the accumulator out to memory before the
        bias add and the final store.
        """
        if OUTLIERS_SORTED:
            o_pos = _lower_bound(row_ptr, N_OUTLIERS, n_lo)
            o_end = _lower_bound(row_ptr, N_OUTLIERS, n_hi)
        else:
            o_pos = N_OUTLIERS - N_OUTLIERS
            o_end = N_OUTLIERS

        # `row_ptr`/`col_ptr` are int32 (modules.py stores outlier_row/col
        # at int32, halving their resident VRAM vs the previous int64 —
        # bounded by out_features/cols, always well within int32 range for
        # any quantized Linear layer). `offs_n` is already int32-native
        # here (from tl.arange/pid_n*BLOCK_N), so no upcast needed to match
        # -- comparing at native int32 width is if anything slightly
        # cheaper than the previous int64 comparison.
        while o_pos < o_end:
            offs_o = o_pos + tl.arange(0, BLOCK_O)
            mask_o = offs_o < o_end
            # other=-1 for the row index: a masked-off lane must never
            # compare equal to a real output column (all >= 0), so its
            # selection row is all-zero and it contributes nothing.
            o_row = tl.load(row_ptr + offs_o, mask=mask_o, other=-1)
            o_col = tl.load(col_ptr + offs_o, mask=mask_o, other=0)
            o_delta = tl.load(delta_ptr + offs_o, mask=mask_o, other=0.0).to(tl.float32)

            x_gather = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + o_col[None, :] * stride_xk,
                mask=mask_m[:, None] & mask_o[None, :], other=0.0,
            ).to(tl.float32)

            xv = x_gather * o_delta[None, :]                      # (BLOCK_M, BLOCK_O)
            sel = (o_row[:, None] == offs_n[None, :]).to(tl.float32)  # (BLOCK_O, BLOCK_N)
            acc += tl.dot(xv, sel, allow_tf32=False)
            o_pos += BLOCK_O
        return acc

    @triton.jit
    def _load_scale_gmin(
        scale_ptr, gmin_ptr, offs_n, mask_n, g, stride_sn, stride_sg,
        dq_scale_codes_ptr, dq_scale_meta_ptr, dq_gmin_codes_ptr, dq_gmin_meta_ptr,
        N_GROUPS, DQ_GROUP_SIZE,
        HAS_DOUBLE_QUANT: tl.constexpr, COMPUTE_DTYPE: tl.constexpr,
    ):
        """This group's per-row `(scale, gmin)`, shape `(BLOCK_N,)` each, cast
        to `COMPUTE_DTYPE`. Shared by all three bit-width kernels.

        Plain (`HAS_DOUBLE_QUANT=False`): one direct fp16 load each from the
        full `[rows, n_groups]` scale/gmin buffers.

        Double-quantized (see codec.py's `_double_quantize_np` and
        `QuantLinear._get_scale_gmin`): scale/gmin are themselves stored as
        a uint8 code plus one shared (meta_scale, meta_min) fp16 pair per
        `DQ_GROUP_SIZE`-element super-group, flattened row-major over
        `[rows, n_groups]` (so `flat_idx = row * N_GROUPS + g`). Reconstruct
        as `code * meta_scale + meta_min` — one extra uint8 load (cheaper
        than the fp16 load it replaces) plus a small per-tile gather into
        the meta table, indexed by `flat_idx // DQ_GROUP_SIZE` (which varies
        per row within this BLOCK_N tile whenever N_GROUPS doesn't evenly
        divide DQ_GROUP_SIZE, hence the gather rather than a shared scalar
        load) — the same gather pattern as the `HAS_CODEBOOK` lookup above,
        just keyed by a computed index instead of the raw code.
        """
        if HAS_DOUBLE_QUANT:
            flat_idx = offs_n * N_GROUPS + g
            super_idx = flat_idx // DQ_GROUP_SIZE
            sc_code = tl.load(dq_scale_codes_ptr + flat_idx, mask=mask_n, other=0).to(tl.float32)
            sc_ms = tl.load(dq_scale_meta_ptr + super_idx * 2 + 0, mask=mask_n, other=1.0).to(tl.float32)
            sc_mm = tl.load(dq_scale_meta_ptr + super_idx * 2 + 1, mask=mask_n, other=0.0).to(tl.float32)
            sc = (sc_code * sc_ms + sc_mm).to(COMPUTE_DTYPE)
            gm_code = tl.load(dq_gmin_codes_ptr + flat_idx, mask=mask_n, other=0).to(tl.float32)
            gm_ms = tl.load(dq_gmin_meta_ptr + super_idx * 2 + 0, mask=mask_n, other=1.0).to(tl.float32)
            gm_mm = tl.load(dq_gmin_meta_ptr + super_idx * 2 + 1, mask=mask_n, other=0.0).to(tl.float32)
            gm = (gm_code * gm_ms + gm_mm).to(COMPUTE_DTYPE)
        else:
            s_ptrs = scale_ptr + offs_n * stride_sn + g * stride_sg
            m_ptrs = gmin_ptr + offs_n * stride_sn + g * stride_sg
            sc = tl.load(s_ptrs, mask=mask_n, other=0.0).to(COMPUTE_DTYPE)
            gm = tl.load(m_ptrs, mask=mask_n, other=0.0).to(COMPUTE_DTYPE)
        return sc, gm

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY)
    @triton.jit
    def _fused_dequant_matmul_2bit(
        x_ptr, packed_ptr, scale_ptr, gmin_ptr, bias_ptr, out_ptr,
        orow_ptr, ocol_ptr, odelta_ptr, codebook_ptr,
        dq_scale_codes_ptr, dq_scale_meta_ptr, dq_gmin_codes_ptr, dq_gmin_meta_ptr,
        M, N, COLS, N_GROUPS, N_OUTLIERS, DQ_GROUP_SIZE,
        stride_xm, stride_xk,
        stride_pn, stride_pk,
        stride_sn, stride_sg,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr, HAS_OUTLIERS: tl.constexpr,
        OUTLIERS_SORTED: tl.constexpr,
        HAS_CODEBOOK: tl.constexpr,
        HAS_DOUBLE_QUANT: tl.constexpr,
        COMPUTE_DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr, QUARTER: tl.constexpr,
        BLOCK_O: tl.constexpr,
        NUM_STAGES: tl.constexpr,
    ):
        """2-bit (crumb-packed) variant: one byte holds four codes, bits
        0-1 / 2-3 / 4-5 / 6-7 in ascending column order — the exact layout
        `codec._pack_crumbs` writes. Four `tl.dot`s per group (one per
        crumb position) against the four interleaved column strides, same
        per-group affine dequant `w = code_val * scale + gmin` as every
        other width, where `code_val` is either the raw code (uniform grid)
        or `codebook[code]` (companded / Lloyd-Max grid, see codec.py) —
        `codebook_ptr` is a tiny (<=4-entry, for 2-bit) fp32 lookup table,
        cheap enough to stay hot in cache across the whole kernel."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        q_offs = tl.arange(0, QUARTER)

        for g in tl.range(0, N_GROUPS, num_stages=NUM_STAGES):
            k0 = g * GROUP_SIZE
            byte_offs = k0 // 4 + q_offs
            p_ptrs = packed_ptr + offs_n[:, None] * stride_pn + byte_offs[None, :] * stride_pk
            b = tl.load(p_ptrs, mask=mask_n[:, None], other=0).to(tl.uint8)
            i0 = (b & 0x03).to(tl.int32)
            i1 = ((b >> 2) & 0x03).to(tl.int32)
            i2 = ((b >> 4) & 0x03).to(tl.int32)
            i3 = ((b >> 6) & 0x03).to(tl.int32)
            if HAS_CODEBOOK:
                c0 = tl.load(codebook_ptr + i0).to(COMPUTE_DTYPE)
                c1 = tl.load(codebook_ptr + i1).to(COMPUTE_DTYPE)
                c2 = tl.load(codebook_ptr + i2).to(COMPUTE_DTYPE)
                c3 = tl.load(codebook_ptr + i3).to(COMPUTE_DTYPE)
            else:
                c0 = i0.to(COMPUTE_DTYPE)
                c1 = i1.to(COMPUTE_DTYPE)
                c2 = i2.to(COMPUTE_DTYPE)
                c3 = i3.to(COMPUTE_DTYPE)

            sc, gm = _load_scale_gmin(
                scale_ptr, gmin_ptr, offs_n, mask_n, g, stride_sn, stride_sg,
                dq_scale_codes_ptr, dq_scale_meta_ptr, dq_gmin_codes_ptr, dq_gmin_meta_ptr,
                N_GROUPS, DQ_GROUP_SIZE,
                HAS_DOUBLE_QUANT=HAS_DOUBLE_QUANT, COMPUTE_DTYPE=COMPUTE_DTYPE,
            )

            w0 = c0 * sc[:, None] + gm[:, None]
            w1 = c1 * sc[:, None] + gm[:, None]
            w2 = c2 * sc[:, None] + gm[:, None]
            w3 = c3 * sc[:, None] + gm[:, None]

            col0 = k0 + 4 * q_offs
            col1 = col0 + 1
            col2 = col0 + 2
            col3 = col0 + 3

            xb = x_ptr + offs_m[:, None] * stride_xm
            x0 = tl.load(xb + col0[None, :] * stride_xk,
                         mask=mask_m[:, None] & (col0 < COLS)[None, :], other=0.0).to(COMPUTE_DTYPE)
            x1 = tl.load(xb + col1[None, :] * stride_xk,
                         mask=mask_m[:, None] & (col1 < COLS)[None, :], other=0.0).to(COMPUTE_DTYPE)
            x2 = tl.load(xb + col2[None, :] * stride_xk,
                         mask=mask_m[:, None] & (col2 < COLS)[None, :], other=0.0).to(COMPUTE_DTYPE)
            x3 = tl.load(xb + col3[None, :] * stride_xk,
                         mask=mask_m[:, None] & (col3 < COLS)[None, :], other=0.0).to(COMPUTE_DTYPE)

            acc += tl.dot(x0, tl.trans(w0), allow_tf32=False)
            acc += tl.dot(x1, tl.trans(w1), allow_tf32=False)
            acc += tl.dot(x2, tl.trans(w2), allow_tf32=False)
            acc += tl.dot(x3, tl.trans(w3), allow_tf32=False)

        if HAS_OUTLIERS:
            acc = _add_outlier_tile(
                acc, x_ptr, offs_m, mask_m, offs_n,
                pid_n * BLOCK_N, pid_n * BLOCK_N + BLOCK_N,
                orow_ptr, ocol_ptr, odelta_ptr, N_OUTLIERS,
                stride_xm, stride_xk,
                BLOCK_O=BLOCK_O, OUTLIERS_SORTED=OUTLIERS_SORTED,
            )

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

        o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY)
    @triton.jit
    def _fused_dequant_matmul_4bit(
        x_ptr, packed_ptr, scale_ptr, gmin_ptr, bias_ptr, out_ptr,
        orow_ptr, ocol_ptr, odelta_ptr, codebook_ptr,
        dq_scale_codes_ptr, dq_scale_meta_ptr, dq_gmin_codes_ptr, dq_gmin_meta_ptr,
        M, N, COLS, N_GROUPS, N_OUTLIERS, DQ_GROUP_SIZE,
        stride_xm, stride_xk,
        stride_pn, stride_pk,
        stride_sn, stride_sg,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr, HAS_OUTLIERS: tl.constexpr,
        OUTLIERS_SORTED: tl.constexpr,
        HAS_CODEBOOK: tl.constexpr,
        HAS_DOUBLE_QUANT: tl.constexpr,
        COMPUTE_DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr, HALF: tl.constexpr,
        BLOCK_O: tl.constexpr,
        NUM_STAGES: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        half_offs = tl.arange(0, HALF)

        for g in tl.range(0, N_GROUPS, num_stages=NUM_STAGES):
            k0 = g * GROUP_SIZE
            byte_offs = k0 // 2 + half_offs
            p_ptrs = packed_ptr + offs_n[:, None] * stride_pn + byte_offs[None, :] * stride_pk
            b = tl.load(p_ptrs, mask=mask_n[:, None], other=0).to(tl.uint8)
            i_low = (b & 0x0F).to(tl.int32)
            i_high = ((b >> 4) & 0x0F).to(tl.int32)
            if HAS_CODEBOOK:
                low = tl.load(codebook_ptr + i_low).to(COMPUTE_DTYPE)
                high = tl.load(codebook_ptr + i_high).to(COMPUTE_DTYPE)
            else:
                low = i_low.to(COMPUTE_DTYPE)
                high = i_high.to(COMPUTE_DTYPE)

            sc, gm = _load_scale_gmin(
                scale_ptr, gmin_ptr, offs_n, mask_n, g, stride_sn, stride_sg,
                dq_scale_codes_ptr, dq_scale_meta_ptr, dq_gmin_codes_ptr, dq_gmin_meta_ptr,
                N_GROUPS, DQ_GROUP_SIZE,
                HAS_DOUBLE_QUANT=HAS_DOUBLE_QUANT, COMPUTE_DTYPE=COMPUTE_DTYPE,
            )

            w_low = low * sc[:, None] + gm[:, None]
            w_high = high * sc[:, None] + gm[:, None]

            col_even = k0 + 2 * half_offs
            col_odd = col_even + 1
            mask_even = col_even < COLS
            mask_odd = col_odd < COLS

            xe_ptrs = x_ptr + offs_m[:, None] * stride_xm + col_even[None, :] * stride_xk
            xo_ptrs = x_ptr + offs_m[:, None] * stride_xm + col_odd[None, :] * stride_xk
            x_even = tl.load(xe_ptrs, mask=mask_m[:, None] & mask_even[None, :], other=0.0).to(COMPUTE_DTYPE)
            x_odd = tl.load(xo_ptrs, mask=mask_m[:, None] & mask_odd[None, :], other=0.0).to(COMPUTE_DTYPE)

            acc += tl.dot(x_even, tl.trans(w_low), allow_tf32=False)
            acc += tl.dot(x_odd, tl.trans(w_high), allow_tf32=False)

        if HAS_OUTLIERS:
            acc = _add_outlier_tile(
                acc, x_ptr, offs_m, mask_m, offs_n,
                pid_n * BLOCK_N, pid_n * BLOCK_N + BLOCK_N,
                orow_ptr, ocol_ptr, odelta_ptr, N_OUTLIERS,
                stride_xm, stride_xk,
                BLOCK_O=BLOCK_O, OUTLIERS_SORTED=OUTLIERS_SORTED,
            )

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

        o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.autotune(configs=_AUTOTUNE_CONFIGS, key=_AUTOTUNE_KEY)
    @triton.jit
    def _fused_dequant_matmul_8bit(
        x_ptr, packed_ptr, scale_ptr, gmin_ptr, bias_ptr, out_ptr,
        orow_ptr, ocol_ptr, odelta_ptr, codebook_ptr,
        dq_scale_codes_ptr, dq_scale_meta_ptr, dq_gmin_codes_ptr, dq_gmin_meta_ptr,
        M, N, COLS, N_GROUPS, N_OUTLIERS, DQ_GROUP_SIZE,
        stride_xm, stride_xk,
        stride_pn, stride_pk,
        stride_sn, stride_sg,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr, HAS_OUTLIERS: tl.constexpr,
        OUTLIERS_SORTED: tl.constexpr,
        HAS_CODEBOOK: tl.constexpr,
        HAS_DOUBLE_QUANT: tl.constexpr,
        COMPUTE_DTYPE: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_O: tl.constexpr,
        NUM_STAGES: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        k_offs = tl.arange(0, GROUP_SIZE)

        for g in tl.range(0, N_GROUPS, num_stages=NUM_STAGES):
            k0 = g * GROUP_SIZE
            col = k0 + k_offs
            mask_k = col < COLS

            p_ptrs = packed_ptr + offs_n[:, None] * stride_pn + col[None, :] * stride_pk
            code_idx = tl.load(p_ptrs, mask=mask_n[:, None], other=0).to(tl.int32)
            if HAS_CODEBOOK:
                codes = tl.load(codebook_ptr + code_idx).to(COMPUTE_DTYPE)
            else:
                codes = code_idx.to(COMPUTE_DTYPE)

            sc, gm = _load_scale_gmin(
                scale_ptr, gmin_ptr, offs_n, mask_n, g, stride_sn, stride_sg,
                dq_scale_codes_ptr, dq_scale_meta_ptr, dq_gmin_codes_ptr, dq_gmin_meta_ptr,
                N_GROUPS, DQ_GROUP_SIZE,
                HAS_DOUBLE_QUANT=HAS_DOUBLE_QUANT, COMPUTE_DTYPE=COMPUTE_DTYPE,
            )

            w = codes * sc[:, None] + gm[:, None]

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + col[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(COMPUTE_DTYPE)

            acc += tl.dot(x_tile, tl.trans(w), allow_tf32=False)

        if HAS_OUTLIERS:
            acc = _add_outlier_tile(
                acc, x_ptr, offs_m, mask_m, offs_n,
                pid_n * BLOCK_N, pid_n * BLOCK_N + BLOCK_N,
                orow_ptr, ocol_ptr, odelta_ptr, N_OUTLIERS,
                stride_xm, stride_xk,
                BLOCK_O=BLOCK_O, OUTLIERS_SORTED=OUTLIERS_SORTED,
            )

        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc += bias[None, :]

        o_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        tl.store(o_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _outlier_correction_kernel(
        x_ptr, row_ptr, col_ptr, delta_ptr, out_ptr,
        N_OUTLIERS,
        stride_xm, stride_xk, stride_om, stride_on,
        BLOCK: tl.constexpr,
    ):
        """One atomic-add pass replacing PyTorch's `index_add_` for the
        sparse outlier correction. Grid is (n_outlier_blocks, M) — program
        `(pid, m)` handles outliers `[pid*BLOCK, (pid+1)*BLOCK)` for batch
        row `m`. `tl.atomic_add` handles the case where several outliers in
        this block land in the same output column (row-major within the
        same output row) without a separate reduction step.

        Still the right tool for two cases even though the GEMM kernels can
        now fold the correction in themselves (see `_add_outlier_tile`):
        very large outlier tables (`MAX_FUSED_OUTLIERS`), where the fused
        form's per-tile redundancy outweighs one extra streaming pass, and
        the public `apply_outlier_correction` entry point, which exists for
        callers that already have an uncorrected GEMM result in hand."""
        pid = tl.program_id(0)
        m = tl.program_id(1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N_OUTLIERS

        row = tl.load(row_ptr + offs, mask=mask, other=0)
        col = tl.load(col_ptr + offs, mask=mask, other=0)
        delta = tl.load(delta_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        x_val = tl.load(x_ptr + m * stride_xm + col * stride_xk, mask=mask, other=0.0).to(tl.float32)
        contrib = x_val * delta

        out_offs = out_ptr + m * stride_om + row * stride_on
        tl.atomic_add(out_offs, contrib, mask=mask)


def _select_block_m(m: int) -> int:
    """BLOCK_M tier for a given batch-row count.

    A wider tile amortizes the packed-weight read (the kernel's real
    bandwidth bottleneck) across more rows, which matters once M grows past
    decode-step size (M=1-8). Coarse tiers rather than `next_power_of_2(M)`
    because BLOCK_M is part of the autotune key — each distinct value pays
    its own one-time benchmark, so a handful of tiers keeps that bounded.

    This kernel never caches a dequantized weight tile across row-tiles: for
    a fixed (pid_n) each of the `cdiv(M, BLOCK_M)` row-tile programs
    independently re-reads and re-dequantizes that N-tile's full K range.
    Capping BLOCK_M at 64 is fine for decode-shaped calls (M<=64 -> exactly
    one row-tile, zero redundancy) but leaves large-M callers -- e.g. image
    model attention, where M is a spatial token count times batch and
    regularly reaches the thousands -- redundantly re-dequantizing the same
    weight bytes dozens of times per call. Measured on a real SDXL UNet
    (RTX 4060): M=1152 at the old BLOCK_M=64 cap redundantly re-dequantized
    every weight tile 18x per call; M=4608 did it 72x. The two tiers below
    keep that redundancy bounded (9x and 18x respectively for those same M)
    without adding tiers so fine-grained that autotune warmup cost balloons.
    """
    if m <= 16:
        return 16
    if m <= 128:
        return 32
    if m <= 512:
        return 64
    if m <= 2048:
        return 128
    return 256


# ---------------------------------------------------------------------------
# shared scratch-output buffer pool
# ---------------------------------------------------------------------------

class _OutputBufferPool:
    """Thread-local pool of fp32 `(M, N)` GEMM scratch buffers, shared by
    every `QuantLinear` on the thread instead of one buffer owned per layer.

    The fp32 buffer a fused kernel writes into is scratch, not the output:
    `QuantLinear.forward()` casts it to the compute dtype and returns that
    copy, so the buffer is dead the instant the cast returns. That means one
    buffer can back every layer, instead of the old scheme's one permanent
    fp32 buffer per layer (which added up to ~1GB+ of idle VRAM on a
    144-layer model).

    Contract: **the returned buffer is only valid until the next
    `fused_quantized_linear` call on the same thread.** Anything that must
    outlive that needs to be a copy. Pass an explicit `out=` (or
    `use_buffer_pool=False`) to opt out.

    Thread-local (not global) because two threads running forward passes
    concurrently (e.g. `streaming.py`'s prefetch thread) must not share
    scratch memory. Entries are keyed by `(device, M, N)` with LRU eviction
    capped at `max_entries`.
    """

    __slots__ = ("_bufs", "_order", "max_entries", "n_hits", "n_allocs")

    def __init__(self, max_entries: int = 4):
        self._bufs: dict = {}
        self._order: list = []  # least-recently-used first
        self.max_entries = max_entries
        self.n_hits = 0
        self.n_allocs = 0

    def get(self, m: int, n: int, device: torch.device) -> torch.Tensor:
        key = (device.type, device.index, m, n)
        buf = self._bufs.get(key)
        if buf is not None:
            self.n_hits += 1
            if self._order[-1] != key:
                self._order.remove(key)
                self._order.append(key)
            return buf
        buf = torch.empty((m, n), device=device, dtype=torch.float32)
        self.n_allocs += 1
        self._bufs[key] = buf
        self._order.append(key)
        while len(self._order) > self.max_entries:
            evicted = self._order.pop(0)
            self._bufs.pop(evicted, None)
        return buf

    def clear(self) -> None:
        self._bufs.clear()
        self._order.clear()

    def stats(self) -> dict:
        return {
            "n_entries": len(self._bufs),
            "n_hits": self.n_hits,
            "n_allocs": self.n_allocs,
            "resident_bytes": sum(b.numel() * b.element_size() for b in self._bufs.values()),
        }


_pool_tls = threading.local()


def get_buffer_pool() -> _OutputBufferPool:
    """This thread's `_OutputBufferPool`, created on first use."""
    pool = getattr(_pool_tls, "pool", None)
    if pool is None:
        pool = _OutputBufferPool()
        _pool_tls.pool = pool
    return pool


def clear_buffer_pool() -> None:
    """Drop this thread's pooled scratch buffers (frees their VRAM). Safe to
    call at any point where no fused result is still un-consumed — e.g.
    between a prefill and a decode loop, or before a memory measurement."""
    pool = getattr(_pool_tls, "pool", None)
    if pool is not None:
        pool.clear()


def buffer_pool_stats() -> dict:
    """Hit/allocation counters and resident bytes for this thread's pool."""
    return get_buffer_pool().stats()


def _is_bf16_supported(device: torch.device) -> bool:
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - very old torch without the helper
        try:
            return torch.cuda.get_device_capability(device)[0] >= 8
        except Exception:
            return False


def fused_quantized_linear(
    x: torch.Tensor,
    packed: torch.Tensor,
    scale: Optional[torch.Tensor],
    gmin: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    bits: int,
    group_size: int,
    cols: int,
    out_features: int,
    compute_dtype: torch.dtype,
    return_fp32: bool = False,
    out: Optional[torch.Tensor] = None,
    outlier_row: Optional[torch.Tensor] = None,
    outlier_col: Optional[torch.Tensor] = None,
    outlier_delta: Optional[torch.Tensor] = None,
    outliers_sorted: bool = True,
    use_buffer_pool: bool = True,
    codebook: Optional[torch.Tensor] = None,
    dq_scale_codes: Optional[torch.Tensor] = None,
    dq_scale_meta: Optional[torch.Tensor] = None,
    dq_gmin_codes: Optional[torch.Tensor] = None,
    dq_gmin_meta: Optional[torch.Tensor] = None,
    dq_group_size: int = 0,
    n_groups: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Fused unpack + per-group affine dequant + GEMM (+ optional outlier
    correction). Returns None when this isn't a shape/environment the kernel
    set covers (no Triton, no CUDA input, unsupported group_size/bit-width/
    dtype, or bf16 on a pre-Ampere card) — the caller falls back to the
    pure-PyTorch path.

    `compute_dtype` (fp16 or bf16) sets the kernel's tile dtype; the GEMM
    always accumulates in fp32. Result is cast to `compute_dtype` before
    returning unless `return_fp32=True` (useful if the caller wants to do
    its own fp32 outlier correction afterward, e.g. via
    `apply_outlier_correction`).

    Pass `outlier_row`/`outlier_col`/`outlier_delta` (QuantLinear's
    same-named buffers) to apply the outlier correction here; the returned
    tensor is then always fully corrected, either folded into the GEMM
    (small tables) or via the standalone `_outlier_correction_kernel` pass
    (see `MAX_FUSED_OUTLIERS`). `outliers_sorted=True` (default) lets the
    kernel binary-search instead of scanning the whole table — requires the
    tables sorted by `outlier_row`, which `QuantLinear.__init__` guarantees.

    `codebook`, if given (a small fp32 `[1 << bits]` tensor — see
    `codec.py`'s Lloyd-Max companded-quantization helpers), makes every code
    extracted from `packed` index into it (`w = codebook[code] * scale +
    gmin`) instead of being used directly as the reconstruction value
    (`w = code * scale + gmin`). The gather is a handful of extra bytes read
    from a table small enough to stay cache-resident for the whole kernel,
    so this is the fast path for companded weights too — previously they
    fell back to the uncached pure-PyTorch dequant, ~3x slower per call.

    `packed`/`scale`/`gmin` are not defensively re-`.contiguous()`'d here for
    performance; safe because these are always `QuantLinear`'s own buffers,
    which are built contiguous and never mutated in place.

    Double quantization (see codec.py): pass `scale=None, gmin=None` plus
    `dq_scale_codes`/`dq_scale_meta`/`dq_gmin_codes`/`dq_gmin_meta`/
    `dq_group_size` (`QuantLinear`'s own buffers when `double_quant=True`)
    instead of full `scale`/`gmin` tensors. Each group's scale/gmin is then
    reconstructed in-kernel as `code * meta_scale + meta_min` via one extra
    uint8 load (the code) and a small per-tile gather into the compact
    `meta` table (indexed by `flat_idx // dq_group_size`, mirroring the
    `HAS_CODEBOOK` codebook gather above) instead of one direct fp16 load —
    same fast path, roughly half the resident scale/gmin footprint. `scale`
    and `gmin` must either both be given or both be `None`; same for the
    four `dq_*` arguments (all four or none). When `scale`/`gmin` are
    `None`, `n_groups` must be passed explicitly (it's otherwise inferred
    from `scale.shape[1]`).

    `out`, if given, must be a contiguous fp32 `(M, out_features)` CUDA
    tensor, reused in place. When `out` is None and `use_buffer_pool=True`
    (default), the buffer comes from a shared per-thread `_OutputBufferPool`
    instead of a fresh allocation — that buffer is scratch and only valid
    until the next call on this thread. Pass `use_buffer_pool=False` (or an
    explicit `out=`) for a privately-owned buffer.
    """
    if (
        not TRITON_AVAILABLE
        or not x.is_cuda
        or bits not in SUPPORTED_BITS
        or group_size not in _SUPPORTED_GROUP_SIZES
        or compute_dtype not in _TL_COMPUTE_DTYPE
    ):
        return None
    if compute_dtype is torch.bfloat16 and not _is_bf16_supported(x.device):
        return None

    orig_shape = x.shape
    if x.dim() == 2 and x.is_contiguous():
        x2d = x
    else:
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, K = x2d.shape
    N = out_features
    if n_groups is None:
        if scale is None:
            return None  # neither scale nor an explicit n_groups was given
        n_groups = scale.shape[1]

    if out is not None and (out.shape != (M, N) or out.dtype != torch.float32 or out.device != x.device):
        out = None
    if out is None:
        if use_buffer_pool:
            out = get_buffer_pool().get(M, N, x.device)
        else:
            out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    BLOCK_M = _select_block_m(M)

    n_outliers = 0 if outlier_row is None else int(outlier_row.numel())
    # Fold the outlier correction into the GEMM itself only in the regime
    # where that measured faster (small table AND a narrow row tile — see
    # MAX_FUSED_OUTLIERS); otherwise run the standalone atomic pass after
    # the GEMM. Either way the caller gets a fully corrected result.
    fuse_outliers = 0 < n_outliers <= MAX_FUSED_OUTLIERS and BLOCK_M <= MAX_FUSED_BLOCK_M
    post_correct = n_outliers > 0 and not fuse_outliers

    grid = lambda meta: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, meta["BLOCK_N"]))

    has_bias = bias is not None
    bias_arg = bias if has_bias else out  # dummy pointer, never read when HAS_BIAS=False
    # Same trick for the three outlier pointers: Triton still needs a valid
    # tensor to bind them to even when HAS_OUTLIERS=False compiles their
    # only use away.
    orow = outlier_row if fuse_outliers else out
    ocol = outlier_col if fuse_outliers else out
    odelta = outlier_delta if fuse_outliers else out
    has_codebook = codebook is not None
    # Same dummy-pointer trick as bias_arg/orow/ocol/odelta above: Triton
    # still needs a valid tensor bound to the argument even when
    # HAS_CODEBOOK=False compiles its only use away, and `out` (always a
    # live fp32 CUDA tensor here) keeps the pointer's inferred dtype fp32
    # in both branches so this doesn't force a second kernel specialization
    # to compile for the plain (non-companded) call path.
    codebook_arg = codebook if has_codebook else out

    has_double_quant = dq_scale_codes is not None
    if has_double_quant:
        # Real scale/gmin pointers unused (dead code in the HAS_DOUBLE_QUANT
        # branch) -- `out` (fp32, always live) is the same dummy-pointer
        # convention as bias_arg/orow/etc above. stride_sn/stride_sg are
        # likewise dead in that branch, so their values don't matter.
        scale_arg, gmin_arg = out, out
        stride_sn, stride_sg = 0, 0
        dq_scale_codes_arg, dq_gmin_codes_arg = dq_scale_codes, dq_gmin_codes
        dq_scale_meta_arg, dq_gmin_meta_arg = dq_scale_meta, dq_gmin_meta
    else:
        scale_arg, gmin_arg = scale, gmin
        stride_sn, stride_sg = scale.stride(0), scale.stride(1)
        # Dummy pointers for the dq_* args, dead code when HAS_DOUBLE_QUANT
        # is False. `packed` (uint8) and `scale`/`gmin` (fp16) dtype-match
        # the real dq_*_codes/dq_*_meta args, avoiding an extra Triton
        # specialization for this (common, non-double-quant) call path —
        # same reasoning as codebook_arg's dummy-pointer comment above.
        dq_scale_codes_arg, dq_gmin_codes_arg = packed, packed
        dq_scale_meta_arg, dq_gmin_meta_arg = scale, gmin

    common = dict(
        HAS_BIAS=has_bias,
        HAS_OUTLIERS=fuse_outliers,
        OUTLIERS_SORTED=bool(outliers_sorted),
        HAS_CODEBOOK=has_codebook,
        HAS_DOUBLE_QUANT=has_double_quant,
        COMPUTE_DTYPE=_TL_COMPUTE_DTYPE[compute_dtype],
        BLOCK_M=BLOCK_M,
        GROUP_SIZE=group_size,
        BLOCK_O=_BLOCK_O,
    )
    strides = (
        x2d.stride(0), x2d.stride(1),
        packed.stride(0), packed.stride(1),
        stride_sn, stride_sg,
        out.stride(0), out.stride(1),
    )

    # Defense-in-depth around the kernel launch itself, on top of the config
    # list already trimmed in _AUTOTUNE_CONFIGS: Triton's autotune pruning
    # doesn't see this kernel's *manual* NUM_STAGES pipelining depth (see
    # that list's docstring for the full story — found via a real
    # `OutOfResources: out of resource: shared memory` crash), so a
    # shape/GPU/dtype combination this session didn't happen to hit could
    # still pick a config that overflows shared memory. Any Triton launch
    # failure here degrades to the pure-PyTorch dequant path (this
    # function's contract for "not covered": return None) instead of
    # crashing the whole process or, worse, hanging in a slow
    # repeated-failure loop the way it did before this was caught.
    try:
        if bits == 2:
            _fused_dequant_matmul_2bit[grid](
                x2d, packed, scale_arg, gmin_arg, bias_arg, out,
                orow, ocol, odelta, codebook_arg,
                dq_scale_codes_arg, dq_scale_meta_arg, dq_gmin_codes_arg, dq_gmin_meta_arg,
                M, N, cols, n_groups, n_outliers if fuse_outliers else 0, dq_group_size,
                *strides,
                QUARTER=group_size // 4,
                **common,
            )
        elif bits == 4:
            _fused_dequant_matmul_4bit[grid](
                x2d, packed, scale_arg, gmin_arg, bias_arg, out,
                orow, ocol, odelta, codebook_arg,
                dq_scale_codes_arg, dq_scale_meta_arg, dq_gmin_codes_arg, dq_gmin_meta_arg,
                M, N, cols, n_groups, n_outliers if fuse_outliers else 0, dq_group_size,
                *strides,
                HALF=group_size // 2,
                **common,
            )
        else:
            _fused_dequant_matmul_8bit[grid](
                x2d, packed, scale_arg, gmin_arg, bias_arg, out,
                orow, ocol, odelta, codebook_arg,
                dq_scale_codes_arg, dq_scale_meta_arg, dq_gmin_codes_arg, dq_gmin_meta_arg,
                M, N, cols, n_groups, n_outliers if fuse_outliers else 0, dq_group_size,
                *strides,
                **common,
            )
    except Exception as _e:
        import os as _os
        if _os.environ.get("TS_DEBUG_TRITON_FALLBACK"):
            print(f"[triton fallback] bits={bits} M={M} N={N} has_codebook={has_codebook}: {_e!r}")
        return None

    if post_correct:
        apply_outlier_correction(out, x2d, outlier_row, outlier_col, outlier_delta)

    if not return_fp32:
        out = out.to(compute_dtype)
    return out.reshape(*orig_shape[:-1], N)


def apply_outlier_correction(
    out: torch.Tensor,
    x2d: torch.Tensor,
    outlier_row: torch.Tensor,
    outlier_col: torch.Tensor,
    outlier_delta: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Triton-kernel counterpart to `QuantLinear._apply_outlier_correction`'s
    PyTorch fallback (gather + `index_add_`) — see `_outlier_correction_kernel`'s
    docstring for why this exists. Returns None (caller falls back to the
    PyTorch path) when Triton isn't available or `out` isn't a CUDA tensor;
    a no-op (empty outlier tables) is handled by the caller, not here.

    Still the entry point for callers holding an already-computed,
    uncorrected GEMM result. A caller that hasn't run the GEMM yet should
    instead pass the outlier tables straight to `fused_quantized_linear`,
    which folds the correction into the GEMM kernel and skips this launch
    entirely for all but the largest outlier tables.
    """
    if not TRITON_AVAILABLE or not out.is_cuda:
        return None
    n = outlier_row.numel()
    if n == 0:
        return out

    # `out` is always already a contiguous 2D tensor in practice (it's
    # `fused_quantized_linear`'s own output, reshaped from a 2D buffer via
    # a view) -- skip the reshape/contiguous call in that common case
    # rather than paying its Python-dispatch cost on every layer, every
    # forward call.
    if out.dim() == 2 and out.is_contiguous():
        out2d = out
    else:
        out2d = out.reshape(-1, out.shape[-1]).contiguous()
    M = out2d.shape[0]
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK), M)
    _outlier_correction_kernel[grid](
        x2d, outlier_row, outlier_col, outlier_delta, out2d,
        n,
        x2d.stride(0), x2d.stride(1), out2d.stride(0), out2d.stride(1),
        BLOCK=BLOCK,
    )
    return out2d if out2d is out else out2d.reshape(out.shape)


# ---------------------------------------------------------------------------
# Dequantize-only kernel (no GEMM) for large-M workloads
# ---------------------------------------------------------------------------
# For large batch sizes (UNet attention, long-prompt prefill), the fused
# dequant+GEMM kernel loses to separate dequant + cuBLAS because each row
# tile independently re-reads and re-dequantizes the same packed weight.
# This kernel replaces the 6+ PyTorch kernel launches in _dequantize_rows
# (unpack, cast, reshape, multiply, add, slice) with a single fused pass.

if TRITON_AVAILABLE:

    @triton.jit
    def _dequant_4bit_kernel(
        packed_ptr, scale_ptr, gmin_ptr, out_ptr, codebook_ptr,
        rows, cols, padded_cols, n_groups,
        stride_pn, stride_pk,
        stride_sn, stride_sg,
        stride_on, stride_ok,
        GROUP_SIZE: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
        BLOCK_COL: tl.constexpr,
        HAS_CODEBOOK: tl.constexpr,
    ):
        """Unpack 4-bit packed codes and apply group-affine reconstruction.
        Each program handles a BLOCK_ROW x BLOCK_COL tile of the output.
        When HAS_CODEBOOK is True, codes index into a non-uniform codebook
        (NF4 or Lloyd-Max) before the affine transform."""
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)

        row_start = pid_row * BLOCK_ROW
        col_start = pid_col * BLOCK_COL

        row_offs = row_start + tl.arange(0, BLOCK_ROW)
        col_offs = col_start + tl.arange(0, BLOCK_COL)

        row_mask = row_offs < rows
        col_mask = col_offs < cols

        # Byte index: each byte holds 2 nibbles
        byte_idx = col_offs // 2
        is_high = (col_offs % 2) == 1

        # Load packed bytes
        packed_offsets = row_offs[:, None] * stride_pn + byte_idx[None, :] * stride_pk
        mask_2d = row_mask[:, None] & (byte_idx[None, :] < (padded_cols // 2))
        packed_bytes = tl.load(packed_ptr + packed_offsets, mask=mask_2d, other=0)

        # Unpack nibbles
        low = (packed_bytes & 0x0F)
        high = ((packed_bytes >> 4) & 0x0F)
        code_int = tl.where(is_high[None, :], high, low)

        if HAS_CODEBOOK:
            # Non-uniform codebook lookup (NF4 or Lloyd-Max)
            codes = tl.load(codebook_ptr + code_int.to(tl.int32)).to(tl.float16)
        else:
            codes = code_int.to(tl.float16)

        # Group-affine reconstruction
        group_idx = col_offs // GROUP_SIZE
        scale_offsets = row_offs[:, None] * stride_sn + group_idx[None, :] * stride_sg
        group_mask = row_mask[:, None] & (group_idx[None, :] < n_groups)
        scale_vals = tl.load(scale_ptr + scale_offsets, mask=group_mask, other=1.0)
        gmin_vals = tl.load(gmin_ptr + scale_offsets, mask=group_mask, other=0.0)

        result = codes * scale_vals + gmin_vals

        # Store output
        out_offsets = row_offs[:, None] * stride_on + col_offs[None, :] * stride_ok
        out_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(out_ptr + out_offsets, result, mask=out_mask)

    @triton.jit
    def _dequant_8bit_kernel(
        packed_ptr, scale_ptr, gmin_ptr, out_ptr, codebook_ptr,
        rows, cols, padded_cols, n_groups,
        stride_pn, stride_pk,
        stride_sn, stride_sg,
        stride_on, stride_ok,
        GROUP_SIZE: tl.constexpr,
        BLOCK_ROW: tl.constexpr,
        BLOCK_COL: tl.constexpr,
        HAS_CODEBOOK: tl.constexpr,
    ):
        """Unpack 8-bit codes and apply group-affine reconstruction."""
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)

        row_start = pid_row * BLOCK_ROW
        col_start = pid_col * BLOCK_COL

        row_offs = row_start + tl.arange(0, BLOCK_ROW)
        col_offs = col_start + tl.arange(0, BLOCK_COL)

        row_mask = row_offs < rows
        col_mask = col_offs < cols

        packed_offsets = row_offs[:, None] * stride_pn + col_offs[None, :] * stride_pk
        mask_2d = row_mask[:, None] & (col_offs[None, :] < padded_cols)
        raw_codes = tl.load(packed_ptr + packed_offsets, mask=mask_2d, other=0)

        if HAS_CODEBOOK:
            codes = tl.load(codebook_ptr + raw_codes.to(tl.int32)).to(tl.float16)
        else:
            codes = raw_codes.to(tl.float16)

        group_idx = col_offs // GROUP_SIZE
        scale_offsets = row_offs[:, None] * stride_sn + group_idx[None, :] * stride_sg
        group_mask = row_mask[:, None] & (group_idx[None, :] < n_groups)
        scale_vals = tl.load(scale_ptr + scale_offsets, mask=group_mask, other=1.0)
        gmin_vals = tl.load(gmin_ptr + scale_offsets, mask=group_mask, other=0.0)

        result = codes * scale_vals + gmin_vals

        out_offsets = row_offs[:, None] * stride_on + col_offs[None, :] * stride_ok
        out_mask = row_mask[:, None] & col_mask[None, :]
        tl.store(out_ptr + out_offsets, result, mask=out_mask)


def dequantize_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    gmin: torch.Tensor,
    bits: int,
    rows: int,
    cols: int,
    padded_cols: int,
    n_groups: int,
    group_size: int = SUPPORTED_GROUP_SIZE,
    compute_dtype: torch.dtype = torch.float16,
    codebook: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Dequantize packed weight to dense fp16/bf16 using a single fused Triton
    kernel. Returns None when Triton/CUDA is unavailable or the bit-width
    is unsupported.

    `codebook`, if given (fp32 `[1 << bits]` tensor), makes codes index into
    it before the affine transform — supports NF4 and Lloyd-Max codebooks.

    For large-M workloads (UNet, prefill), call this then torch.matmul —
    faster than the fused GEMM kernel which redundantly re-dequantizes per
    row-tile."""
    if not TRITON_AVAILABLE or not packed.is_cuda or bits not in (4, 8):
        return None

    out = torch.empty((rows, cols), device=packed.device, dtype=compute_dtype)

    BLOCK_ROW = 4
    BLOCK_COL = group_size  # must equal group_size for affine reconstruction

    grid = (
        (rows + BLOCK_ROW - 1) // BLOCK_ROW,
        (cols + BLOCK_COL - 1) // BLOCK_COL,
    )

    # Cast scale/gmin to compute_dtype for the kernel
    scale_cd = scale.to(compute_dtype) if scale.dtype != compute_dtype else scale
    gmin_cd = gmin.to(compute_dtype) if gmin.dtype != compute_dtype else gmin

    has_codebook = codebook is not None
    # Dummy pointer when no codebook — Triton needs a valid tensor bound.
    codebook_arg = codebook if has_codebook else out

    try:
        if bits == 4:
            _dequant_4bit_kernel[grid](
                packed, scale_cd, gmin_cd, out, codebook_arg,
                rows, cols, padded_cols, n_groups,
                packed.stride(0), packed.stride(1),
                scale_cd.stride(0), scale_cd.stride(1),
                out.stride(0), out.stride(1),
                GROUP_SIZE=group_size,
                BLOCK_ROW=BLOCK_ROW,
                BLOCK_COL=BLOCK_COL,
                HAS_CODEBOOK=has_codebook,
            )
        else:  # 8-bit
            _dequant_8bit_kernel[grid](
                packed, scale_cd, gmin_cd, out, codebook_arg,
                rows, cols, padded_cols, n_groups,
                packed.stride(0), packed.stride(1),
                scale_cd.stride(0), scale_cd.stride(1),
                out.stride(0), out.stride(1),
                GROUP_SIZE=group_size,
                BLOCK_ROW=BLOCK_ROW,
                BLOCK_COL=BLOCK_COL,
                HAS_CODEBOOK=has_codebook,
            )
    except Exception:
        return None

    return out

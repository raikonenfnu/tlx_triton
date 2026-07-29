"""Reproduce TLX async_load of a logical 2-D view of a physical 4-D tensor.

The input has physical shape [block, token_group, head_dim, token_in_group].
The direct path gathers/transposes that storage while copying global memory to
LDS.  The workaround copies the contiguous physical view and applies the view
transform in LDS.

Usage:
  python tlx_5d_repro.py --mode direct
  python tlx_5d_repro.py --mode layout
  python tlx_5d_repro.py --mode workaround
  python tlx_5d_repro.py --mode all
  python tlx_5d_repro.py --mode both --benchmark
"""

import argparse

import torch
import triton
import triton.language as tl
from triton.language.extra import tlx


@triton.jit
def repro(
    V,
    Out,
    stride_v_b: tl.constexpr,
    stride_v_po: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_v_x: tl.constexpr,
    USE_WORKAROUND: tl.constexpr,
    USE_EXPLICIT_LAYOUT: tl.constexpr,
    DIRECT_LAYOUT: tl.constexpr,
):
    if USE_WORKAROUND:
        # Copy V's contiguous physical [block * token_group, head_dim *
        # token_in_group] view, then form logical [token, head_dim] in LDS.
        physical_row = tl.arange(0, 16)
        physical_col = tl.arange(0, 512)
        block = physical_row // 8
        token_group = physical_row % 8
        ptrs = (
            V
            + block[:, None] * stride_v_b
            + token_group[:, None] * stride_v_po
            + physical_col[None, :]
        )

        smem = tlx.local_alloc((16, 512), V.dtype.element_ty, 2)
        copy_token = tlx.async_load(ptrs, tlx.local_view(smem, 0))
        tlx.async_load_commit_group([copy_token])
        tlx.async_load_wait_group(0)

        view = tlx.local_reshape(tlx.local_view(smem, 0), [16, 64, 8])
        view = tlx.local_trans(view, (0, 2, 1))
        view = tlx.local_reshape(view, [128, 64])
        value = tlx.local_load(view)
    else:
        # Express the logical gather/transpose directly in the global pointer
        # tensor passed to async_load.
        n = tl.arange(0, 128)
        d = tl.arange(0, 64)
        page = n // 64
        token = n % 64
        ptrs = (
            V
            + page[:, None] * stride_v_b
            + (token[:, None] // 8) * stride_v_po
            + d[None, :] * stride_v_d
            + (token[:, None] % 8) * stride_v_x
        )
        if USE_EXPLICIT_LAYOUT:
            # State the eight-element physical x-contiguity along logical n.
            # This mode is useful for separating AxisInfo inference from the
            # async-copy/LDS layout transformation.
            ptrs = tl.multiple_of(ptrs, (16, 2))
            ptrs = tl.max_contiguous(ptrs, (8, 1))

        if USE_EXPLICIT_LAYOUT:
            smem = tlx.local_alloc(
                (128, 64),
                V.dtype.element_ty,
                2,
                layout=DIRECT_LAYOUT,
            )
        else:
            smem = tlx.local_alloc((128, 64), V.dtype.element_ty, 2)
        copy_token = tlx.async_load(ptrs, tlx.local_view(smem, 0))
        tlx.async_load_commit_group([copy_token])
        tlx.async_load_wait_group(0)
        value = tlx.local_load(tlx.local_view(smem, 0))

    n = tl.arange(0, 128)
    d = tl.arange(0, 64)
    tl.store(Out + n[:, None] * 64 + d[None, :], value)


def reference(v):
    # [block, token_group, head_dim, token_in_group]
    return v.permute(0, 1, 3, 2).reshape(128, 64)


def run(use_workaround, use_explicit_layout=False, benchmark=False):
    v = torch.arange(2 * 8 * 64 * 8, device="cuda", dtype=torch.float32)
    v = v.reshape(2, 8, 64, 8).to(torch.bfloat16)
    out = torch.empty((128, 64), device="cuda", dtype=torch.bfloat16)
    direct_layout = tlx.swizzled_shared_layout_encoding.make_default(2)
    direct_layout = direct_layout.make_permute((1, 0))

    def launch():
        repro[(1,)](
            v,
            out,
            *v.stride(),
            USE_WORKAROUND=use_workaround,
            USE_EXPLICIT_LAYOUT=use_explicit_layout,
            DIRECT_LAYOUT=direct_layout,
            num_warps=4,
        )

    launch()
    torch.testing.assert_close(out, reference(v), rtol=0, atol=0)
    mode = (
        "workaround"
        if use_workaround
        else ("layout" if use_explicit_layout else "direct")
    )
    result = f"{mode}: PASS"
    if benchmark:
        result += f", {triton.testing.do_bench(launch):.4f} ms"
    print(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("direct", "layout", "workaround", "both", "all"),
        default="all",
    )
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    if args.mode in ("direct", "both", "all"):
        run(False, benchmark=args.benchmark)
    if args.mode in ("layout", "all"):
        run(False, use_explicit_layout=True, benchmark=args.benchmark)
    if args.mode in ("workaround", "both", "all"):
        run(True, benchmark=args.benchmark)


if __name__ == "__main__":
    main()

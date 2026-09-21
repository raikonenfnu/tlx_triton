"""Optional Origami plan selection for the gfx950 TLX GEMM kernels.

Origami models GEMM geometry, but it does not describe every compiler/backend
knob needed by the TLX register kernel.  Keep the executable plans here and
let Origami choose only between plans that TLX has declared launchable.

The dependency is intentionally imported on first use.  A normal
``triton.tlx.ops`` import therefore does not require the RAD-owned Python
bindings.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any


class OrigamiUnavailable(RuntimeError):
    """The optional Origami Python package is not installed."""


@dataclasses.dataclass(frozen=True)
class _ModelConfig:
    """Small Triton-config facade consumed by ``OrigamiMatmulSelector``."""

    kwargs: dict[str, int]


def _launch_plan(block_m, block_n, block_k, group_m, num_xcds, num_warps, num_stages):
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "GROUP_M": group_m,
        "NUM_XCDS": num_xcds,
        "matrix_instr_nonkdim": 16,
        "waves_per_eu": 0,
        "kpack": 1,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


# One executable recipe per modeled macrotile.  Keeping the keys unique avoids
# pretending Origami can distinguish compiler schedules that have identical
# GEMM geometry.  These recipes are a conservative subset of gfx950.py's
# register autotune space and stay within MI350's 160 KiB LDS capacity.
_PLANS = {
    (64, 32, 128): _launch_plan(64, 32, 128, 8, 1, 4, 2),
    (128, 64, 64): _launch_plan(128, 64, 64, 4, 8, 4, 2),
    (128, 128, 64): _launch_plan(128, 128, 64, 16, 8, 4, 2),
    (128, 128, 128): _launch_plan(128, 128, 128, 16, 8, 8, 2),
    (128, 256, 64): _launch_plan(128, 256, 64, 8, 1, 8, 3),
    (256, 128, 64): _launch_plan(256, 128, 64, 4, 1, 8, 2),
    (256, 256, 64): _launch_plan(256, 256, 64, 4, 8, 8, 2),
}


# Origami interprets waves_per_eu as modeled occupancy and expects it to be
# positive.  TLX's launch value 0 means "leave the compiler default alone", so
# the model-facing candidates use occupancy 1 while the selected launch recipe
# preserves TLX's measured value.
_MODEL_CONFIGS = tuple(
    _ModelConfig({
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "BLOCK_K": block_k,
        "waves_per_eu": 1,
    }) for block_m, block_n, block_k in _PLANS
)


def _selector_class():
    try:
        from origami import OrigamiMatmulSelector
    except (ImportError, AttributeError) as error:
        raise OrigamiUnavailable(
            "space='origami' requires the RAD Origami Python package"
        ) from error
    return OrigamiMatmulSelector


def _arch_name(selector: Any) -> str:
    arch = getattr(getattr(selector, "_hardware", None), "arch", None)
    if arch is None:
        return ""
    name = getattr(arch, "name", arch)
    if name is None:
        return ""
    return str(name).split(".")[-1]


def _select(selector_cls, m, n, k, dtype, device, a_stride, b_stride):
    selector = selector_cls(
        config_gen=_MODEL_CONFIGS,
        m=m,
        n=n,
        k=k,
        a_dtype=dtype,
        b_dtype=dtype,
        out_dtype=dtype,
        device=device,
        a_stride=tuple(a_stride),
        b_stride=tuple(b_stride),
    )
    arch = _arch_name(selector)
    if arch and arch != "gfx950":
        raise ValueError(f"Origami returned hardware {arch!r}; TLX expected 'gfx950'")

    tile = (
        int(selector.macrotile_m),
        int(selector.macrotile_n),
        int(selector.macrotile_k),
    )
    try:
        plan = dict(_PLANS[tile])
    except KeyError as error:
        raise ValueError(f"Origami selected unsupported TLX macrotile {tile}") from error

    # Workgroup mapping is a cache-locality decision the model does understand.
    # Accept only values already exercised by the register kernel.
    group_m = int(selector.wgm)
    if group_m in (1, 2, 4, 8, 16):
        plan["GROUP_M"] = group_m
    return plan


@functools.lru_cache(maxsize=512)
def _select_cached(m, n, k, dtype, device, a_stride, b_stride):
    return _select(
        _selector_class(), m, n, k, dtype, device, a_stride, b_stride
    )


def select_plan(m, n, k, dtype, device, a_stride, b_stride, *, selector_cls=None):
    """Return a gfx950 TLX register launch plan selected by Origami.

    ``selector_cls`` is an injection point for contract tests and for evaluating
    a new Origami release before making it the supported dependency.
    """
    if selector_cls is not None:
        return _select(
            selector_cls, m, n, k, dtype, device, a_stride, b_stride
        )
    return dict(
        _select_cached(
            m,
            n,
            k,
            dtype,
            device,
            tuple(a_stride),
            tuple(b_stride),
        )
    )

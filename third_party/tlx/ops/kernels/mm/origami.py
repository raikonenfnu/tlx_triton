"""Origami-backed launch selection for gfx950 TLX GEMM macro-kernels.

The registry is the contract between the compiler-owned TLX kernels and the
RAD-owned Origami model. TLX registers executable macro-kernels, grouped by
problem variant, and Origami chooses among their unique tile geometries. The
returned decision includes both the chosen kernel and Origami's launch shape;
callers must not infer a different kernel family from the problem dimensions.

Origami is an optional, lazily imported dependency. Importing ``triton.tlx``
therefore does not require the Origami Python wheel or shared library.
"""

from __future__ import annotations

import dataclasses
import functools
from types import MappingProxyType
from typing import Any, Mapping


class OrigamiUnavailable(RuntimeError):
    """The optional Origami Python package is not installed."""


@dataclasses.dataclass(frozen=True)
class _ModelConfig:
    """Small Triton-config facade consumed by ``OrigamiMatmulSelector``."""

    kwargs: Mapping[str, int]


@dataclasses.dataclass(frozen=True)
class MacroKernel:
    """One compiler-validated implementation exposed to Origami.

    Tiles must be unique within a problem variant because the current Origami
    Python API returns the selected geometry, not an opaque candidate ID.
    ``launch`` names a launcher in gfx950.py; ``options`` are compiler knobs
    owned by that launcher rather than modeled choices.
    """

    name: str
    variant: str
    tile: tuple[int, int, int]
    launch: str
    model: Mapping[str, int]
    options: Mapping[str, int]


@dataclasses.dataclass(frozen=True)
class LaunchDecision:
    """Complete model result consumed by the gfx950 launcher."""

    kernel: MacroKernel
    grid_size: int
    reduction: str
    wgm: int
    wgmxcc: int
    wgmxccchunk: int
    number_of_cus: int

    @property
    def tile(self) -> tuple[int, int, int]:
        return self.kernel.tile


def _options(**kwargs: int) -> Mapping[str, int]:
    return MappingProxyType(kwargs)


def _register_kernel(name, tile, *, variant="register", model_waves_per_eu=1, **kwargs):
    return MacroKernel(name, variant, tile, "register", _options(waves_per_eu=model_waves_per_eu),
                       _options(**kwargs))


# Stream-K is the default mm variant. Its candidate list intentionally contains
# only tiles implemented by the adaptive gfx950 macro-kernel. The same registered
# tile consumes Origami grids smaller than, equal to, or larger than the output-
# tile count as persistent/Stream-K, data-parallel, or split-K schedules. Exact
# parallel split-K grids use the inter-wave kernel's dependency-free FP32
# workspace reduction; genuine Stream-K and persistent grids retain the lock-
# based scheduling shell.
_STREAMK_KERNELS = (
    MacroKernel("streamk_128x128x64", "streamk", (128, 128, 64), "adaptive_streamk",
                _options(waves_per_eu=1, streamk_grid=1),
                _options(num_warps=8, GROUP_M=4, cooperative_fixup=0, parallel_workspace=1)),
    MacroKernel("streamk_256x256x64", "streamk", (256, 256, 64), "adaptive_streamk",
                _options(waves_per_eu=1, streamk_grid=1),
                _options(num_warps=8, GROUP_M=4, cooperative_fixup=0, parallel_workspace=1)),
)


_FUSED_STREAMK_KERNELS = tuple(
    dataclasses.replace(
        kernel,
        name=kernel.name.replace("streamk_", "streamk_fused_"),
        variant="fused_streamk",
    )
    for kernel in _STREAMK_KERNELS
)


_TAIL_DATA_KERNELS = tuple(
    dataclasses.replace(
        kernel,
        name=kernel.name.replace("streamk_", "streamk_tail_data_"),
        variant="tail_data",
        model=_options(waves_per_eu=kernel.model["waves_per_eu"], streamk_grid=0),
    )
    for kernel in _STREAMK_KERNELS
)


# Fused addmm currently uses the register macro-kernel family. It remains a
# separate problem variant so a bias-capable Stream-K template can replace it
# without changing the selector/registry interface.
_FUSED_ADDMM_KERNELS = (
    _register_kernel("register_fused_64x32x128", (64, 32, 128), variant="fused_addmm",
                     GROUP_M=8, NUM_XCDS=1, num_warps=4, num_stages=2),
    _register_kernel("register_fused_128x64x64", (128, 64, 64), variant="fused_addmm",
                     GROUP_M=4, NUM_XCDS=8, num_warps=4, num_stages=2),
    _register_kernel("register_fused_128x128x64", (128, 128, 64), variant="fused_addmm",
                     GROUP_M=16, NUM_XCDS=8, num_warps=4, num_stages=3),
    _register_kernel("register_fused_128x128x128", (128, 128, 128), variant="fused_addmm",
                     GROUP_M=16, NUM_XCDS=8, num_warps=8, num_stages=2),
    _register_kernel("register_fused_128x256x64", (128, 256, 64), variant="fused_addmm",
                     GROUP_M=8, NUM_XCDS=1, num_warps=8, num_stages=3),
    _register_kernel("register_fused_256x128x64", (256, 128, 64), variant="fused_addmm",
                     GROUP_M=4, NUM_XCDS=1, num_warps=8, num_stages=2),
    _register_kernel("register_fused_256x256x64", (256, 256, 64), variant="fused_addmm",
                     GROUP_M=4, NUM_XCDS=8, num_warps=8, num_stages=2),
)


_TAIL_KERNELS = tuple(
    dataclasses.replace(
        kernel,
        name=kernel.name.replace("register_fused_", "register_tail_"),
        variant="tail",
    )
    for kernel in _FUSED_ADDMM_KERNELS
)


_TAIL_LDS_KERNELS = (
    MacroKernel("lds_tail_128x128x64", "tail_lds", (128, 128, 64), "lds",
                _options(waves_per_eu=1), _options(num_warps=4, GROUP_M=4)),
    MacroKernel("lds_tail_256x256x64", "tail_lds", (256, 256, 64), "lds",
                _options(waves_per_eu=1), _options(num_warps=8, GROUP_M=4)),
)


MACRO_KERNEL_REGISTRY = MappingProxyType({
    ("gfx950", "streamk"): _STREAMK_KERNELS,
    ("gfx950", "fused_streamk"): _FUSED_STREAMK_KERNELS,
    ("gfx950", "tail_data"): _TAIL_DATA_KERNELS,
    ("gfx950", "fused_addmm"): _FUSED_ADDMM_KERNELS,
    ("gfx950", "tail"): _TAIL_KERNELS,
    ("gfx950", "tail_lds"): _TAIL_LDS_KERNELS,
})


def _validate_registry() -> None:
    for (arch, variant), kernels in MACRO_KERNEL_REGISTRY.items():
        if not kernels:
            raise ValueError(f"empty Origami macro-kernel registry for {arch}/{variant}")
        tiles = [kernel.tile for kernel in kernels]
        if len(tiles) != len(set(tiles)):
            raise ValueError(f"duplicate Origami tile in {arch}/{variant}: {tiles}")
        if any(kernel.variant != variant for kernel in kernels):
            raise ValueError(f"variant mismatch in Origami registry for {arch}/{variant}")


_validate_registry()


def _model_configs(kernels):
    # Origami interprets waves_per_eu as modeled occupancy and expects it to be
    # positive. It is deliberately independent of Triton's launch option.
    return tuple(
        _ModelConfig({
            "BLOCK_M": kernel.tile[0],
            "BLOCK_N": kernel.tile[1],
            "BLOCK_K": kernel.tile[2],
            "waves_per_eu": kernel.model["waves_per_eu"],
        })
        for kernel in kernels
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


def _reduction_name(selector: Any, *, streamk: bool) -> str:
    if not streamk:
        return "none"
    try:
        import origami

        reduction = origami.select_reduction(
            selector._problem,
            selector._hardware,
            selector._result.config,
            origami.grid_selection_t.k_split_aware,
        )
    except (AttributeError, ImportError):
        # Keep compatibility with injectable selectors and older Origami wheels.
        return "unknown"
    return str(reduction).rsplit(".", 1)[-1].lower()


def _select(selector_cls, m, n, k, dtype, device, a_stride, b_stride, variant):
    registry_key = ("gfx950", variant)
    try:
        kernels = MACRO_KERNEL_REGISTRY[registry_key]
    except KeyError as error:
        raise ValueError(f"unknown Origami problem variant {variant!r} for gfx950") from error

    streamk = all(kernel.model.get("streamk_grid", 0) for kernel in kernels)
    selector = selector_cls(
        config_gen=_model_configs(kernels),
        m=m,
        n=n,
        k=k,
        a_dtype=dtype,
        b_dtype=dtype,
        out_dtype=dtype,
        device=device,
        a_stride=tuple(a_stride),
        b_stride=tuple(b_stride),
        streamk=streamk,
    )
    arch = _arch_name(selector)
    if arch and arch != "gfx950":
        raise ValueError(f"Origami returned hardware {arch!r}; TLX expected 'gfx950'")

    tile = (
        int(selector.macrotile_m),
        int(selector.macrotile_n),
        int(selector.macrotile_k),
    )
    by_tile = {kernel.tile: kernel for kernel in kernels}
    try:
        kernel = by_tile[tile]
    except KeyError as error:
        raise ValueError(f"Origami selected unsupported TLX macrotile {tile}") from error

    grid_size = int(selector.grid_size)
    if grid_size <= 0:
        raise ValueError(f"Origami selected invalid grid size {grid_size}")
    wgm = int(selector.wgm)
    wgmxcc = int(selector.wgmxcc)
    wgmxccchunk = int(selector.wgmxccchunk)
    number_of_cus = int(selector.number_of_cus)
    if wgm <= 0 or wgmxcc < 0 or wgmxccchunk < 0 or number_of_cus <= 0:
        raise ValueError(
            "Origami selected invalid workgroup mapping "
            f"(wgm={wgm}, wgmxcc={wgmxcc}, wgmxccchunk={wgmxccchunk}, cus={number_of_cus})"
        )
    return LaunchDecision(
        kernel=kernel,
        grid_size=grid_size,
        reduction=_reduction_name(selector, streamk=streamk),
        wgm=wgm,
        wgmxcc=wgmxcc,
        wgmxccchunk=wgmxccchunk,
        number_of_cus=number_of_cus,
    )


@functools.lru_cache(maxsize=512)
def _select_cached(m, n, k, dtype, device, a_stride, b_stride, variant):
    return _select(
        _selector_class(), m, n, k, dtype, device, a_stride, b_stride, variant
    )


def select_plan(m, n, k, dtype, device, a_stride, b_stride, *, variant="streamk", selector_cls=None):
    """Select a registered macro-kernel and its Origami launch parameters."""
    if selector_cls is not None:
        return _select(
            selector_cls, m, n, k, dtype, device, a_stride, b_stride, variant
        )
    return _select_cached(
        m,
        n,
        k,
        dtype,
        device,
        tuple(a_stride),
        tuple(b_stride),
        variant,
    )

"""Standalone fused addmm for gfx950 using TLX GEMM kernels.

The operation is ``out = bias + a @ b``. ``a`` must be row-major and ``b``
must be column-major so the kernel can issue coalesced loads. The bias may be
one-dimensional ``(N,)`` or broadcastable to ``(M, N)``.
"""

import torch

import triton
from triton.language.extra.tlx.tutorials.gfx9_gemm.inter_wave.a16w16.matmul_kernel import (
    BLOCK_K,
    BLOCK_M,
    BLOCK_N,
    NUM_CU,
    _aligned_split_tail_plan,
    _has_streamk_schedule,
    _launch,
    _launch_aligned_split_tail,
    _launch_rebased_persistent,
    _launch_register,
    _needs_i64_offsets,
    _streamk_schedule,
    streamk_matmul,
)

_PATH_AUTOTUNE_WARMUP = 25
_PATH_AUTOTUNE_REP = 100
_PATH_CACHE: dict[tuple[object, ...], str] = {}
_MAX_DEFERRED_EPILOGUE_ELEMENTS = 16 * 1024 * 1024
_MAX_BUFFER_BYTES = 2**31 - 1


def _can_use_inter_wave(a: torch.Tensor, b: torch.Tensor) -> bool:
    # The direct-to-LDS buffer resource uses signed 32-bit byte offsets. Keep the
    # whole operand addressable; wrapping above 2 GiB silently corrupts later rows.
    addressable = max(a.numel() * a.element_size(), b.numel() * b.element_size()) <= _MAX_BUFFER_BYTES
    return addressable and a.shape[1] >= 2 * BLOCK_K and a.shape[1] % BLOCK_K == 0


def _can_use_inter_wave_tail(a: torch.Tensor, b: torch.Tensor) -> bool:
    M, K = a.shape
    N = b.shape[1]
    output_elements = M * N
    return (K > 1536 and K % BLOCK_K != 0 and K * a.element_size() % 16 == 0 and a.element_size() == 2
            and 2 * 1024 * 1024 < output_elements <= _MAX_DEFERRED_EPILOGUE_ELEMENTS)


def _can_use_streamk(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> bool:
    M, K = a.shape
    N = b.shape[1]
    # N-contiguous bias loads preserve the accumulator layout without extra LDS.
    return bias.stride(1) == 1 and _can_use_inter_wave(a, b) and _has_streamk_schedule(M, N, K)


def _can_use_rebased_persistent(a: torch.Tensor, b: torch.Tensor) -> bool:
    M, K = a.shape
    N = b.shape[1]
    aligned = M % BLOCK_M == 0 and N % BLOCK_N == 0 and K >= 2 * BLOCK_K and K % (2 * BLOCK_K) == 0
    return (aligned and _needs_i64_offsets(a) and not _needs_i64_offsets(b)
            and not _streamk_schedule(M, N, K)["HAS_STREAMK"])


def _path_key(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    split_k,
) -> tuple[object, ...]:
    return (
        a.device.type,
        a.device.index,
        a.dtype,
        a.shape,
        b.shape,
        bias.stride(),
        a.stride(),
        b.stride(),
        split_k,
    )


def _launch_inter_wave_with_tail(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run aligned K through inter-wave and finish the tail from its fp32 partials."""
    M, K = a.shape
    _, N = b.shape
    # The fused epilogue can afford a second CU wave when the output grid alone
    # underfills the device; the common planner still balances shorter K slices
    # against the larger fp32 reduction and masked tail.
    plan = _aligned_split_tail_plan(M, N, K, program_budget=2 * NUM_CU)
    if plan is None:
        plan = (K // (2 * BLOCK_K) * (2 * BLOCK_K), 1)
    return _launch_aligned_split_tail(a, b, *plan, bias=bias)


def available_paths(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[str, ...]:
    """Return the paths that are valid for these operands, in registry order.

    Correctness tests iterate this and assert every path against a reference,
    rather than letting `_autotune_path` race them and check only the winner.
    That matters for more than wall clock: the race admits a candidate only if
    it already agrees with `register`, so a genuinely wrong `inter_wave` would
    be dropped from the timing pool and the suite would still pass green.
    """
    if _can_use_rebased_persistent(a, b):
        return ("register", "persistent")
    if _can_use_inter_wave(a, b):
        return (("register", "inter_wave", "stream_k") if _can_use_streamk(bias, a, b) else ("register", "inter_wave"))
    if _can_use_inter_wave_tail(a, b):
        return ("register", "inter_wave_tail")
    return ("register", )


def _dispatch(path: str, bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor, split_k, config=None):
    if path == "stream_k":
        return streamk_matmul(a, b, bias)
    if path == "persistent":
        return _launch_rebased_persistent(a, b, bias)
    if path == "inter_wave":
        return _launch(a, b, bias=bias, SPLIT_K=split_k)
    if path == "inter_wave_tail":
        return _launch_inter_wave_with_tail(bias, a, b)
    if path == "register":
        return _launch_register(a, b, bias=bias, config=config)
    raise ValueError(f"Unknown addmm path {path!r}; expected one of {available_paths(bias, a, b)}")


def _autotune_path(
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    split_k,
) -> str:
    key = _path_key(bias, a, b, split_k)
    cached = _PATH_CACHE.get(key)
    if cached is not None:
        return cached

    register = lambda: _launch_register(a, b, bias=bias)
    register_output = register()
    candidates = {"register": register}
    if _can_use_rebased_persistent(a, b):
        persistent = lambda: _launch_rebased_persistent(a, b, bias)
        persistent_output = persistent()
        if torch.allclose(register_output, persistent_output, rtol=1e-2, atol=1e-2):
            candidates["persistent"] = persistent
    elif _can_use_inter_wave(a, b):
        inter_wave = lambda: _launch(a, b, bias=bias, SPLIT_K=split_k)
        inter_wave_output = inter_wave()
        if torch.allclose(register_output, inter_wave_output, rtol=1e-2, atol=1e-2):
            candidates["inter_wave"] = inter_wave
        if _can_use_streamk(bias, a, b):
            stream_k = lambda: streamk_matmul(a, b, bias)
            stream_k_output = stream_k()
            if torch.allclose(register_output, stream_k_output, rtol=1e-2, atol=1e-2):
                candidates["stream_k"] = stream_k
    elif _can_use_inter_wave_tail(a, b):
        inter_wave_tail = lambda: _launch_inter_wave_with_tail(bias, a, b)
        inter_wave_tail_output = inter_wave_tail()
        if torch.allclose(register_output, inter_wave_tail_output, rtol=1e-2, atol=1e-2):
            candidates["inter_wave_tail"] = inter_wave_tail
    timings = {
        name:
        triton.testing.do_bench(
            candidate,
            warmup=_PATH_AUTOTUNE_WARMUP,
            rep=_PATH_AUTOTUNE_REP,
            return_mode="median",
        )
        for name, candidate in candidates.items()
    }
    winner = min(timings, key=timings.__getitem__)
    _PATH_CACHE[key] = winner
    return winner


def addmm(bias: torch.Tensor, a: torch.Tensor, b: torch.Tensor, SPLIT_K=None, path=None, config=None):
    """Return ``bias + a @ b`` using a gfx950 TLX path.

    ``path`` pins one of `available_paths`; the default (None) times the valid
    paths against each other and keeps the winner per shape. ``config`` pins
    the register path's kernel config, bypassing its autotuner. Correctness
    tests should pass both: neither the path race nor the config sweep tells
    them anything, and together they dominate the suite's wall clock.
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("addmm expects two-dimensional matrix operands")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Incompatible matrix dimensions: {tuple(a.shape)} and {tuple(b.shape)}")
    if not a.is_contiguous():
        raise ValueError("a must be row-major contiguous")
    if b.stride(0) != 1:
        raise ValueError("b must be column-major (stride(0) == 1)")

    M, _ = a.shape
    _, N = b.shape
    try:
        bias_2d = torch.broadcast_to(bias, (M, N))
    except RuntimeError as error:
        raise ValueError(f"Bias shape {tuple(bias.shape)} is not broadcastable to ({M}, {N})") from error
    if SPLIT_K not in (None, 1):
        if not _can_use_inter_wave(a, b):
            raise ValueError("SPLIT_K is only supported by the inter-wave kernel")
        if path not in (None, "inter_wave"):
            raise ValueError(f"SPLIT_K > 1 requires the inter_wave path, got {path!r}")
        return _launch(a, b, bias=bias_2d, SPLIT_K=SPLIT_K)
    if path is not None:
        valid = available_paths(bias_2d, a, b)
        if path not in valid:
            raise ValueError(f"Path {path!r} is not valid for this shape; valid paths are {valid}")
        return _dispatch(path, bias_2d, a, b, SPLIT_K, config=config)
    return _dispatch(_autotune_path(bias_2d, a, b, SPLIT_K), bias_2d, a, b, SPLIT_K, config=config)

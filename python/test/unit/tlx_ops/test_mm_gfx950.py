"""gfx950 correctness coverage for ``tlx.ops.mm``."""

import time

import pytest
import torch
from triton._internal_testing import is_hip_cdna4
from triton.tlx.ops import InvalidInput, UnsupportedOp
from triton.tlx.ops.kernels.mm import gfx950 as _gfx950
from triton.tlx.ops.kernels.mm._shapes import GFX950_FOCUS, operand

pytestmark = pytest.mark.skipif(not is_hip_cdna4(), reason="Requires gfx950")

MAX_SECONDS_PER_CASE = 60


def _assert_strides(tensor, wanted):
    for dim, (got, expected) in enumerate(zip(tensor.stride(), wanted)):
        if tensor.shape[dim] != 1:
            assert got == expected, (f"dim {dim}: stride {got}, recorded {expected}")


@pytest.mark.parametrize("m,n,k,a_strides,b_strides,dtype_name", GFX950_FOCUS)
def test_mm(m, n, k, a_strides, b_strides, dtype_name):
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[dtype_name]
    from triton.tlx.ops import mm as tlx_mm

    a = operand(m, k, a_strides, dtype)
    b = operand(k, n, b_strides, dtype)
    _assert_strides(a, a_strides)
    _assert_strides(b, b_strides)

    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        out = tlx_mm(a, b, arch="gfx950", space="heuristic")
    except (InvalidInput, UnsupportedOp) as declined:
        pytest.fail(f"gfx950 declines its focus shape: {declined}")
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert elapsed < MAX_SECONDS_PER_CASE, (f"mm({m}x{n}x{k}, {dtype}) took {elapsed:.1f}s, "
                                            f"over the {MAX_SECONDS_PER_CASE}s budget")

    expected = torch.matmul(a, b)
    tolerance = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(
        out,
        expected,
        atol=tolerance * expected.abs().max().item(),
        rtol=tolerance,
    )


@pytest.mark.parametrize(
    "m,n,k,dtype",
    [
        (279, 256, 4096, torch.float16),
        (1024, 4096, 800, torch.bfloat16),
    ],
    ids=["intermediate-fp16", "full-grid-bf16"],
)
def test_mm_register_fallback(m, n, k, dtype):
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((m, k), device="cuda", dtype=dtype)
    b = torch.randn((n, k), device="cuda", dtype=dtype).T

    out = tlx_mm(a, b, arch="gfx950", space="heuristic")
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        out,
        expected,
        atol=1e-2 * expected.abs().max().item(),
        rtol=1e-2,
    )


def test_mm_rejects_invalid_rank():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((16, ), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 16), device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="rank-2"):
        tlx_mm(a, b, arch="gfx950")


def test_mm_rejects_mismatched_reduction_dimensions():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((17, 8), device="cuda", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="reduction dimensions"):
        tlx_mm(a, b, arch="gfx950")


def test_mm_rejects_mismatched_dtype():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 8), device="cuda", dtype=torch.float32)
    with pytest.raises(InvalidInput, match="same dtype and device"):
        tlx_mm(a, b, arch="gfx950")


def test_mm_rejects_mismatched_device():
    from triton.tlx.ops import mm as tlx_mm

    a = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 8), device="cpu", dtype=torch.float16)
    with pytest.raises(InvalidInput, match="same dtype and device"):
        tlx_mm(a, b, arch="gfx950")


def test_mm_rejects_invalid_space():
    from triton.tlx.ops.kernels.mm.gfx950 import mm

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    with pytest.raises(InvalidInput, match="space='heuristic' or space='origami'"):
        mm(a, b, space="full")


def test_origami_plan_translation_uses_only_validated_tlx_knobs():
    from triton.tlx.ops.kernels.mm.origami import select_plan

    class Arch:
        name = "gfx950"

    class Hardware:
        arch = Arch()

    class FakeSelector:
        def __init__(self, **kwargs):
            assert kwargs["m"] == 4096
            assert kwargs["n"] == 4096
            assert kwargs["k"] == 4096
            assert kwargs["a_stride"] == (4096, 1)
            assert kwargs["b_stride"] == (1, 4096)
            assert kwargs["streamk"]
            assert all(c.kwargs["waves_per_eu"] == 1 for c in kwargs["config_gen"])
            self._hardware = Hardware()
            self.macrotile_m = 256
            self.macrotile_n = 256
            self.macrotile_k = 64
            self.grid_size = 192
            self.wgm = 8
            self.wgmxcc = 8
            self.wgmxccchunk = 4
            self.number_of_cus = 256

    plan = select_plan(
        4096,
        4096,
        4096,
        torch.float16,
        torch.device("cuda:0"),
        (4096, 1),
        (1, 4096),
        selector_cls=FakeSelector,
    )
    assert plan.kernel.name == "streamk_256x256x64"
    assert plan.kernel.options["cooperative_fixup"] == 0
    assert plan.tile == (256, 256, 64)
    assert plan.grid_size == 192
    assert plan.reduction == "unknown"
    assert (plan.wgm, plan.wgmxcc, plan.wgmxccchunk) == (8, 8, 4)
    assert plan.number_of_cus == 256


def test_origami_plan_rejects_wrong_architecture():
    from triton.tlx.ops.kernels.mm.origami import select_plan

    class Selector:
        def __init__(self, **_):
            self._hardware = type("Hardware", (), {"arch": "gfx942"})()
            self.macrotile_m = 128
            self.macrotile_n = 128
            self.macrotile_k = 64
            self.grid_size = 64
            self.wgm = 4
            self.wgmxcc = 8
            self.wgmxccchunk = 0
            self.number_of_cus = 304

    with pytest.raises(ValueError, match="expected 'gfx950'"):
        select_plan(
            1024,
            1024,
            1024,
            torch.float16,
            torch.device("cuda:0"),
            (1024, 1),
            (1, 1024),
            selector_cls=Selector,
        )


@pytest.mark.parametrize(
    "tiles,grid,expected",
    [
        (64, 128, (0, True)),       # split-K: more workgroups than output tiles
        (256, 256, (256, False)),   # data parallel: one workgroup per tile
        (320, 213, (213, True)),    # genuine Stream-K tail
        (1024, 256, (256, True)),   # persistent: several tiles per workgroup
    ],
)
def test_origami_grid_drives_streamk_schedule(tiles, grid, expected):
    # Use a 128x128 tile and choose M/N to produce exactly ``tiles`` tiles.
    schedule = _gfx950._origami_streamk_schedule(128, tiles * 128, 1024, 128, 128, grid)
    assert (schedule["NUM_FULL_TILES"], schedule["HAS_STREAMK"]) == expected
    assert schedule["NUM_PROGRAMS"] == grid


def test_origami_data_parallel_schedule_handles_k_tail():
    schedule = _gfx950._origami_streamk_schedule(1024, 1024, 43500, 256, 256, 16)
    assert schedule["NUM_PROGRAMS"] == schedule["NUM_FULL_TILES"] == 16
    assert not schedule["HAS_STREAMK"]
    assert schedule["HAS_K_TAIL"]
    assert schedule["K_PIPE_STEPS"] * 64 == 43392


def test_origami_parallel_reduction_uses_exact_interwave_split():
    from triton.tlx.ops.kernels.mm.origami import MACRO_KERNEL_REGISTRY, LaunchDecision

    kernel = MACRO_KERNEL_REGISTRY[("gfx950", "streamk")][1]
    decision = LaunchDecision(kernel, 48, "parallel", 1, 8, 0, 256)
    assert _gfx950._origami_parallel_split_k(768, 256, 851968, 2, decision) == 16

    # A genuine Stream-K grid is not an integer split of the output tiles.
    decision = LaunchDecision(kernel, 213, "tree", 1, 8, 0, 256)
    assert _gfx950._origami_parallel_split_k(1024, 20480, 6144, 2, decision) is None


@pytest.mark.parametrize(
    "m,n,k,budget,expected",
    [
        (1024, 1024, 43500, 256, (43264, 13)),
        (3072, 3072, 11800, 512, (11776, 3)),
        # An already aligned K does not need a fused masked-tail launch.
        (1024, 1024, 43520, 256, None),
        # A sufficiently broad output grid should use ordinary data parallelism.
        (4096, 4096, 43500, 256, None),
    ],
)
def test_aligned_split_tail_plan_is_geometry_driven(m, n, k, budget, expected):
    assert _gfx950._aligned_split_tail_plan(
        m,
        n,
        k,
        tile=(256, 256),
        program_budget=budget,
    ) == expected


def test_origami_registry_parallel_grid_uses_template_cost():
    from triton.tlx.ops.kernels.mm.origami import MACRO_KERNEL_REGISTRY, _parallel_workspace_grid

    kernel = MACRO_KERNEL_REGISTRY[("gfx950", "streamk")][1]
    grid, policy = _parallel_workspace_grid(
        768,
        256,
        851968,
        kernel,
        256,
        48,
        "parallel",
    )
    assert (grid, policy) == (192, "registry_parallel_cost")

    # Non-parallel reductions remain entirely model-owned.
    assert _parallel_workspace_grid(
        768,
        256,
        851968,
        kernel,
        256,
        47,
        "tree",
    ) == (47, "origami")


def test_origami_registry_resident_tail_is_a_wave_regime():
    from triton.tlx.ops.kernels.mm.origami import MACRO_KERNEL_REGISTRY, _resident_streamk_grid

    kernel = MACRO_KERNEL_REGISTRY[("gfx950", "streamk")][1]
    assert _resident_streamk_grid(
        1024,
        20480,
        6144,
        kernel,
        256,
        320,
        "tree",
        "origami",
    ) == (256, "registry_resident_tail")
    # At two complete waves there is no Stream-K tail to fold into residents.
    assert _resident_streamk_grid(
        1024,
        32768,
        6144,
        kernel,
        256,
        512,
        "tree",
        "origami",
    ) == (512, "origami")


def test_origami_registry_short_k_uses_independent_output_tiles():
    from triton.tlx.ops.kernels.mm.origami import MACRO_KERNEL_REGISTRY, _short_k_data_parallel_grid

    kernel = MACRO_KERNEL_REGISTRY[("gfx950", "streamk")][1]
    assert _short_k_data_parallel_grid(
        851968,
        256,
        768,
        kernel,
        256,
        256,
        "tree",
        "origami",
    ) == (3328, "registry_data_parallel")
    # Longer K retains the model-selected persistent traversal.
    assert _short_k_data_parallel_grid(
        851968,
        256,
        1152,
        kernel,
        256,
        256,
        "tree",
        "origami",
    ) == (256, "origami")


def test_origami_registry_rebases_only_the_oversized_streamed_operand():
    from triton.tlx.ops.kernels.mm.origami import MACRO_KERNEL_REGISTRY, _rebased_persistent_grid

    kernel = MACRO_KERNEL_REGISTRY[("gfx950", "fused_streamk")][1]
    args = (2252800, 256, 512, kernel, 256, 8800, "tree", "origami")
    assert _rebased_persistent_grid(
        *args,
        (512, 1),
        (1, 512),
        2,
    ) == (256, "registry_rebased_persistent")
    # Rebasing A cannot rescue a view whose B resource also exceeds i32.
    assert _rebased_persistent_grid(
        *args,
        (512, 1),
        (2252800, 1),
        2,
    ) == (8800, "origami")


def test_origami_adaptive_dispatches_parallel_split_to_interwave(monkeypatch):
    from triton.tlx.ops.kernels.mm.origami import MACRO_KERNEL_REGISTRY, LaunchDecision

    launches = []

    def fake_lds(a, b, **kwargs):
        launches.append(kwargs)
        return kwargs["out"]

    def fail_streamk(*_args, **_kwargs):
        pytest.fail("exact parallel split-K should not use lock-based Stream-K")

    monkeypatch.setattr(_gfx950, "_launch_lds", fake_lds)
    monkeypatch.setattr(_gfx950, "_launch_origami_streamk", fail_streamk)
    kernel = MACRO_KERNEL_REGISTRY[("gfx950", "streamk")][1]
    decision = LaunchDecision(kernel, 48, "parallel", 1, 8, 0, 256)
    a = torch.empty((768, 851968), device="meta", dtype=torch.bfloat16)
    b = torch.empty((851968, 256), device="meta", dtype=torch.bfloat16)
    out = torch.empty((768, 256), device="meta", dtype=torch.bfloat16)

    assert _gfx950._launch_origami_adaptive(a, b, decision, out) is out
    assert len(launches) == 1
    assert launches[0]["bias"] is None
    assert launches[0]["SPLIT_K"] == 16
    assert launches[0]["TILE"] == (256, 256)
    assert launches[0]["out"] is out


def test_origami_adaptive_dispatches_short_k_to_data_parallel(monkeypatch):
    from triton.tlx.ops.kernels.mm.origami import MACRO_KERNEL_REGISTRY, LaunchDecision

    launches = []

    def fake_lds(a, b, **kwargs):
        launches.append(kwargs)
        return kwargs["out"]

    def fail_streamk(*_args, **_kwargs):
        pytest.fail("independent short-K output tiles should not use persistent Stream-K")

    monkeypatch.setattr(_gfx950, "_launch_lds", fake_lds)
    monkeypatch.setattr(_gfx950, "_launch_origami_streamk", fail_streamk)
    kernel = MACRO_KERNEL_REGISTRY[("gfx950", "streamk")][1]
    decision = LaunchDecision(
        kernel,
        3328,
        "tree",
        1,
        8,
        0,
        256,
        model_grid_size=256,
        grid_policy="registry_data_parallel",
    )
    a = torch.empty((851968, 768), device="meta", dtype=torch.bfloat16)
    b = torch.empty((768, 256), device="meta", dtype=torch.bfloat16)
    out = torch.empty((851968, 256), device="meta", dtype=torch.bfloat16)

    assert _gfx950._launch_origami_adaptive(a, b, decision, out) is out
    assert launches == [{
        "bias": None,
        "SPLIT_K": 1,
        "TILE": (256, 256),
        "out": out,
    }]


@pytest.mark.parametrize("op", ["mm", "addmm"])
def test_origami_space_matches_eager(op):
    pytest.importorskip("origami")
    from triton.tlx.ops import addmm as tlx_addmm
    from triton.tlx.ops import mm as tlx_mm

    m, n, k = 512, 512, 512
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    if op == "mm":
        actual = tlx_mm(a, b, arch="gfx950", space="origami")
        expected = torch.mm(a, b)
    else:
        bias = torch.randn((n,), device="cuda", dtype=torch.float16)
        actual = tlx_addmm(bias, a, b, arch="gfx950", space="origami")
        expected = torch.addmm(bias, a, b)
    torch.testing.assert_close(
        actual,
        expected,
        atol=2e-2 * expected.abs().max().item(),
        rtol=2e-2,
    )


def test_mm_rejects_unsupported_operands():
    from triton.tlx.ops.kernels.mm.gfx950 import matmul, mm, supports

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    unsupported_a = torch.randn((17, 64), device="cuda", dtype=torch.float16)
    unsupported_b = torch.randn((8192, 64), device="cuda", dtype=torch.float16).T
    assert supports(a, b)
    assert not supports(unsupported_a, unsupported_b)
    assert not supports(a.to(torch.float32), b.to(torch.float32))
    # The gfx950 LDS producer now has explicit row-major-B and column-major-A
    # swizzles, so either dense orientation is a supported operand layout.
    assert supports(a, b.contiguous())
    with pytest.raises(InvalidInput, match="does not support"):
        mm(unsupported_a, unsupported_b)
    with pytest.raises(InvalidInput, match="does not support"):
        matmul(unsupported_a, unsupported_b)


def test_mm_lds_respects_output_strides():
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    m, n, k = 2048, 512, 2048
    assert _gfx950._dispatch_plan(m, n, k, torch.float16, 2) == (
        "lds",
        (128, 128, 2),
    )
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((n, k), device="cuda", dtype=torch.float16).T
    out = torch.empty((n, m), device="cuda", dtype=torch.float16).T

    actual = matmul(a, b, out=out)
    expected = torch.matmul(a, b)

    assert actual is out
    assert out.stride() == (1, m)
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


def test_mm_offset_width_selection():
    i32_max_element = (1 << 30) - 1
    within_i32 = torch.empty(
        (i32_max_element + 1,), device="meta", dtype=torch.float16
    )
    beyond_i32 = torch.empty(
        (i32_max_element + 2,), device="meta", dtype=torch.float16
    )

    assert not _gfx950._needs_i64_offsets(within_i32)
    assert _gfx950._needs_i64_offsets(beyond_i32)


def test_mm_output_offset_width_selection(monkeypatch):
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((grid, kwargs["USE_I64_C_OFFSETS"]))

            return launch

    monkeypatch.setattr(_gfx950, "a16w16_8wave", FakeKernel())
    for m, n in [(256, 256), (925210, 4096)]:
        a = torch.empty((m, 128), device="meta", dtype=torch.float16)
        b = torch.empty((128, n), device="meta", dtype=torch.float16)
        _gfx950._launch_lds(a, b, SPLIT_K=1, TILE=(256, 256))

    assert [use_i64_c_offsets for _, use_i64_c_offsets in launches] == [
        False,
        True,
    ]


def test_mm_input_offset_width_selection(monkeypatch):
    launches = []

    class FakeKernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append(
                    (
                        kwargs["USE_I64_A_OFFSETS"],
                        kwargs["USE_I64_B_OFFSETS"],
                        kwargs["HAS_M_TAIL"],
                        kwargs["HAS_N_TAIL"],
                    )
                )

            return launch

    monkeypatch.setattr(_gfx950, "a16w16_8wave", FakeKernel())
    cases = [
        ((256, 256, 4096), (False, False, False, False)),
        ((257, 256, 4096), (False, False, True, False)),
        ((256, 257, 4096), (False, False, False, True)),
        ((262400, 256, 4096), (True, False, False, False)),
        ((256, 262400, 4096), (False, True, False, False)),
    ]
    for (m, n, k), _ in cases:
        a = torch.empty((m, k), device="meta", dtype=torch.float16)
        b = torch.empty((k, n), device="meta", dtype=torch.float16)
        _gfx950._launch_lds(a, b, SPLIT_K=1, TILE=(256, 256))

    assert launches == [expected for _, expected in cases]


def test_mm_irregular_shape_policy():
    assert _gfx950._lds_plan_for_shape(677, 4096, 8192) == (256, 256, 4)
    assert _gfx950._strong_lds_plan(677, 4096, 8192) == (192, 256, 4)

    square_focus = _gfx950._register_plan_for_shape(2041, 2041, 2048)
    assert (
        square_focus["BLOCK_M"],
        square_focus["BLOCK_N"],
        square_focus["BLOCK_K"],
        square_focus["GROUP_M"],
        square_focus["NUM_XCDS"],
        square_focus["num_warps"],
    ) == (128, 128, 128, 16, 8, 8)

    thin_focus = _gfx950._register_plan_for_shape(2048, 256, 1024)
    assert (
        thin_focus["BLOCK_M"],
        thin_focus["BLOCK_N"],
        thin_focus["BLOCK_K"],
        thin_focus["GROUP_M"],
        thin_focus["NUM_XCDS"],
        thin_focus["num_warps"],
    ) == (128, 128, 64, 16, 1, 4)

    deep_k = _gfx950._intermediate_register_config(677, 2048, 4096)
    assert (
        deep_k["BLOCK_M"],
        deep_k["BLOCK_N"],
        deep_k["BLOCK_K"],
        deep_k["matrix_instr_nonkdim"],
        deep_k["num_warps"],
        deep_k["num_stages"],
    ) == (128, 64, 128, 32, 8, 3)

    high_padding = _gfx950._intermediate_register_config(279, 2048, 4096)
    assert (
        high_padding["BLOCK_M"],
        high_padding["BLOCK_N"],
        high_padding["matrix_instr_nonkdim"],
    ) == (64, 32, 16)


def test_mm_rejects_large_workspace():
    m, n, k = 262145, 2048, 256
    a = torch.empty((m, k), device="meta", dtype=torch.float16)
    b = torch.empty((k, n), device="meta", dtype=torch.float16)

    with pytest.raises(
        ValueError,
        match="FP32 workspace exceeds signed-i32 byte offsets",
    ):
        _gfx950._launch_lds(
            a,
            b,
            SPLIT_K=2,
            TILE=(256, 256),
        )


def test_mm_validated_register_launch_still_checks_bias():
    a = torch.empty((279, 4096), device="meta", dtype=torch.float16)
    b = torch.empty((4096, 256), device="meta", dtype=torch.float16)
    bias = torch.empty((278, 256), device="meta", dtype=torch.float16)

    with pytest.raises(ValueError, match="Bias must expand"):
        _gfx950._launch_register_plan(
            a,
            b,
            config=_gfx950._register_plan_for_shape(279, 256, 4096),
            bias=bias,
            _validated=True,
        )


@pytest.mark.parametrize("failure", ["type", "shape", "dtype", "device"])
def test_mm_rejects_invalid_output(failure):
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    a = torch.randn((7, 2048), device="cuda", dtype=torch.float16)
    b = torch.randn((8192, 2048), device="cuda", dtype=torch.float16).T
    if failure == "type":
        out, match = object(), "torch.Tensor"
    elif failure == "shape":
        out = torch.empty((7, 8191), device="cuda", dtype=torch.float16)
        match = "output shape"
    elif failure == "dtype":
        out = torch.empty((7, 8192), device="cuda", dtype=torch.float32)
        match = "output dtype"
    else:
        out = torch.empty((7, 8192), device="cpu", dtype=torch.float16)
        match = "output device"
    with pytest.raises(InvalidInput, match=match):
        matmul(a, b, out=out)


def test_mm_rejects_plan_that_does_not_cover_m(monkeypatch):
    import triton.tlx.ops.kernels.mm.gfx950 as gfx950

    a = torch.randn((17, 32), device="cuda", dtype=torch.float16)
    b = torch.randn((16, 32), device="cuda", dtype=torch.float16).T
    monkeypatch.setitem(
        gfx950._KNOWN_PLANS,
        (17, 16, 32),
        gfx950._Plan(16, 16, 2, 16, 4),
    )
    with pytest.raises(InvalidInput, match="at most 16 rows"):
        gfx950.matmul(a, b)


def test_mm_supports_unaligned_contiguous_k_views():
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    a = torch.randn((7, 2049), device="cuda", dtype=torch.float16)[:, 1:]
    b = torch.randn((8192, 2049), device="cuda", dtype=torch.float16)[:, 1:].T
    actual = matmul(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-3 * expected.abs().max().item(),
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "n,k,pattern_period,segment_k",
    [(8192, 2048, 512, 128), (2048, 4096, 1024, 256)],
)
def test_mm_matches_aten_for_cancellation(n, k, pattern_period, segment_k):
    """Exercise cancellation-sensitive ordered partial reduction."""
    from triton.tlx.ops.kernels.mm.gfx950 import matmul

    values = torch.zeros(k, device="cuda", dtype=torch.float16)
    for base in range(0, k, pattern_period):
        values[base:base + segment_k] = 65504.0
        values[base + segment_k:base + 2 * segment_k] = 0.001
        values[base + 2 * segment_k:base + 3 * segment_k] = -65504.0
        values[base + 3 * segment_k:base + 4 * segment_k] = 0.001
    a = torch.ones((7, k), device="cuda", dtype=torch.float16)
    b = values[None, :].repeat(n, 1).contiguous().T

    actual = matmul(a, b)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

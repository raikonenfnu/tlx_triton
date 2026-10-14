"""Known entry points. Agents can propose additional adapters in run artifacts."""

import importlib.util

from contracts import DENSE_ATTENTION, GEMM_OPERATIONS, REPOSITORY, attention_scale


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def candidates(backend, operation, *, search=False):
    if backend == "tlx_tutorials" and operation == "flash_attn_mxfp8":
        return []
    if backend == "aiter":
        public = {"id": "public_api", "params": {}}
        if not search:
            return [public]
        if operation in {"addmm", "bmm"}:
            return [public]
        if operation == "mm":
            return [
                public, {"id": "triton_gemm", "params": {"implementation": "triton"}},
                {"id": "triton_persistent", "params": {"implementation": "triton", "persistent": True}}
            ]
        return [
            public, {"id": "triton_mha", "params": {"backend": "triton"}},
            {"id": "gluon_mha", "params": {"backend": "gluon"}}
        ]
    if backend == "tlx_tutorials" and operation == "flash_attn_varlen":
        # No independent packed-varlen tutorial at the initial source revision.
        return []
    if backend == "tlx_tutorials" and operation in {"addmm", "bmm"}:
        return [{"id": "public_tutorial", "params": {}}]
    if backend == "tlx_tutorials" and operation == "mm":
        return [{"id": version, "params": {"version": version}}
                for version in ("v8_warp_pipeline", "v9_beyond_hotloop")]
    if backend == "tlx_tutorials":
        return [{"id": "pipelined", "params": {"prefetch": False}}, {"id": "prefetch", "params": {"prefetch": True}}]
    if backend == "pytorch" and operation not in GEMM_OPERATIONS:
        result = [{"id": "sdpa_auto", "params": {"sdpa": "auto"}}, {"id": "sdpa_math", "params": {"sdpa": "math"}}]
        if operation == "flash_attn_varlen":
            result.append({"id": "sdpa_padded", "params": {"sdpa": "auto", "padded": True}})
        return result
    return [{"id": "public_api", "params": {}}]


def build(backend, operation, case, params):
    """Return a callable over harness-owned inputs; no answers are prepared here."""
    import torch

    if backend == "tlx_ops":
        from triton.tlx import ops
        if operation == "mm":
            return lambda x: ops.mm(x["a"], x["b"], space=params.get("space", "heuristic"))
        if operation == "addmm":
            return lambda x: ops.addmm(x["bias"], x["a"], x["b"], space=params.get("space", "heuristic"))
        if operation == "bmm":
            from torch._inductor import config
            if not hasattr(config.triton, "tlx_mode"):
                raise NotImplementedError("this Torch build lacks TorchTLX BMM compiler integration")
            from triton.tlx.ops.kernels.bmm.gfx950_torch import bmm

            def torchtlx(x):
                with config.patch({"triton.cudagraphs": False}):
                    return bmm(x["a"], x["b"], mode="force")

            torchtlx.scout_metadata = {"implementation": "torchtlx_force", "cudagraphs": False}
            return torchtlx
        if operation == "flash_attn_mxfp8":

            def mxfp8(x):
                return ops.flash_attn_mxfp8(x["q"], x["k"], x["v"], causal=case["causal"],
                                            sm_scale=attention_scale(case), space=params.get("space", "full"))

            mxfp8.scout_metadata = {"implementation": "mxfp8_public_op"}
            return mxfp8
        if operation == "flash_attn":
            return lambda x: ops.flash_attn(x["q"], x["k"], x["v"], causal=case["causal"], sm_scale=attention_scale(
                case), space=params.get("space", "full"))
        return lambda x: ops.flash_attn_varlen(x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], max(case[
            "q_lengths"]), max(case["kv_lengths"]), causal=case["causal"], sm_scale=attention_scale(case))
    if backend == "tlx_tutorials":
        tutorials = REPOSITORY / "third_party/tlx/tutorials"
        if operation == "flash_attn_mxfp8":
            raise NotImplementedError("no independent gfx950 MXFP8 attention tutorial is registered")
        if operation == "addmm":
            tutorial = module(tutorials / "amd_addmm_gfx950.py", "scout_addmm_tutorial")
            return lambda x: tutorial.addmm(x["bias"], x["a"], x["b"], **params)
        if operation == "bmm":
            tutorial = module(tutorials / "amd_bmm_shared_a.py", "scout_bmm_tutorial")
            return lambda x: tutorial.bmm(x["a"], x["b"])
        if operation == "mm":
            bench = module(tutorials / "gfx9_gemm/a16w16/bench.py", "scout_gemm_tutorial")
            version = params["version"]
            bench.validate_shape_for_providers((case["m"], case["n"], case["k"]), int(version[1]), ["tlx"])
            kernel = bench.load_matmul_module(version, "scout")
            return lambda x: bench.launch_tutorial_matmul(kernel, version, x["a"], x["b"])
        tutorial = module(tutorials / "amd_fa_pipelined.py", "scout_fa_tutorial")
        config = {"BLOCK_M": 256, "BLOCK_N": 64, "num_warps": 4, "PREFETCH": params["prefetch"]}
        return lambda x: tutorial.attention(x["q"], x["k"], x["v"], attention_scale(case), case["causal"], config=config
                                            )
    if backend == "aiter":
        if operation == "bmm":
            if case["dtype"] != "bfloat16":
                raise NotImplementedError("AITER's public batched_gemm_bf16_CK accepts BF16 inputs only")
            from aiter import batched_gemm_bf16_CK

            return lambda x: batched_gemm_bf16_CK(x["a"], x["b"].transpose(1, 2).contiguous(), dtype=x["a"].dtype)
        if operation in {"mm", "addmm"}:
            if params.get("implementation") != "triton":
                from aiter.tuned_gemm import gemm_a16w16 as public_gemm
                from aiter.tuned_gemm import get_GEMM_A16W16_config
                dtype = getattr(torch, case["dtype"])
                config = get_GEMM_A16W16_config(M=case["m"], N=case["n"], K=case["k"], bias=operation == "addmm",
                                                dtype=str(dtype), otype=str(dtype), scaleAB=False, bpreshuffle=False)

                def public(x):
                    a = x["a"].contiguous() if case.get("a_layout") == "column_major" else x["a"]
                    options = {"bias": x["bias"]} if operation == "addmm" else {}
                    return public_gemm(a, x["b"].T.contiguous(), otype=x["a"].dtype, **options)

                public.scout_metadata = {
                    "implementation": str(config["libtype"]), "kernel": str(config.get("kernelName", ""))
                }
                return public
            from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
            # AITER computes X @ W.T and defaults to BF16 output.
            options = {key: value for key, value in params.items() if key != "implementation"}
            return lambda x: gemm_a16w16(x["a"], x["b"].T.contiguous(), dtype=x["a"].dtype, **options)
        if params.get("backend"):
            from aiter.ops.triton.attention.mha import flash_attn_func, flash_attn_varlen_func
        else:
            from aiter import flash_attn_func, flash_attn_varlen_func
        options = {"dropout_p": 0.0, "softmax_scale": attention_scale(case), "causal": case["causal"], **params}
        if operation in DENSE_ATTENTION:

            def dense(x):
                out = flash_attn_func(*(x[k].transpose(1, 2) for k in ("q", "k", "v")), **options)
                return out.transpose(1, 2).contiguous()

            if operation == "flash_attn_mxfp8":
                dense.scout_metadata = {"implementation": "bf16_attention_reference"}
            return dense
        return lambda x: flash_attn_varlen_func(x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], max(case["q_lengths"]),
                                                max(case["kv_lengths"]), **options)
    if backend != "pytorch":
        raise ValueError(f"no seed adapter for {backend}")
    if operation == "mm":
        return lambda x: torch.mm(x["a"], x["b"])
    if operation == "addmm":
        return lambda x: torch.addmm(x["bias"], x["a"], x["b"])
    if operation == "bmm":
        return lambda x: torch.bmm(x["a"], x["b"])
    from contextlib import nullcontext

    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def sdpa(q, k, v, ql, kl):
        # An explicit bottom-right mask also handles unequal-length causal GQA.
        mask = None
        native_causal = case["causal"] and operation in DENSE_ATTENTION
        if case["causal"] and not native_causal:
            mask = torch.arange(kl, device=q.device)[None, :] <= torch.arange(ql, device=q.device)[:, None] + kl - ql
        context = sdpa_kernel(SDPBackend.MATH) if params.get("sdpa") == "math" else nullcontext()
        with context:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=native_causal,
                                                  scale=attention_scale(case), dropout_p=0.0, enable_gqa=case["heads"]
                                                  != case["kv_heads"])

    if operation in DENSE_ATTENTION:

        def dense(x):
            return sdpa(x["q"], x["k"], x["v"], case["q_lengths"][0], case["kv_lengths"][0])

        if operation == "flash_attn_mxfp8":
            dense.scout_metadata = {"implementation": "bf16_attention_reference"}
        return dense

    def padded(x):
        ql, kl = case["q_lengths"], case["kv_lengths"]
        mq, mk = max(ql), max(kl)
        b, h, hk, d = len(ql), case["heads"], case["kv_heads"], case["head_dim"]
        qp, kp, vp = (x["q"].new_zeros(shape) for shape in ((b, h, mq, d), (b, hk, mk, d), (b, hk, mk, d)))
        qi = ki = 0
        for batch, (nq, nk) in enumerate(zip(ql, kl)):
            qp[batch, :, :nq].copy_(x["q"][qi:qi + nq].transpose(0, 1))
            kp[batch, :, :nk].copy_(x["k"][ki:ki + nk].transpose(0, 1))
            vp[batch, :, :nk].copy_(x["v"][ki:ki + nk].transpose(0, 1))
            qi += nq
            ki += nk
        q_lengths = torch.tensor(ql, device=qp.device)[:, None, None]
        k_lengths = torch.tensor(kl, device=qp.device)[:, None, None]
        q_positions = torch.arange(mq, device=qp.device)[None, :, None]
        k_positions = torch.arange(mk, device=qp.device)[None, None, :]
        mask = (q_positions < q_lengths) & (k_positions < k_lengths)
        if case["causal"]:
            mask &= k_positions <= q_positions + k_lengths - q_lengths
        result = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask[:, None], scale=attention_scale(case),
                                                dropout_p=0.0, enable_gqa=h != hk)
        return torch.cat([result[i, :, :nq].transpose(0, 1) for i, nq in enumerate(ql)])

    if params.get("padded"):
        return padded

    def packed(x):
        # Slicing, SDPA calls, and packing the result are all inside the timed call.
        outputs = []
        qi = ki = 0
        for ql, kl in zip(case["q_lengths"], case["kv_lengths"]):
            q = x["q"][qi:qi + ql].transpose(0, 1).unsqueeze(0)
            k = x["k"][ki:ki + kl].transpose(0, 1).unsqueeze(0)
            v = x["v"][ki:ki + kl].transpose(0, 1).unsqueeze(0)
            outputs.append(sdpa(q, k, v, ql, kl).squeeze(0).transpose(0, 1))
            qi += ql
            ki += kl
        return torch.cat(outputs)

    return packed

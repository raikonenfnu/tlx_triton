"""Known entry points. Agents can propose additional adapters in run artifacts."""

import importlib.util

from contracts import REPOSITORY


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def candidates(backend, operation):
    if backend == "aiter":
        if operation == "mm":
            return [{"id": "triton_gemm", "params": {}}, {"id": "triton_persistent", "params": {"persistent": True}}]
        return [{"id": "public_api", "params": {}}, {"id": "triton_mha", "params": {"backend": "triton"}},
                {"id": "gluon_mha", "params": {"backend": "gluon"}}]
    if backend == "tlx_tutorials" and operation == "flash_attn_varlen":
        # No independent packed-varlen tutorial at the initial source revision.
        return []
    if backend == "tlx_tutorials" and operation == "mm":
        return [{"id": version, "params": {"version": version}}
                for version in ("v8_warp_pipeline", "v9_beyond_hotloop")]
    if backend == "tlx_tutorials":
        return [{"id": "pipelined", "params": {"prefetch": False}}, {"id": "prefetch", "params": {"prefetch": True}}]
    if backend == "pytorch" and operation != "mm":
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
        if operation == "flash_attn":
            return lambda x: ops.flash_attn(x["q"], x["k"], x["v"], causal=case["causal"], space=params.get(
                "space", "full"))
        return lambda x: ops.flash_attn_varlen(x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], max(case["q_lengths"]),
                                               max(case["kv_lengths"]), causal=case["causal"])
    if backend == "tlx_tutorials":
        tutorials = REPOSITORY / "third_party/tlx/tutorials"
        if operation == "mm":
            bench = module(tutorials / "gfx9_gemm/a16w16/bench.py", "scout_gemm_tutorial")
            version = params["version"]
            bench.validate_shape_for_providers((case["m"], case["n"], case["k"]), int(version[1]), ["tlx"])
            kernel = bench.load_matmul_module(version, "scout")
            return lambda x: bench.launch_tutorial_matmul(kernel, version, x["a"], x["b"])
        tutorial = module(tutorials / "amd_fa_pipelined.py", "scout_fa_tutorial")
        config = {"BLOCK_M": 256, "BLOCK_N": 64, "num_warps": 4, "PREFETCH": params["prefetch"]}
        return lambda x: tutorial.attention(x["q"], x["k"], x["v"], case["head_dim"]**-0.5, case["causal"], config=
                                            config)
    if backend == "aiter":
        if operation == "mm":
            from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
            # AITER computes X @ W.T and defaults to BF16 output.
            return lambda x: gemm_a16w16(x["a"], x["b"].T.contiguous(), dtype=x["a"].dtype, **params)
        if params.get("backend"):
            from aiter.ops.triton.attention.mha import flash_attn_func, flash_attn_varlen_func
        else:
            from aiter import flash_attn_func, flash_attn_varlen_func
        options = {"dropout_p": 0.0, "softmax_scale": case["head_dim"]**-0.5, "causal": case["causal"], **params}
        if operation == "flash_attn":

            def dense(x):
                out = flash_attn_func(*(x[k].transpose(1, 2) for k in ("q", "k", "v")), **options)
                return out.transpose(1, 2).contiguous()

            return dense
        return lambda x: flash_attn_varlen_func(x["q"], x["k"], x["v"], x["cu_q"], x["cu_k"], max(case["q_lengths"]),
                                                max(case["kv_lengths"]), **options)
    if backend != "pytorch":
        raise ValueError(f"no seed adapter for {backend}")
    if operation == "mm":
        return lambda x: torch.mm(x["a"], x["b"])
    from contextlib import nullcontext

    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def sdpa(q, k, v, ql, kl):
        # An explicit bottom-right mask also handles unequal-length causal GQA.
        mask = None
        if case["causal"]:
            mask = torch.arange(kl, device=q.device)[None, :] <= torch.arange(ql, device=q.device)[:, None] + kl - ql
        context = sdpa_kernel(SDPBackend.MATH) if params.get("sdpa") == "math" else nullcontext()
        with context:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, enable_gqa=case["heads"]
                                                  != case["kv_heads"])

    if operation == "flash_attn":
        return lambda x: sdpa(x["q"], x["k"], x["v"], case["q_lengths"][0], case["kv_lengths"][0])

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
        result = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask[:, None], dropout_p=0.0, enable_gqa=h != hk)
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

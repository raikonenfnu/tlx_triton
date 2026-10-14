"""The harness alone defines input data and the FP32 correctness oracle."""

import ctypes
import hashlib
import itertools
import json

import torch
from contracts import DENSE_ATTENTION, GEMM_OPERATIONS, attention_scale


def input_digest(inputs):
    """Check actual tensor bits and strides across independently configured runtimes."""
    hasher = hashlib.sha256()
    for name, value in sorted(inputs.items()):
        metadata = (name, str(value.dtype), tuple(value.shape), value.stride())
        hasher.update(json.dumps(metadata).encode())
        unique = value.detach()
        # An expanded batch has identical bits in every zero-stride slice.
        for dim, stride in enumerate(value.stride()):
            if stride == 0 and value.shape[dim] > 1:
                unique = unique.narrow(dim, 0, 1)
        dense = unique.contiguous().cpu()
        hasher.update(ctypes.string_at(dense.data_ptr(), dense.numel() * dense.element_size()))
    return hasher.hexdigest()


def relative_rms_error(actual, expected):
    numerator = torch.linalg.vector_norm(actual.float() - expected).item()
    denominator = torch.linalg.vector_norm(expected).item()
    return numerator / denominator if denominator else (0.0 if numerator == 0 else float("inf"))


def make_inputs(case, operation, seed, *, device="cuda"):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = getattr(torch, case["dtype"])

    def random(shape, std=1.0):
        return (torch.randn(shape, generator=generator, dtype=torch.float32) * std).to(device=device, dtype=dtype)

    def matrix(rows, cols, layout, std, batch=()):
        return (random((*batch, cols, rows), std).transpose(-1, -2) if layout == "column_major" else random(
            (*batch, rows, cols), std))

    if operation in GEMM_OPERATIONS:
        batch = (case["batch"], ) if operation == "bmm" else ()
        shared = operation == "bmm" and case["shared_a"]
        a = matrix(case["m"], case["k"], case.get("a_layout", "row_major"), case.get("a_std", 1.0),
                   () if shared else batch)
        if shared:
            a = a.unsqueeze(0).expand(case["batch"], -1, -1)
        inputs = {"a": a, "b": matrix(case["k"], case["n"], case["b_layout"], case.get("b_std", 1.0), batch)}
        if operation == "addmm":
            inputs["bias"] = random((case["n"], ), case.get("bias_std", 1.0))
        return inputs
    ql, kl = case["q_lengths"], case["kv_lengths"]
    h, hk, d = (case[key] for key in ("heads", "kv_heads", "head_dim"))
    std = case.get("qkv_std", 1.0)
    if operation in DENSE_ATTENTION:
        return {
            "q": random((len(ql), h, ql[0], d), std), "k": random((len(kl), hk, kl[0], d), std), "v": random(
                (len(kl), hk, kl[0], d), std)
        }
    return {
        "q": random((sum(ql), h, d), std), "k": random((sum(kl), hk, d), std), "v": random((sum(kl), hk, d), std),
        "cu_q": torch.tensor([0, *itertools.accumulate(ql)], dtype=torch.int32, device=device), "cu_k":
        torch.tensor([0, *itertools.accumulate(kl)], dtype=torch.int32, device=device)
    }


def reference(case, operation, inputs):
    if operation in GEMM_OPERATIONS:
        output = inputs["a"].float() @ inputs["b"].float()
        return output + inputs["bias"].float() if operation == "addmm" else output
    scale = attention_scale(case)
    group = case["heads"] // case["kv_heads"]
    outputs = []
    qi = ki = 0
    for batch, (ql, kl) in enumerate(zip(case["q_lengths"], case["kv_lengths"])):
        if operation == "flash_attn_varlen":
            q = inputs["q"][qi:qi + ql].transpose(0, 1)
            k = inputs["k"][ki:ki + kl].transpose(0, 1)
            v = inputs["v"][ki:ki + kl].transpose(0, 1)
        else:
            q, k, v = (inputs[key][batch] for key in ("q", "k", "v"))
        qi += ql
        ki += kl
        k, v = (x.float().repeat_interleave(group, dim=0) for x in (k, v))
        if kl == 0:
            out = torch.zeros_like(q, dtype=torch.float32)
        else:
            scores = (q.float() @ k.transpose(-1, -2)) * scale
            if case["causal"]:
                visible = torch.arange(kl,
                                       device=q.device)[None, :] <= torch.arange(ql, device=q.device)[:, None] + kl - ql
                scores = scores.masked_fill(~visible, -torch.inf)
            probabilities = torch.softmax(scores, dim=-1).nan_to_num(0)
            out = probabilities @ v
        outputs.append(out.transpose(0, 1) if operation == "flash_attn_varlen" else out)
    return torch.cat(outputs) if operation == "flash_attn_varlen" else torch.stack(outputs)

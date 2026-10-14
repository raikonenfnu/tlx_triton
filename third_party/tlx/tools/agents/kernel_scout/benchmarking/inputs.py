"""The harness alone defines input data and the FP32 correctness oracle."""

import ctypes
import hashlib
import itertools
import json

import torch


def input_digest(inputs):
    """Check actual tensor bits and strides across independently configured runtimes."""
    hasher = hashlib.sha256()
    for name, value in sorted(inputs.items()):
        metadata = (name, str(value.dtype), tuple(value.shape), value.stride())
        hasher.update(json.dumps(metadata).encode())
        dense = value.detach().contiguous().cpu()
        hasher.update(ctypes.string_at(dense.data_ptr(), dense.numel() * dense.element_size()))
    return hasher.hexdigest()


def make_inputs(case, operation, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = getattr(torch, case["dtype"])

    def random(shape):
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(device="cuda", dtype=dtype)

    if operation == "mm":
        a = random((case["m"], case["k"]))
        b = random((case["n"], case["k"])).T if case["b_layout"] == "column_major" else random((case["k"], case["n"]))
        return {"a": a, "b": b}
    ql, kl = case["q_lengths"], case["kv_lengths"]
    h, hk, d = (case[key] for key in ("heads", "kv_heads", "head_dim"))
    if operation == "flash_attn":
        return {
            "q": random((len(ql), h, ql[0], d)), "k": random((len(kl), hk, kl[0], d)), "v": random(
                (len(kl), hk, kl[0], d))
        }
    return {
        "q": random((sum(ql), h, d)), "k": random((sum(kl), hk, d)), "v": random(
            (sum(kl), hk, d)), "cu_q": torch.tensor([0, *itertools.accumulate(ql)], dtype=torch.int32, device="cuda"),
        "cu_k": torch.tensor([0, *itertools.accumulate(kl)], dtype=torch.int32, device="cuda")
    }


def reference(case, operation, inputs):
    if operation == "mm":
        return inputs["a"].float() @ inputs["b"].float()
    scale = case["head_dim"]**-0.5
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
                visible = torch.arange(kl, device="cuda")[None, :] <= torch.arange(ql, device="cuda")[:, None] + kl - ql
                scores = scores.masked_fill(~visible, -torch.inf)
            probabilities = torch.softmax(scores, dim=-1).nan_to_num(0)
            out = probabilities @ v
        outputs.append(out.transpose(0, 1) if operation == "flash_attn_varlen" else out)
    return torch.cat(outputs) if operation == "flash_attn_varlen" else torch.stack(outputs)

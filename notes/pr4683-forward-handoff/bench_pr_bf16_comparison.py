"""PR-description measurement: each comparison uses two alternating runners."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import torch
from bench_common import ownership

R = Path(__file__).resolve().parent
P = R.parents[1] / 'third_party/tlx/tutorials/amd_mxfp8_varlen_attention'
sys.path.insert(0, str(P))
import bench_prequantized_forward as b
h = b.harness


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--family', choices=['short', 'long'], required=True)
    p.add_argument('--kfd-id', type=int, required=True)
    p.add_argument('--expected-pci', required=True)
    p.add_argument('--json', type=Path, required=True)
    a = p.parse_args()
    foreign = [x for x in ownership() if x['kfd_id'] == a.kfd_id]
    if foreign:
        raise RuntimeError(f'GPU occupied: {foreign}')
    torch.set_num_threads(8)
    hip = ctypes.CDLL(str(Path(torch.__file__).parent / 'lib/libamdhip64.so'))
    pci = ctypes.create_string_buffer(64)
    assert hip.hipDeviceGetPCIBusId(pci, len(pci), 0) == 0
    expected = a.expected_pci.lower()
    assert pci.value.decode().lower() == expected
    base, bh = b.load_forward(R/'baseline/amd_mxfp8_varlen_attention', '_bf16_compare_base')
    cand, ch = b.load_forward(P, '_bf16_compare_candidate')
    from aiter.jit.core import get_module
    native = get_module('module_fmha_v3_varlen_fwd')
    flusher = h.L2Flusher(256)
    report = {'completed': False, 'pci': expected, 'baseline_hashes': bh, 'candidate_hashes': ch,
              'protocol': {'samples': 50, 'rounds': 3, 'warmups': 10, 'flush_MiB': 256,
                           'seed': 7, 'block_m': 128, 'num_ctas': 512,
                           'pairwise': True, 'schedule_construction_excluded': True}, 'results': []}
    import time
    for case in h.PRODUCTION_SHAPES:
        if case.family != a.family:
            continue
        if any(x['kfd_id'] == a.kfd_id for x in ownership()):
            raise RuntimeError('GPU became occupied')
        torch.manual_seed(7)
        ql = [16*x for x in h.lengths(432, case.total_q//16, case.max_q//16, 3)]
        kl = h.lengths(432, case.total_kv, case.max_kv, 5)
        cq, ck = h.cumulative(ql), h.cumulative(kl)
        q = torch.randn((case.total_q,4,128),device='cuda',dtype=torch.bfloat16)
        k = torch.randn((case.total_kv,4,128),device='cuda',dtype=torch.bfloat16)
        v = torch.randn_like(k)
        start = time.perf_counter()
        schedule = cand.prepare_work_schedule(ql,kl,q.device)
        torch.cuda.synchronize()
        schedule_wall_ms = (time.perf_counter()-start)*1000
        fast = case.family == 'long'
        def baseline_bf16():
            return base.forward(q,k,v,cq,ck,h.SOFTMAX_SCALE,block_m=128,num_ctas=512,fast_exp=False)[:2]
        def candidate_bf16():
            return cand.forward(q,k,v,cq,ck,h.SOFTMAX_SCALE,block_m=128,num_ctas=512,fast_exp=fast,work_schedule=schedule)[:2]
        def aiter_bf16():
            return native.fmha_v3_varlen_fwd(q,k,v,cq,ck,case.max_q,case.max_kv,0,0.,h.SOFTMAX_SCALE,0.,False,False,-1,-1,True,False,0)[:2]
        meta = cand._metadata(cq,ck,case.total_q,case.total_kv,128)
        state = cand.prepare_kv(k,v,ck,meta)
        q8, qs = cand.prepare_q(q,cq)
        def prepared():
            return cand.forward_prequantized(q8,qs,*state,cq,ck,h.SOFTMAX_SCALE,total_kv=case.total_kv,fast_exp=fast,work_schedule=schedule)
        for f in [baseline_bf16,candidate_bf16,aiter_bf16,prepared]:
            f.name = f.__name__
        bo,bl = baseline_bf16()
        co,cl = candidate_bf16()
        po,pl = prepared()
        ao,al = aiter_bf16()
        err = {'candidate_vs_prepared_bitwise': torch.equal(co,po) and torch.equal(cl,pl),
               'candidate_vs_baseline_bitwise': torch.equal(co,bo) and torch.equal(cl,bl),
               'candidate_aiter_output_relative_l2': float((co.float()-ao.float()).norm()/ao.float().norm()),
               'candidate_aiter_lse_max_abs': float((cl-al).abs().max())}
        assert err['candidate_vs_prepared_bitwise'], (case.name,err)
        assert case.family != 'short' or err['candidate_vs_baseline_bitwise']
        del bo,bl,co,cl,po,pl,ao,al
        pairs = {}
        for label, runners in [('before_after_bf16',[baseline_bf16,candidate_bf16]),
                               ('bf16_vs_prepared',[candidate_bf16,prepared]),
                               ('bf16_vs_aiter',[candidate_bf16,aiter_bf16])]:
            pairs[label] = h.benchmark_interleaved(runners,flusher,10,50,3)
        row = {'case':case.name,'family':case.family,'correctness':err,'pairs':pairs,
               'schedule_first_construction_upload_sync_wall_ms':schedule_wall_ms}
        report['results'].append(row)
        a.json.write_text(json.dumps(report,indent=2)+'\n')
        print(case.name, {label:{name:round(d['timing']['median_ms']*1000,1) for name,d in pair.items()} for label,pair in pairs.items()},flush=True)
        del q,k,v,state,q8,qs,schedule
        torch.cuda.empty_cache()
    report['completed'] = True
    a.json.write_text(json.dumps(report,indent=2)+'\n')

if __name__ == '__main__':
    main()

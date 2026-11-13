"""Separate prepared-Q effect from subsequent forward optimizations."""
import argparse
import ctypes
import json
from pathlib import Path
import sys
import torch
from bench_common import ownership

R = Path(__file__).resolve().parent
P = R.parents[1]/'third_party/tlx/tutorials/amd_mxfp8_varlen_attention'
sys.path.insert(0,str(P))
import bench_prequantized_forward as b
h = b.harness


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['short','long'],required=True)
    p.add_argument('--kfd-id',type=int,required=True)
    p.add_argument('--expected-pci',required=True)
    p.add_argument('--json',type=Path,required=True)
    p.add_argument('--skip-q-only',action='store_true')
    a=p.parse_args()
    def check():
        foreign=[x for x in ownership() if x['kfd_id']==a.kfd_id]
        if foreign: raise RuntimeError(f'GPU occupied: {foreign}')
    check()
    torch.set_num_threads(8)
    hip=ctypes.CDLL(str(Path(torch.__file__).parent/'lib/libamdhip64.so'))
    pci=ctypes.create_string_buffer(64)
    if hip.hipDeviceGetPCIBusId(pci,len(pci),0): raise RuntimeError('PCI lookup failed')
    expected=a.expected_pci.lower()
    if pci.value.decode().lower()!=expected: raise RuntimeError(f'Wrong GPU: {pci.value}')
    base,bh=b.load_forward(R/'experiments/base','_prepared_q_only_baseline')
    cand,ch=b.load_forward(P,'_prepared_after_opt')
    report={'completed':False,'pci':expected,'baseline_hashes':bh,'candidate_hashes':ch,
            'baseline':'PR4683 plus prepared-Q API and LSE fix; no wait overlap, cost schedule or metadata/fill optimization',
            'protocol':{'warmups':15,'samples':100,'rounds':3,'seed':7,'flush_MiB':256,
                        'BM':128,'num_ctas':512,'fast_exp':'false for short, true for long on every runner',
                        'pairwise':True,'operand_preparation_and_schedule_excluded':True},'results':[]}
    flusher=h.L2Flusher(256)
    for case in h.PRODUCTION_SHAPES:
        if case.family!=a.family:continue
        check(); torch.manual_seed(7)
        ql=[16*x for x in h.lengths(432,case.total_q//16,case.max_q//16,3)]
        kl=h.lengths(432,case.total_kv,case.max_kv,5)
        cq,ck=h.cumulative(ql),h.cumulative(kl)
        q=torch.randn((case.total_q,4,128),device='cuda',dtype=torch.bfloat16)
        k=torch.randn((case.total_kv,4,128),device='cuda',dtype=torch.bfloat16)
        v=torch.randn_like(k)
        state=base.prepare_kv(k,v,ck,base._metadata(cq,ck,case.total_q,case.total_kv,128))
        q8,qs=base.prepare_q(q,cq)
        schedule=cand.prepare_work_schedule(ql,kl,q.device)
        fast=case.family=='long'
        def bf16_q():
            return base.forward(q,k,v,cq,ck,h.SOFTMAX_SCALE,kv_state=state,block_m=128,num_ctas=512,fast_exp=fast)[:2]
        def prepared_baseline():
            return base.forward_prequantized(q8,qs,*state,cq,ck,h.SOFTMAX_SCALE,total_kv=case.total_kv,block_m=128,num_ctas=512,fast_exp=fast)
        def after_opt():
            return cand.forward_prequantized(q8,qs,*state,cq,ck,h.SOFTMAX_SCALE,total_kv=case.total_kv,block_m=128,num_ctas=512,fast_exp=fast,work_schedule=schedule)
        for fn in [bf16_q,prepared_baseline,after_opt]:fn.name=fn.__name__
        o,l=bf16_q()
        for fn in [prepared_baseline,after_opt]:
            oo,ll=fn()
            if not torch.equal(o,oo) or not torch.equal(l,ll):raise RuntimeError(f'Numerical change {case.name}/{fn.name}')
        del o,l,oo,ll
        pairs={}
        for label,runners in [('q_only',[bf16_q,prepared_baseline]),('optimization',[prepared_baseline,after_opt])]:
            if label=='q_only' and a.skip_q_only:continue
            pairs[label]=h.benchmark_interleaved(runners,flusher,15,100,3)
        kvbytes=4*(2*128+8)*sum(b.triton.cdiv(nq,128)*b.triton.cdiv(nk,256)*256 for nq,nk in zip(ql,kl) if nk)
        row={'case':case.name,'family':case.family,'shape':vars(case),'kv_bytes':kvbytes,
             'all_output_lse_bitwise_equal':True,'pairs':pairs}
        report['results'].append(row)
        a.json.write_text(json.dumps(report,indent=2)+'\n')
        print(case.name,{label:{name:round(x['timing']['median_ms']*1000,2) for name,x in pair.items()} for label,pair in pairs.items()},flush=True)
        del q,k,v,q8,qs,state,schedule
        torch.cuda.empty_cache()
    report['completed']=True
    a.json.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()

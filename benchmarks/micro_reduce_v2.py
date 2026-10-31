"""Thirty strict recurrence oracles, with explicit fallback visibility."""
import argparse, pathlib, os, time, sys, json, fcntl
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--strategy',choices=['segmented','generic','direct-state','fused-reduce-v2'],required=True)
p.add_argument('--workdir',type=pathlib.Path,required=True)
a=p.parse_args();deadline=float(os.environ['C500_DEADLINE'])
assert time.time()<deadline-60
repo=pathlib.Path(__file__).resolve().parent.parent;root=a.workdir.resolve();(root/'results').mkdir(parents=True,exist_ok=True)
lock=(root/'gpu-work.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (root/'results'/('micro-'+a.strategy+'.json')).exists(),'Use a fresh isolated run; do not overwrite oracle evidence'
folder='experimental/fused_reduce_v2' if a.strategy=='fused-reduce-v2' else ('experimental/direct_state' if a.strategy=='direct-state' else 'strategies/'+a.strategy)
sys.path.insert(0,str(repo/folder))
if a.strategy=='fused-reduce-v2':import exact_gdn_reduce_v2 as e
elif a.strategy=='direct-state':import exact_gdn_inplace as e
else:import exact_gdn as e
import torch
assert torch.cuda.device_count()==1 and 'C500' in torch.cuda.get_device_name(0)
torch.set_num_threads(8);torch.manual_seed(923);rows=[]
with torch.inference_mode():
 for b in [1,3,8,17,48]:
  for layout in ['row','col','head_fast']:
   for norm in [False,True]:
    assert time.time()<deadline-10
    q=torch.randn(b,1,48,128,device='cuda:0',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q)
    g=-torch.rand(b,1,48,device='cuda:0')*3;beta=torch.rand(b,1,48,device='cuda:0',dtype=torch.bfloat16)
    h=torch.randn(b,48,128,128,device='cuda:0')
    if layout=='col':h=h.transpose(-2,-1)
    if layout=='head_fast':h=h.permute(0,2,3,1).contiguous().permute(0,3,1,2)
    before=dict(e.COUNTS);ref,hs=e.NATIVE(q,k,v,g,beta,h,True,norm)
    kw={'mutate_state':True} if a.strategy=='direct-state' else {}
    o,s=e.fused(q,k,v,g,beta,h.clone(),True,norm,**kw)
    counts={k:e.COUNTS[k]-before[k] for k in before}
    row={'batch':b,'norm':norm,'layout':layout,'state_stride':list(h.stride()),'output_exact':torch.equal(ref,o),'state_exact':torch.equal(hs,s),'returned_stride_exact':hs.stride()==s.stride(),'finite':bool(torch.isfinite(o).all() and torch.isfinite(s).all()),'dispatch_counts':counts,'actually_fused':counts['fallback']==0 and sum(counts.values())>0}
    rows.append(row);print('ORACLE',json.dumps(row),flush=True)
    dest=root/'results'/('micro-'+a.strategy+'.json');tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2)+'\n');tmp.replace(dest)
    assert row['output_exact'] and row['state_exact'] and row['finite']
    assert row['actually_fused'] or (a.strategy=='segmented' and layout=='head_fast') or (a.strategy=='fused-reduce-v2' and layout!='row'),'Unexpected fallback'
    del q,k,v,g,beta,h,ref,hs,o,s
assert len(rows)==30

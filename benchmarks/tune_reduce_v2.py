"""Strict micro configuration search; timings are recurrence-only, not model TPS."""
import pathlib,sys,os,time,json,gc,statistics,fcntl
import torch
R=pathlib.Path(sys.argv[1]);D=float(os.environ['C500_DEADLINE'])
lock=(R/'gpu-work.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent/'experimental/fused_reduce_v2'))
import exact_gdn_reduce_v2 as e
torch.set_num_threads(8);rows=[]
with torch.inference_mode():
 for b in [1,8,48]:
  torch.manual_seed(1901+b)
  q=torch.randn(b,1,48,128,device='cuda',dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn_like(q)
  g=-torch.rand(b,1,48,device='cuda')*3;beta=torch.rand(b,1,48,device='cuda',dtype=torch.bfloat16);h=torch.randn(b,48,128,128,device='cuda')
  rr,hh=e.NATIVE(q,k,v,g,beta,h,True,True)
  for bv in [32,64,128]:
   for warps in [4,8]:
    assert time.time()<D-200
    e.BLOCK=bv;e.WARPS=warps
    o,s=e.fused(q,k,v,g,beta,h,True,True)
    row={'batch':b,'block_v':bv,'warps':warps,'output_exact':torch.equal(o,rr),'state_exact':torch.equal(s,hh),'scope':'micro recurrence CUDA graph; no model TPS'}
    if row['output_exact'] and row['state_exact']:
     del o,s
     graph=torch.cuda.CUDAGraph()
     torch.cuda.synchronize()
     with torch.cuda.graph(graph):
      for _ in range(10):o,s=e.fused(q,k,v,g,beta,h,True,True)
     graph.replay();torch.cuda.synchronize();times=[]
     for repeat in range(3):
      a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record()
      for _ in range(10):graph.replay()
      z.record();z.synchronize();times.append(a.elapsed_time(z)/100)
     row.update(repeat_ms=times,median_ms=statistics.median(times))
     del graph,o,s;gc.collect();torch.cuda.empty_cache()
    rows.append(row);(R/'results/reduce-config-search-v2.json').write_text(json.dumps(rows,indent=2)+'\n');print('CONFIG',json.dumps(row),flush=True)
  del q,k,v,g,beta,h,rr,hh;gc.collect();torch.cuda.empty_cache()
 valid=[r for r in rows if r['batch']==48 and r.get('median_ms') and all(z['output_exact'] and z['state_exact'] for z in rows if (z['block_v'],z['warps'])==(r['block_v'],r['warps']))]
 best=min(valid,key=lambda r:r['median_ms'])
 selected={k:best[k] for k in ['block_v','warps']};selected.update(selection_policy='fixed configuration minimizing B48 micro median among configs exact at B1/B8/B48; no per-cell model cherry-picking',scope='configuration only; not whole-model result',trials=len(rows))
 (R/'results/reduce-selected-config-v2.json').write_text(json.dumps(selected,indent=2)+'\n');print('SELECTED',json.dumps(selected),flush=True)

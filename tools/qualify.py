"""Fail-closed exactness admission and paired n>=3 aggregation; CPU-only."""
import math,statistics
FIELDS={'key_cache','value_cache','conv_states','recurrent_states'}

def gate_pass(g):
 return (g.get('logits_finite') is True and g.get('full_vocab_prefill_logits_exact') is True and g.get('full_vocab_decode_logits_exact') is True and g.get('complete_ids_exact') is True and set(g.get('cache_exact',{}))==FIELDS and all(v is True for v in g['cache_exact'].values()) and len(g.get('fresh_request_probes',[]))==2 and all(all(p.get(k) is True for k in ['complete_ids_exact','full_vocab_logits_exact','all_caches_exact','all_logits_finite']) for p in g['fresh_request_probes']))

def summarize(rows,gates,warmups,candidate):
 assert len(gates)==3 and all(gate_pass(g) for g in gates),'Holdout or fresh-request gate missing/failed'
 assert {(g['batch'],g['input_len']) for g in gates}=={(1,13),(3,127),(17,32)}
 out=[]
 for b in sorted({r['batch'] for r in rows}):
  modes={m:[r for r in rows if r['batch']==b and r['mode']==m] for m in ['native',candidate]}
  assert all(len(v)>=3 and len({r['repeat'] for r in v})==len(v) for v in modes.values()),'At least3 distinct repeats per mode'
  assert {r['repeat'] for r in modes['native']}=={r['repeat'] for r in modes[candidate]},'Paired repetition indices differ'
  assert all(any(w['mode']==m and w['batch']==b and w['raw_full_vocab_finite_calls']==128 for w in warmups) for m in modes),'Both128-step finite warmups required'
  group=sum(modes.values(),[])
  assert len({r['input_sha256'] for r in group})==len({r['complete_output_sha256'] for r in group})==1,'Different inputs or complete outputs'
  assert all(r['input_len']==r['output_len']==128 and r['output_tokens']==b*128 and r['all_ids_exact_vs_warmup'] is True and r['detokenized_text_exact_vs_warmup'] is True for r in group)
  assert all(math.isfinite(r['e2e_s']) and r['e2e_s']>0 and math.isclose(r['e2e_output_tokens_per_s'],b*128/r['e2e_s'],rel_tol=1e-12) for r in group)
  assert all(r['dispatch_counts'].get('fallback')==0 and sum(v for k,v in r['dispatch_counts'].items() if k!='fallback')>0 for r in modes[candidate]),'Actual fused dispatch required'
  total=lambda r:sum(v for k,v in r['dispatch_counts'].items() if k!='fallback')
  assert min(map(total,modes[candidate]))>max(map(total,modes['native'])),'No new fused dispatch during graph warmup/capture'
  med={m:statistics.median(r['e2e_output_tokens_per_s'] for r in v) for m,v in modes.items()}
  out.append({'batch':b,'native_tps':med['native'],'candidate_tps':med[candidate],'speedup':med[candidate]/med['native'],'repeats':{m:len(v) for m,v in modes.items()},'latency_median_s':{m:statistics.median(r['e2e_s'] for r in v) for m,v in modes.items()},'tps_min_max':{m:[min(r['e2e_output_tokens_per_s'] for r in v),max(r['e2e_output_tokens_per_s'] for r in v)] for m,v in modes.items()},'candidate':candidate,'scope':'128->128 local request E2E, not HTTP/coldstart','peak_allocated_gib':{m:max(r['peak_allocated_gib'] for r in v) for m,v in modes.items()}})
 return out

if __name__=='__main__':
 import argparse,pathlib,json
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--results',type=pathlib.Path,required=True)
 p.add_argument('--candidate',required=True,choices=['segmented','direct-state'])
 p.add_argument('--output',type=pathlib.Path,required=True)
 a=p.parse_args()
 def read(n):return json.loads((a.results/n).read_text())
 result=summarize(read('runs.json'),read('holdouts.json'),read('warmups.json'),a.candidate)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

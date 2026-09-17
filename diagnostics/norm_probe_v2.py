"""Isolated official GemmaRMSNorm sub-op native versus dispatch; no model TPS."""
import pathlib,json,time,os
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workdir',required=True,type=pathlib.Path);args=p.parse_args()
ROOT=args.workdir.resolve();R=ROOT/'results/norm-probe';assert R.is_dir()
assert time.time()<float(os.environ['C500_DEADLINE'])-120
import torch
import vllm
from vllm.plugins import load_general_plugins
load_general_plugins()
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
f=torch.load(R/'fixture.private.pt',map_location='cpu',weights_only=True);x=f['input'].to('cuda:0')
from vllm.config import VllmConfig,set_current_vllm_config
with set_current_vllm_config(VllmConfig()):
 norm=GemmaRMSNorm(x.shape[-1],f['epsilon']).to('cuda:0',dtype=torch.bfloat16)
 with torch.no_grad():
  norm.weight.copy_(f['weight'].to('cuda:0'))
  native=norm.forward_native(x).cpu()
  dispatched=norm(x).cpu()
  repeated=norm(x).cpu()
raw={'native':native,'dispatched':dispatched,'repeated':repeated};torch.save(raw,R/'candidate.private.pt')
rows={}
for label,y in raw.items():rows[label]={'exact_vs_HF_fixture':torch.equal(y,f['native_output']),'finite':bool(torch.isfinite(y).all()),'mismatch_count':int((y!=f['native_output']).sum()),'max_abs_error':float((y.float()-f['native_output'].float()).abs().max())}
result={'shape':list(x.shape),'dtype':str(x.dtype),'class':type(norm).__name__,'forward_method':str(norm._forward_method),'internal_is_compiled':bool(getattr(norm,'_is_compiled',False)),'rows':rows,'dispatch_repeated_exact':torch.equal(dispatched,repeated),'scope':'single actual first-layer RMSNorm input, not whole-model first-divergence proof; checkpoint weights stay only in private isolated run; no model throughput','finding':'Engine enforce_eager does not by itself disable every per-operator torch.compile: inspect GemmaRMSNorm.forward_cuda.'}
(R/'comparison-v2.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

"""Tiny real-token first-layer RMSNorm fixture; never publishes checkpoint weights."""
import pathlib,json,time,os,hashlib
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=pathlib.Path);p.add_argument('--workdir',required=True,type=pathlib.Path);args=p.parse_args()
ROOT=args.workdir.resolve();R=ROOT/'results/norm-probe';R.mkdir()
assert time.time()<float(os.environ['C500_DEADLINE'])-180
import torch
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
MODEL=args.model.resolve()
ref=torch.load(ROOT/'results/hf-reference/holdout-b1-l13.pt',map_location='cpu',weights_only=True);ids=ref['input_ids'][0]
index=json.load(open(MODEL/'model.safetensors.index.json'))['weight_map'];config=json.load(open(MODEL/'config.json'))['text_config']
ekey='model.language_model.embed_tokens.weight';wkey='model.language_model.layers.0.input_layernorm.weight'
with safe_open(str(MODEL/index[ekey]),framework='pt',device='cpu') as f:
 sl=f.get_slice(ekey);x=torch.cat([sl[int(i):int(i)+1] for i in ids])
with safe_open(str(MODEL/index[wkey]),framework='pt',device='cpu') as f:w=f.get_tensor(wkey)
assert x.dtype==w.dtype==torch.bfloat16
norm=Qwen3_5RMSNorm(config['hidden_size'],config['rms_norm_eps']).to('cuda:0',dtype=torch.bfloat16)
with torch.no_grad():norm.weight.copy_(w.to('cuda:0'));y=norm(x.to('cuda:0')).cpu()
torch.save({'input':x,'weight':w,'native_output':y,'epsilon':config['rms_norm_eps']},R/'fixture.private.pt')
(R/'fixture-metadata.json').write_text(json.dumps({'shape':list(x.shape),'dtype':str(x.dtype),'source':'actual B1L13 holdout token embedding and first input layernorm','checkpoint_weights_in_repo':False,'finite':bool(torch.isfinite(y).all()),'no_model_throughput':True},indent=2)+'\n')
print('NORM_FIXTURE_READY',x.shape)

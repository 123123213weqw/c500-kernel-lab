"""Fresh native BF16 reference. Exact logits + cache SHA; no performance claim."""
import pathlib,json,time,os,hashlib,traceback,gc,fcntl
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=pathlib.Path);p.add_argument('--workdir',required=True,type=pathlib.Path);args=p.parse_args()
ROOT=args.workdir.resolve();(ROOT/'results').mkdir(parents=True,exist_ok=True);R=ROOT/'results/hf-reference';R.mkdir()
DEADLINE=float(os.environ['C500_DEADLINE']);assert time.time()<DEADLINE-600
lock=(ROOT/'gpu-work.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
import torch,numpy as np
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
assert torch.cuda.device_count()==1 and 'C500' in torch.cuda.get_device_name(0)
torch.set_num_threads(8);torch.manual_seed(42)
MODEL=str(args.model.resolve())
def dump(n,x):(R/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def digest(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
try:
 model,info=m.Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True,output_loading_info=True)
 assert not info['missing_keys'];model.eval();cfg=model.config.get_text_config();tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 dump('metadata.json',{'model':'Qwen/Qwen3.5-27B','parameters':sum(p.numel() for p in model.parameters()),'dtype':'BF16 weights, native FP32 recurrent state','attn_implementation':'eager','torch_version':torch.__version__,'generation_config':model.generation_config.to_dict(),'loading':info,'deadline':DEADLINE})
 eos=model.generation_config.eos_token_id
 if isinstance(eos,int):eos=[eos]
 results=[]
 with torch.inference_mode():
  allowed=np.array([i for i in range(100,cfg.vocab_size) if i not in set(tok.all_special_ids)])
  for b,l in [(1,13),(3,127),(17,32)]:
   assert time.time()<DEADLINE-300
   x=torch.tensor(np.random.default_rng(993+b).choice(allowed,size=(b,l)),device='cuda')
   r=model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=m.Qwen3_5DynamicCache(cfg),do_sample=False,min_new_tokens=4,max_new_tokens=4,pad_token_id=tok.eos_token_id,logits_to_keep=1,return_dict_in_generate=True,output_logits=True)
   assert len(r.logits)==4 and all(bool(torch.isfinite(z).all()) for z in r.logits)
   cached={}
   for f in ['key_cache','value_cache','conv_states','recurrent_states']:
    cached[f]=[None if z is None else {'shape':list(z.shape),'stride':list(z.stride()),'dtype':str(z.dtype),'sha256':digest(z),'finite':bool(torch.isfinite(z).all())} for z in getattr(r.past_key_values,f)]
    assert all(z is None or z['finite'] for z in cached[f])
   p=R/f'holdout-b{b}-l{l}.pt'
   torch.save({'input_ids':x.cpu(),'sequences':r.sequences.cpu(),'raw_full_vocab_logits':[z.float().cpu() for z in r.logits],'cache_digests':cached},p)
   results.append({'batch':b,'input_len':l,'output_len':4,'file':p.name,'input_sha256':digest(x),'complete_output_sha256':digest(r.sequences),'all_logits_and_caches_finite':True,'cache_digests':cached})
   dump('holdouts.json',results);print('GOLDEN',b,l,p.stat().st_size,flush=True)
   del x,r,cached;gc.collect();torch.cuda.empty_cache()
 dump('completion.json',{'complete':True,'time_unix':time.time()})
except Exception as e:dump('failure.json',{'error':repr(e),'traceback':traceback.format_exc()});raise

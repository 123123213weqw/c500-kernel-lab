"""Diagnose long-request direct-state failure; no candidate throughput admission."""
import pathlib,time,os,json,sys,gc,traceback
import argparse
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--workdir',required=True,type=pathlib.Path);args=p.parse_args()
ROOT=args.workdir.resolve();R=ROOT/'results/long-graph-probe';R.mkdir()
D=float(os.environ['C500_DEADLINE']);assert time.time()<D-500
REPO=pathlib.Path(__file__).resolve().parent.parent
sys.path[:0]=[str(REPO/'experimental/direct_state'),str(REPO/'strategies/segmented')]
import torch,exact_gdn_inplace as e,direct_state_graph_engine as g
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
from transformers import AutoTokenizer
MODEL=str(pathlib.Path(args.model).resolve())
torch.set_num_threads(8);torch.manual_seed(42)
def dump(n,v):(R/n).write_text(json.dumps(v,indent=2,default=str)+'\n')
def cache_cpu(c):return {f:[None if z is None else z.cpu() for z in getattr(c,f)] for f in g.FIELDS}
def caches_equal(c,ref):return {f:all((x is None and y is None) or (x is not None and y is not None and torch.equal(x.cpu(),y)) for x,y in zip(getattr(c,f),ref[f])) for f in g.FIELDS}
try:
 model,info=m.Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True,output_loading_info=True);assert not info['missing_keys'];model.eval();cfg=model.config.get_text_config();tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 original=json.load(open(ROOT/'results/direct-state/warm-output-b1.json'));x=torch.tensor(original['input_ids'],device='cuda:0');b,l=x.shape
 eos=model.generation_config.eos_token_id
 if isinstance(eos,int):eos=[eos]
 mask=torch.zeros(cfg.vocab_size,device='cuda:0',dtype=torch.bool);mask[eos]=True
 def generate():return model.generate(input_ids=x,attention_mask=torch.ones_like(x),past_key_values=m.Qwen3_5DynamicCache(cfg),do_sample=False,min_new_tokens=128,max_new_tokens=128,pad_token_id=tok.eos_token_id,logits_to_keep=1,return_dict_in_generate=True,output_logits=True)
 with torch.inference_mode():
  for layer in model.model.language_model.layers:
   if hasattr(layer,'linear_attn'):
    layer.linear_attn.chunk_gated_delta_rule=m.torch_chunk_gated_delta_rule;layer.linear_attn.recurrent_gated_delta_rule=e.NATIVE
  ref=generate();ref_ids=ref.sequences.cpu();ref_raw=[z.cpu() for z in ref.logits];ref_cache=cache_cpu(ref.past_key_values)
  assert ref_ids.tolist()==original['complete_output_ids'] and all(bool(torch.isfinite(z).all()) for z in ref_raw)
  del ref;gc.collect();torch.cuda.empty_cache()
  # First determine whether plain eager inplace mutation is itself the problem.
  for layer in model.model.language_model.layers:
   if hasattr(layer,'linear_attn'):layer.linear_attn.recurrent_gated_delta_rule=g.recurrent
  rr=generate();ung={'all_logits_finite':all(bool(torch.isfinite(z).all()) for z in rr.logits),'complete_ids_exact':torch.equal(rr.sequences.cpu(),ref_ids),'all_raw_logits_exact':all(torch.equal(z.cpu(),w) for z,w in zip(rr.logits,ref_raw)),'cache_exact':caches_equal(rr.past_key_values,ref_cache)}
  dump('ungraphed-direct-state.json',ung);print('UNGRAPHED',json.dumps(ung),flush=True);del rr;gc.collect();torch.cuda.empty_cache()
  for keep in [False,True]:
   assert time.time()<D-180
   g.enable_graph(model);engine=g.Engine(model,cfg,mask,x,n=128,keep_logits=keep)
   engine.input.copy_(x);segments=[]
   for j,graph in enumerate(engine.graphs):
    graph.replay();torch.cuda.synchronize();valid=min(1+8*j,128);yy=engine.out[:,:l+valid].cpu();target=ref_ids[:,:l+valid];bad=(yy!=target).nonzero().tolist()
    segments.append({'segment':j,'valid_generated_tokens':valid,'ids_exact':torch.equal(yy,target),'first_mismatch':bad[0] if bad else None,'prefix_finite':bool(engine.finite[:valid].all())})
   yy=engine.out.cpu();record={'keep_logits':keep,'segments':segments,'complete_ids_exact':torch.equal(yy,ref_ids),'generated_ids':yy[:,l:].tolist(),'native_ids':ref_ids[:,l:].tolist(),'all_raw_finite':bool(engine.finite.all()),'cache_exact':caches_equal(engine.cache,ref_cache),'full_vocab_steps':None}
   if keep:
    candidate=[engine.first_raw]+engine.raw
    assert len(candidate)==128
    record['full_vocab_steps']=[{'step':i,'exact':torch.equal(z.cpu(),w),'max_abs':float((z.cpu().float()-w.float()).abs().max()),'same_prefix':torch.equal(yy[:,:l+i],ref_ids[:,:l+i])} for i,(z,w) in enumerate(zip(candidate,ref_raw))]
   dump('capture-keep'+str(keep)+'.json',record);print('CAPTURE',keep,'IDs',record['complete_ids_exact'],'cache',record['cache_exact'],flush=True)
   del engine;gc.collect();torch.cuda.empty_cache()
 dump('completion.json',{'diagnostic_only':True,'qualified_performance_runs':0,'finished_unix':time.time()})
except Exception as ex:dump('failure.json',{'error':repr(ex),'traceback':traceback.format_exc()});raise

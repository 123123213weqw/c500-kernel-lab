"""Complete local request pipeline, paired native vs exact partial-fused GDN."""
import pathlib,time,json,hashlib,datetime,gc,traceback,shutil,os
import argparse,sys,statistics
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model',required=True,type=pathlib.Path)
parser.add_argument('--workdir',required=True,type=pathlib.Path)
parser.add_argument('--strategy',choices=['segmented','direct-state'],default='segmented')
parser.add_argument('--batches',default='1,8,48')
args=parser.parse_args()
START=time.monotonic();DEADLINE=float(os.environ['C500_DEADLINE'])
assert time.time()<DEADLINE-500,'At least500seconds of authorized runtime required'
REPO=pathlib.Path(__file__).resolve().parent.parent
ROOT=args.workdir.resolve();(ROOT/'results').mkdir(parents=True,exist_ok=True)
MODEL=args.model.resolve();assert MODEL.is_dir()
BATCHES=[int(z) for z in args.batches.split(',')]
assert BATCHES and len(set(BATCHES))==len(BATCHES) and set(BATCHES)<={1,8,48}
R=ROOT/'results'/args.strategy;R.mkdir()
sys.path.insert(0,str(REPO/'strategies/segmented'))
if args.strategy=='direct-state':
 sys.path.insert(0,str(REPO/'experimental/direct_state'))
 import exact_gdn_inplace as exact_gdn,direct_state_graph_engine as graph_engine
else:
 import exact_gdn,segmented_graph_engine as graph_engine
micro=json.load(open(ROOT/'results'/('micro-'+args.strategy+'.json')))
assert len(micro)==30 and all(x['output_exact'] and x['state_exact'] and x['finite'] for x in micro)
config=json.load(open(REPO/'strategies/segmented/FROZEN_CONFIG.json'))['config']
exact_gdn.BLOCK=config['block'];exact_gdn.WARPS=config['warps']
import torch,numpy as np,transformers,importlib.metadata as md
expected=json.load(open(REPO/'strategies/segmented/FROZEN_CONFIG.json'))['versions']
assert all(md.version(n)==expected[n] for n in ['torch','triton','transformers']),'Untested stack: qualify separately, do not silently relax pinned assumptions'
from transformers import AutoTokenizer,LogitsProcessor,LogitsProcessorList
from transformers.models.qwen3_5 import modeling_qwen3_5 as m
import fcntl
lock=(ROOT/'gpu-work.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert torch.cuda.device_count()==1 and 'C500' in torch.cuda.get_device_name(0)
torch.set_num_threads(8);torch.manual_seed(42)
rows=[];gates=[]
def dump(name,x):
 p=R/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str));t.replace(p)
def emit(x):
 rows.append(x);dump('runs.json',rows);print('RESULT',json.dumps(x,ensure_ascii=False),flush=True)
def digest(x):return hashlib.sha256(x.numpy().tobytes()).hexdigest()
(R/'source').mkdir(exist_ok=True)
for p in [pathlib.Path(__file__)]+list((REPO/'strategies/segmented').glob('*.py'))+list((REPO/'experimental/direct_state').glob('*.py')):shutil.copyfile(p,R/'source'/p.name)
try:
 t=time.monotonic()
 model,loading=m.Qwen3_5ForConditionalGeneration.from_pretrained(MODEL,dtype=torch.bfloat16,device_map={'':'cuda:0'},attn_implementation='eager',local_files_only=True,output_loading_info=True)
 assert not loading.get('missing_keys'),loading
 model.eval();cfg=model.config.get_text_config()
 assert {p.device.type for p in model.parameters()}=={'cuda'}
 tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
 assert model.generation_config.repetition_penalty in (None,1.0)
 eos=model.generation_config.eos_token_id
 if isinstance(eos,int):eos=[eos]
 eos_mask=torch.zeros(cfg.vocab_size,device='cuda',dtype=torch.bool);eos_mask[eos]=True
 metadata={'model':'Qwen/Qwen3.5-27B','parameters_loaded':sum(p.numel() for p in model.parameters()),'dtype':'BF16 weights, native FP32 recurrent state','backend':'native HF eager attention/vendor conv, paired native Torch GDN vs decode-only exact partial fusion; original normalization+reductions retained','frozen_config':config,'variant':args.strategy,'direct_state_write':args.strategy=='direct-state','graph_strategy':'entire native128token prefill graph plus8-step decode graph segments (shared pool, sequential order); exact partial fusion/inplace state copy; all prefix state recomputed every replay; exact FP32 zeros allocated on device instead of native CPU->GPU zero copy; unpadded-known mask branch host guard; A/B/A request-leak probes; fixed-shape official eager masks precomputed at startup; full-vocab finite every timed step','device':'only allocated C500 cuda:0','loading_info':loading,'load_and_tokenizer_seconds':time.monotonic()-t,'worker_before_load_seconds':t-START,'versions':{n:md.version(n) for n in ['torch','triton','transformers','accelerate','huggingface_hub','flash-linear-attention','flash-attn','causal-conv1d']},'generation_config':model.generation_config.to_dict(),'input_len':128,'output_len':128,'batches':BATCHES,'repeats':3,'deadline_unix':DEADLINE,'strategy':args.strategy,'measurement':'CPU tokenization -> blocking H2D -> native freshcache prefill+128token generation -> D2H -> batch_decode. No HTTP/queue/network. Excludes model loading, correctness, first warmup, hash calculation; those logged separately. No HTTP/serving claim. Fixed config before all model timings, three paired repetitions per case.'}
 dump('metadata.json',metadata)
 def native(x,mask=None,n=128,logits=False,checker=None):
  for layer in model.model.language_model.layers:
   if hasattr(layer,'linear_attn'):layer.linear_attn.chunk_gated_delta_rule=m.torch_chunk_gated_delta_rule
  handle=model.register_forward_hook(checker.raw_hook) if checker is not None else None
  try:return model.generate(input_ids=x,attention_mask=torch.ones_like(x) if mask is None else mask,past_key_values=m.Qwen3_5DynamicCache(cfg),do_sample=False,min_new_tokens=n,max_new_tokens=n,pad_token_id=tok.eos_token_id,logits_to_keep=1,return_dict_in_generate=logits,output_logits=logits,logits_processor=LogitsProcessorList([checker]) if checker is not None else None)
  finally:
   if handle is not None:handle.remove()
 def manual(x,n=4):
  b,l=x.shape;out=torch.empty((b,l+n),device=x.device,dtype=torch.long);out[:,:l].copy_(x);mask=torch.ones_like(out);cache=m.Qwen3_5DynamicCache(cfg)
  pos=torch.arange(l,device=x.device);pp=model._prepare_position_ids_for_generation(x,{'attention_mask':mask[:,:l],'past_key_values':cache});ll=[]
  for i in range(n):
   current=out[:,:l] if i==0 else out[:,l+i-1:l+i]
   inp=model.prepare_inputs_for_generation(current,past_key_values=cache,attention_mask=mask[:,:l+i],cache_position=pos,position_ids=pp,use_cache=True,logits_to_keep=1,is_first_iteration=i==0)
   r=model(**inp);cache=r.past_key_values;raw=r.logits[:,-1].float();ll.append(raw.clone());out[:,l+i].copy_(torch.where(eos_mask,-float('inf'),raw).argmax(-1));pos=pos[-1:]+1;pp=pp[...,-1:]+1
  return out,ll
 class FiniteCheck(LogitsProcessor):
  def __init__(self):self.calls=0;self.raw_calls=0
  def raw_hook(self,module,args,output):
   assert bool(torch.isfinite(output.logits).all()),'Nonfinite RAW full-vocab logits reject warmup'
   self.raw_calls+=1
  def __call__(self,input_ids,scores):
   bad=~torch.isfinite(scores);bad[:,eos]=False
   if bool(bad.any()):
    ij=bad.nonzero()[:32];vals=scores[ij[:,0],ij[:,1]].float().cpu().tolist()
    info={'calls_before_failure':self.calls,'eos_ids':eos,'bad_total':int(bad.sum()),'nan_count':int(torch.isnan(scores).sum()),'posinf_count':int(torch.isposinf(scores).sum()),'neginf_count':int(torch.isneginf(scores).sum()),'sample_coordinates':ij.cpu().tolist(),'sample_values':[str(v) for v in vals],'last_input_ids':input_ids[:1,-16:].cpu().tolist()}
    dump('finite-rejection.json',info);print('FINITE_REJECTION',json.dumps(info),flush=True)
    raise AssertionError('Nonfinite non-EOS scores reject benchmark')
   self.calls+=1;return scores
 audited_original=model._get_logits_processor
 def audited_processors(*args,**kwargs):
  result=audited_original(*args,**kwargs)
  if any(isinstance(z,FiniteCheck) for z in result):dump('warmup-logits-processors.json',[{'type':type(z).__name__,'attributes':vars(z)} for z in result])
  return result
 model._get_logits_processor=audited_processors
 body=('本次实验讨论城市公共图书馆如何提升服务质量。事实包括：开放时间需要覆盖工作日晚间，预约规则必须透明，借阅记录应保护隐私，无障碍设施应定期维护，采购决策要依据读者反馈。团队准备先做小规模试点，再收集匿名问卷，比较满意度、等待时间与服务成本。报告应区分事实、假设和待验证的问题，避免将单次观察当成普遍规律。')
 def prompts(b):return [f'任务编号{i+1}：请根据以下材料，写出三项具体建议并说明理由。\n'+body*8 for i in range(b)]
 def request(texts,checker=None):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
  enc=tok(texts,return_tensors='pt',padding='max_length',truncation=True,max_length=128)
  assert tuple(enc['input_ids'].shape)==(len(texts),128) and bool(enc['attention_mask'].all())
  t1=time.perf_counter();x=enc['input_ids'].to('cuda');mask=enc['attention_mask'].to('cuda');torch.cuda.synchronize();t2=time.perf_counter()
  y=native(x,mask=mask,checker=checker);torch.cuda.synchronize();t3=time.perf_counter()
  cpu=y.cpu();t4=time.perf_counter();decoded=tok.batch_decode(cpu[:,128:],skip_special_tokens=False,clean_up_tokenization_spaces=False);t5=time.perf_counter()
  stats={'e2e_s':t5-t0,'tokenize_s':t1-t0,'h2d_s':t2-t1,'generate_s':t3-t2,'d2h_s':t4-t3,'detokenize_s':t5-t4,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30}
  assert tuple(cpu.shape)==(len(texts),256) and len(decoded)==len(texts)
  return cpu,decoded,stats,enc['input_ids']
 with torch.inference_mode():
  allowed=np.array([i for i in range(100,cfg.vocab_size) if i not in set(tok.all_special_ids)])
  for b,l in [(1,13),(3,127),(17,32)]:
   x=torch.tensor(np.random.default_rng(993+b).choice(allowed,size=(b,l)),device='cuda')
   exact_gdn.configure(model,False);ref=native(x,n=4,logits=True)
   graph_engine.enable_graph(model);t=time.monotonic();engine=graph_engine.Engine(model,cfg,eos_mask,x,n=4,keep_logits=True)
   yy=engine(x);torch.cuda.synchronize()
   cache_equal={}
   for field in graph_engine.FIELDS:
    aa=getattr(ref.past_key_values,field);bb=getattr(engine.cache,field)
    cache_equal[field]=all((u is None and v is None) or (u is not None and v is not None and torch.equal(u,v)) for u,v in zip(aa,bb))
   g={'batch':b,'input_len':l,'output_len':4,'logits_finite':bool(engine.finite.all()),'full_vocab_prefill_logits_exact':torch.equal(ref.logits[0].float(),engine.first_raw.float()),'full_vocab_decode_logits_exact':len(engine.raw)==3 and all(torch.equal(v.float(),w.float()) for v,w in zip(ref.logits[1:],engine.raw)),'complete_ids_exact':torch.equal(ref.sequences,yy),'cache_exact':cache_equal,'warm_capture_gate_s':time.monotonic()-t}
   assert all(g[k] for k in ['logits_finite','full_vocab_prefill_logits_exact','full_vocab_decode_logits_exact','complete_ids_exact']) and all(cache_equal.values())
   del ref,yy,aa,bb
   probe_rows=[]
   x2=torch.tensor(np.random.default_rng(1207+b).choice(allowed,size=(b,l)),device='cuda')
   for probe_name,probe in [('B',x2),('A_again',x)]:
    exact_gdn.configure(model,False);rr=native(probe,n=4,logits=True);oo=engine(probe);torch.cuda.synchronize()
    caches=all(all((u is None and v is None) or (u is not None and v is not None and torch.equal(u,v)) for u,v in zip(getattr(rr.past_key_values,f),getattr(engine.cache,f))) for f in graph_engine.FIELDS)
    pr={'probe':probe_name,'complete_ids_exact':torch.equal(rr.sequences,oo),'full_vocab_logits_exact':torch.equal(rr.logits[0].float(),engine.first_raw.float()) and all(torch.equal(u.float(),v.float()) for u,v in zip(rr.logits[1:],engine.raw)),'all_caches_exact':caches,'all_logits_finite':bool(engine.finite.all())}
    probe_rows.append(pr);assert all(v for k,v in pr.items() if k!='probe');del rr,oo
   g['fresh_request_probes']=probe_rows;gates.append(g);dump('holdouts.json',gates);print('FULL_GRAPH_HOLDOUT',json.dumps(g),flush=True)
   del x2
   del x,engine;gc.collect();torch.cuda.empty_cache()
  warmups=[]
  for b in BATCHES:
   if time.time()>DEADLINE-100:
    dump('stopped-before-next-case.json',{'batch':b,'reason':'deadline_budget','time_unix':time.time()});break
   text=prompts(b);dump(f'prompts-b{b}.json',text);print('CASE_BEGIN',b,time.time(),flush=True)
   try:
    exact_gdn.configure(model,False)
    checker=FiniteCheck();ref,decoded,stats,input_ids=request(text,checker);assert checker.calls==checker.raw_calls==128
    refhash=digest(ref);inputhash=digest(input_ids)
    warmups.append({'mode':'native','batch':b,'raw_full_vocab_finite_calls':checker.raw_calls,'complete_output_sha256':refhash,'input_sha256':inputhash,**stats})
    dump(f'warm-output-b{b}.json',{'input_ids':input_ids.tolist(),'complete_output_ids':ref.tolist(),'generated_text':decoded})
    def record(mode,rep,y,textout,stats,ci):
     assert torch.equal(y,ref) and digest(ci)==inputhash and textout==decoded
     assert mode=='native' or (exact_gdn.COUNTS['fallback']==0 and sum(v for k,v in exact_gdn.COUNTS.items() if k!='fallback')>0),'Candidate fallback or no fused dispatch'
     emit({'layout_counts':dict(exact_gdn.LAYOUTS),'dispatch_counts':dict(exact_gdn.COUNTS),'mode':mode,'batch':b,'repeat':rep,'input_len':128,'output_len':128,'output_tokens':b*128,'e2e_output_tokens_per_s':b*128/stats['e2e_s'],'model_only_output_tokens_per_s':b*128/stats['generate_s'],'per_request_output_tokens_per_s':128/stats['e2e_s'],'complete_output_sha256':digest(y),'input_sha256':inputhash,'all_ids_exact_vs_warmup':True,'detokenized_text_exact_vs_warmup':True,**stats})
    for rep in range(3):
     y,tx,st,ci=request(text);record('native',rep,y,tx,st,ci);del y,tx,ci
    graph_engine.enable_graph(model);x=input_ids.to('cuda');t=time.monotonic();engine=graph_engine.Engine(model,cfg,eos_mask,x,n=128)
    dump(f'graph-setup-b{b}.json',{'warm_capture_s':time.monotonic()-t,'resident_allocated_gib':torch.cuda.memory_allocated()/2**30,'reserved_gib':torch.cuda.memory_reserved()/2**30,'includes_native_prefix_warmup':True,'prefill_captured':True})
    def graph_request(text):
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t0=time.perf_counter()
     enc=tok(text,return_tensors='pt',padding='max_length',truncation=True,max_length=128);assert bool(enc['attention_mask'].all());t1=time.perf_counter()
     xx=enc['input_ids'].to('cuda');mm=enc['attention_mask'].to('cuda');torch.cuda.synchronize();t2=time.perf_counter()
     yy=engine(xx);torch.cuda.synchronize();t3=time.perf_counter()
     cpu=yy.cpu();fin=engine.finite.cpu();assert bool(fin.all());t4=time.perf_counter()
     dec=tok.batch_decode(cpu[:,128:],skip_special_tokens=False,clean_up_tokenization_spaces=False);t5=time.perf_counter()
     st={'e2e_s':t5-t0,'tokenize_s':t1-t0,'h2d_s':t2-t1,'generate_s':t3-t2,'d2h_s':t4-t3,'detokenize_s':t5-t4,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'raw_full_vocab_finite_calls':int(fin.sum())}
     return cpu,dec,st,enc['input_ids']
    y,tx,st,ci=graph_request(text)
    if not (torch.equal(y,ref) and tx==decoded and digest(ci)==inputhash):
     diff=(y!=ref).nonzero().tolist() if y.shape==ref.shape else []
     dump(f'graph-first-request-rejection-b{b}.json',{'complete_ids_exact':torch.equal(y,ref),'text_exact':tx==decoded,'input_sha_exact':digest(ci)==inputhash,'first_mismatch_coordinate':diff[0] if diff else None,'actual_ids':y.tolist(),'native_ids':ref.tolist(),'all_raw_finite_calls':st['raw_full_vocab_finite_calls'],'qualified_performance':False})
     raise AssertionError('Real128 first graph request rejected; see graph-first-request-rejection JSON')
    warmups.append({'mode':args.strategy,'batch':b,'complete_output_sha256':digest(y),'input_sha256':inputhash,**st});dump('warmups.json',warmups);del y,tx,ci
    for rep in range(3):
     if time.time()>DEADLINE-70:raise TimeoutError('Hard deadline')
     y,tx,st,ci=graph_request(text);record(args.strategy,rep,y,tx,st,ci);del y,tx,ci
    del ref,decoded,input_ids,engine,x
   except Exception as e:
    with (R/'errors.jsonl').open('a') as f:f.write(json.dumps({'batch':b,'error':repr(e),'traceback':traceback.format_exc()})+'\n')
    print('CASE_FAILED',b,repr(e),flush=True)
    raise
   finally:gc.collect();torch.cuda.empty_cache()
 dump('completion.json',{'time_unix':time.time(),'run_rows':len(rows),'completed_all_planned':len(rows)==6*len(BATCHES),'gpu_worker_finishing':True,'layout_counts':dict(exact_gdn.LAYOUTS),'dispatch_counts':dict(exact_gdn.COUNTS)});print('E2E_DONE',len(rows),flush=True)
except Exception as e:dump('fatal.json',{'error':repr(e),'traceback':traceback.format_exc()});raise

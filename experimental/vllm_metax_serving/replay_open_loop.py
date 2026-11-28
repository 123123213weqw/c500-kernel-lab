"""Reproducible C16 native vLLM open-loop MTP policy screening."""
import argparse
from collections import Counter
from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time
from statistics import median

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["MACA_VLLM_USE_EXTERNAL_FLA"] = "0"

import torch
from vllm import LLM, SamplingParams
from vllm.plugins import load_general_plugins
from vllm.sampling_params import RequestOutputKind

def write(p,obj):
    t=p.with_suffix(".part")
    t.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    t.replace(p)

p=argparse.ArgumentParser(description=__doc__)
for k in ("model","trace","out"):
    p.add_argument("--"+k,type=Path,required=True)
p.add_argument("--mode",choices=("k1","dynamic","longcap"),required=True)
p.add_argument("--quality",type=Path)
a=p.parse_args()
assert a.model.joinpath("VERIFIED").is_file()
assert torch.cuda.device_count()==1 and "C500" in torch.cuda.get_device_name()
trace_sha=sha256(a.trace.read_bytes()).hexdigest()
traces=json.loads(a.trace.read_text())["traces"]
a.out.mkdir(parents=True,exist_ok=True)
load_general_plugins()
k=2 if a.mode=="dynamic" else 1
long_threshold=1024 if a.mode=="longcap" else 0
spec={"method":"qwen3_5_mtp","num_speculative_tokens":k}
if a.mode=="dynamic":
    spec["num_speculative_tokens_per_batch_size"]=[[1,4,2],[5,16,1]]
graph_rows=[b*(k+1) for b in [1,2,4,6,8,10,12,14,16]]
llm=LLM(model=str(a.model),dtype="bfloat16",max_model_len=4352,language_model_only=True,
        max_num_seqs=16,max_num_batched_tokens=4096,long_prefill_token_threshold=long_threshold,async_scheduling=False,
        gpu_memory_utilization=.95,mamba_ssm_cache_dtype="float32",generation_config="vllm",
        compilation_config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":graph_rows},
        speculative_config=spec,enable_prefix_caching=False,trust_remote_code=False,
        disable_log_stats=True)
eng=llm.llm_engine
core=eng.engine_core.engine_core
runner=core.model_executor.driver_worker.worker.model_runner
sched=core.scheduler
tok=llm.get_tokenizer()
write(a.out/"config.json",{"mode":a.mode,"spec":spec,"graph_rows":graph_rows,"native_max_admitted":16,
    "kv_blocks":runner.kv_cache_config.num_blocks,"long_prefill_token_threshold":long_threshold,"request_stream_is_open_loop":True,"trace_sha256":trace_sha})
print("READY",a.mode,spec,graph_rows,runner.kv_cache_config.num_blocks,flush=True)

def percentile(v,p):
    s=sorted(v)
    return s[math.ceil(len(s)*p)-1] if s else None

def exercise(case,label,cap,ignore_eos,graph_audit=False):
    assert not eng.has_unfinished_requests()
    assert case and all(a["arrival_ms"]<=b["arrival_ms"] for a,b in zip(case,case[1:]))
    old_cap=sched.max_num_scheduled_tokens
    orig=sched.schedule
    old_replay=torch.cuda.CUDAGraph.replay
    queue_max=0
    progress_max=0
    todo=list(case)
    live={}
    by_opaque_id={}
    records={}
    k_hist=Counter()
    scheduled_hist=Counter()
    chunks=Counter()
    counts={"replay":0,"preempt":0}
    step_rows=[]
    def audited_replay(graph,*aa,**kk):
        counts["replay"]+=1
        return old_replay(graph,*aa,**kk)
    def tap(*args,**kwargs):
        out=orig(*args,**kwargs)
        t=(time.perf_counter()-ts)*1000
        keys=out.num_scheduled_tokens
        if keys:
            k_hist[int(out.num_spec_tokens_to_schedule)]+=1
            scheduled_hist[len(keys)]+=1
        for req in keys:
            tag=by_opaque_id.get(req)
            if tag is not None and records[tag]["first_schedule_ms"] is None:
                records[tag]["first_schedule_ms"]=t
        counts["preempt"]+=len(out.preempted_req_ids or ())
        if graph_audit:
            step_rows.append({"scheduled":len(keys),"scheduled_tokens":out.total_num_scheduled_tokens,
                              "k":int(out.num_spec_tokens_to_schedule)})
        return out

    sched.max_num_scheduled_tokens=cap
    sched.schedule=tap
    if graph_audit:
        torch.cuda.CUDAGraph.replay=audited_replay
    try:
        torch.cuda.synchronize()
        ts=time.perf_counter()
        while todo or live:
            elapsed=(time.perf_counter()-ts)*1000
            while todo and todo[0]["arrival_ms"]<=elapsed:
                q=todo.pop(0)
                tag=f"{label}/{q['id']}"
                params=SamplingParams(temperature=0,seed=7,max_tokens=q["max_output_tokens"],
                    ignore_eos=ignore_eos,detokenize=False,output_kind=RequestOutputKind.DELTA)
                internal=eng.add_request(tag,{"prompt_token_ids":q["prompt_ids"]},params)
                live[tag]=q["id"]
                by_opaque_id[internal if isinstance(internal,str) else tag]=tag
                records[tag]={"id":q["id"],"planned_arrival_ms":q["arrival_ms"],
                              "submit_ms":(time.perf_counter()-ts)*1000,
                              "first_schedule_ms":None,"ends_ms":None,"deliveries_ms":[],
                              "output_ids":[],"input_tokens":len(q["prompt_ids"]),
                              "max_output_tokens":q["max_output_tokens"],"suite":q.get("suite")}
            queue_max=max(queue_max,len(sched.waiting))
            progress_max=max(progress_max,len(live))
            if not live:
                time.sleep(max(0,min(.02,todo[0]["arrival_ms"]/1000-(time.perf_counter()-ts))))
                continue
            n0=counts["replay"]
            ready=eng.step()
            if graph_audit:
                if step_rows:
                    step_rows[-1]["graph_replay_calls"]=counts["replay"]-n0
            t=(time.perf_counter()-ts)*1000
            for o in ready:
                rec=records[o.request_id]
                v=o.outputs[0]
                if v.token_ids:
                    rec["output_ids"].extend(v.token_ids)
                    rec["deliveries_ms"].append(t)
                    chunks[len(v.token_ids)]+=1
                if o.finished:
                    rec["ends_ms"]=t
                    rec["finish_reason"]=v.finish_reason
                    del live[o.request_id]
        torch.cuda.synchronize()
        wall=time.perf_counter()-ts
    finally:
        sched.schedule=orig
        sched.max_num_scheduled_tokens=old_cap
        if graph_audit:
            torch.cuda.CUDAGraph.replay=old_replay
    rows=list(records.values())
    for r in rows:
        n=len(r["output_ids"])
        assert r["first_schedule_ms"] is not None
        assert (n==r["max_output_tokens"] if ignore_eos else 0<n<=r["max_output_tokens"])
        r["completion_tokens"]=n
        r["prompt_tokens"]=r["input_tokens"]
        r["ttft_ms"]=r["deliveries_ms"][0]-r["planned_arrival_ms"]
        r["e2e_ms"]=r["ends_ms"]-r["planned_arrival_ms"]
        r["queue_to_first_schedule_ms"]=r["first_schedule_ms"]-r["planned_arrival_ms"]
        r["host_submit_lag_ms"]=r["submit_ms"]-r["planned_arrival_ms"]
        if not ignore_eos:
            r["text"]=tok.decode(r["output_ids"],skip_special_tokens=True)
    total=sum(r["completion_tokens"] for r in rows)
    def p95(key,condition=lambda _:True):
        return percentile([r[key] for r in rows if condition(r)],.95)
    s={"tag":label,"mode":a.mode,"scheduled_token_cap":cap,"fixed_output":ignore_eos,
       "wall_s":wall,"total_output_tokens":total,"output_tok_s":total/wall,"n":len(rows),
       "p95_ttft_ms":p95("ttft_ms"),"p95_e2e_ms":p95("e2e_ms"),
       "p95_short_ttft_ms":p95("ttft_ms",lambda r:r["input_tokens"]<=128),
       "p95_long_ttft_ms":p95("ttft_ms",lambda r:r["input_tokens"]>=2048),
       "p95_queue_ms":p95("queue_to_first_schedule_ms"),"p95_host_submit_lag_ms":p95("host_submit_lag_ms"),
       "p95_nonempty_chunk_gap_ms":percentile([b-a for r in rows for a,b in zip(r["deliveries_ms"],r["deliveries_ms"][1:])],.95),
       "peak_native_waiting":queue_max,"peak_outstanding_client":progress_max,"num_preemptions":counts["preempt"],
       "steps_by_proposed_k":dict(k_hist),"steps_by_scheduled_batch":dict(scheduled_hist),
       "chunks_by_output_size":dict(chunks),"trace_sha256":trace_sha}
    if graph_audit:
        s["actual_graph_replay_calls_in_audit"]=counts["replay"]
    write(a.out/f"{label}.json",{"summary":s,"rows":rows,"graph_step_audit":step_rows})
    print("RESULT",label,round(s["output_tok_s"],2),"p95_first",round(s["p95_ttft_ms"],1),"p95_done",
           round(s["p95_e2e_ms"],1),"K",s["steps_by_proposed_k"],"preempt",counts["preempt"],flush=True)
    return s

def warm(seed):
    # Same full prompt set once, but only 24 generated tokens per request.
    return [dict(x,arrival_ms=min(x["arrival_ms"],2600),max_output_tokens=24) for x in traces[seed]]

caps=(4096,2048) if a.mode=="k1" else (4096,)
for cap in caps:
    w=exercise(warm("31"),f"warm-{cap}",cap,True,True)
    if a.mode=="dynamic":
        assert {1,2}.issubset(w["steps_by_proposed_k"])
for i,(seed,case) in enumerate(traces.items()):
    order=caps if i%2==0 else tuple(reversed(caps))
    for cap in order:
        exercise(case,f"seed-{seed}-{cap}",cap,True)

if a.quality is not None:
    data=json.loads(a.quality.read_text())
    public=[{**x,"max_output_tokens":2048,"arrival_ms":min(7000,i*375)} for i,x in enumerate(data["specs"])]
    if a.mode!="dynamic":
        # Grade scheduler caps separately; they change batch shapes despite matching input and EOS.
        for cap in caps:
            exercise(public,f"quality-{cap}",cap,False)

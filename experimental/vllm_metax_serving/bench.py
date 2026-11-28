"""Native C500 vLLM AR/K1 screening with paired fixed-length and EOS runs."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["MACA_VLLM_USE_EXTERNAL_FLA"] = "0"
import torch
from vllm import LLM, SamplingParams
from vllm.plugins import load_general_plugins
from vllm.sampling_params import RequestOutputKind


def save(path, obj):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


p = argparse.ArgumentParser(description=__doc__)
for k in ("model", "workload", "out"):
    p.add_argument("--"+k, type=Path, required=True)
p.add_argument("--mode", choices=("ar","mtp1"), required=True)
p.add_argument("--natural", action="store_true")
a = p.parse_args()
assert a.model.joinpath("VERIFIED").is_file()
assert torch.cuda.device_count() == 1 and "C500" in torch.cuda.get_device_name()
spec = json.loads(a.workload.read_text())["specs"]
assert len(spec) == len({s["id"] for s in spec}) == 20
load_general_plugins()
k = int(a.mode == "mtp1")
graph_rows = [1,4,6,8,10,12,14,16] if k == 0 else [2,8,12,16,20,24,28,32]
llm = LLM(model=str(a.model), dtype="bfloat16", max_model_len=4352, language_model_only=True,
          max_num_seqs=16, max_num_batched_tokens=4096, async_scheduling=False,
          gpu_memory_utilization=.95, mamba_ssm_cache_dtype="float32", generation_config="vllm",
          compilation_config={"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":graph_rows},
          speculative_config={"method":"qwen3_5_mtp","num_speculative_tokens":k} if k else None,
          enable_prefix_caching=False, trust_remote_code=False, disable_log_stats=False)
eng = llm.llm_engine
core = eng.engine_core.engine_core
runner = core.model_executor.driver_worker.worker.model_runner
tok = llm.get_tokenizer()
config = {"mode":a.mode,"k":k,"graph_capture_rows":graph_rows,"max_model_len":4352,
          "max_num_seqs":16,"max_num_batched_tokens":4096,"kv_blocks":runner.kv_cache_config.num_blocks}
save(a.out/"config.json",config)
print("READY",config,flush=True)

def exercise(label, rows, max_tokens, force):
    assert not eng.has_unfinished_requests()
    todo = list(rows)
    pending = {}
    records = {}
    params = SamplingParams(temperature=0,max_tokens=max_tokens,ignore_eos=force,detokenize=False,
                            output_kind=RequestOutputKind.DELTA,seed=7)
    step_sizes = Counter()
    chunks = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    while todo or pending:
        while todo and len(pending) < 16:
            r = todo.pop(0)
            tag = f"{label}/{r['id']}"
            eng.add_request(tag,{"prompt_token_ids":r["prompt_ids"]},params)
            pending[tag] = r["id"]
            records[r["id"]] = {"id":r["id"],"suite":r["suite"],"prompt_tokens":len(r["prompt_ids"]),
                                "output_ids":[],"arrival_s":time.perf_counter()-start,
                                "first_s":None,"end_s":None}
        ready = eng.step()
        t = time.perf_counter()-start
        step_sizes[len(pending)] += 1
        for out in ready:
            r = records[pending[out.request_id]]
            seq = out.outputs[0]
            if seq.token_ids:
                if r["first_s"] is None:
                    r["first_s"] = t
                r["output_ids"].extend(seq.token_ids)
                chunks.append(len(seq.token_ids))
            if out.finished:
                pending.pop(out.request_id)
                r["end_s"] = t
                r["finish_reason"] = seq.finish_reason
    torch.cuda.synchronize()
    wall = time.perf_counter()-start
    out = list(records.values())
    for r in out:
        r["completion_tokens"] = n = len(r["output_ids"])
        assert n>0 and (not force or n==max_tokens) and r["first_s"] is not None
        r["ttft_ms"] = (r["first_s"]-r["arrival_s"])*1000
        r["tpot_ms"] = (r["end_s"]-r["first_s"])*1000/max(1,n-1)
        r["text"] = tok.decode(r["output_ids"],skip_special_tokens=True)
    count = sum(r["completion_tokens"] for r in out)
    ft = sorted(r["ttft_ms"] for r in out)
    summary = {"tag":label,"mode":a.mode,"fixed":force,"max_tokens":max_tokens,"items":len(out),
               "total_output_tokens":count,"aggregate_tok_s":count/wall,"wall_s":wall,"concurrency":16,
               "mean_ttft_ms":sum(ft)/len(ft),"p95_ttft_ms":ft[math.ceil(.95*len(ft))-1],
               "mean_tpot_ms":sum(r["tpot_ms"] for r in out)/len(out),
               "steps_by_inflight":dict(step_sizes),"nonempty_output_chunks_by_size":dict(Counter(chunks))}
    save(a.out/(label+".json"),{"summary":summary,"rows":out})
    print("RESULT",label,count,round(count/wall,3),round(wall,3),summary["nonempty_output_chunks_by_size"],flush=True)

exercise("warm",spec,32,True)
for r in range(3):
    exercise(f"fixed-{r+1}",spec,160,True)
if a.natural:
    exercise("natural",spec,2048,False)

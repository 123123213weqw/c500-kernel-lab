"""Compare the unchanged author SpecPilot against the exact fixed-output fixture."""
import argparse
import json
from pathlib import Path
import sys

p=argparse.ArgumentParser(description=__doc__)
for k in ("target","draft","pkg","workload","out"):
    p.add_argument("--"+k,type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.pkg))
from scripts.run_batched_bench import _build_engine
import torch

specs=json.loads(a.workload.read_text())["specs"]
assert len(specs)==20 and max(len(s["prompt_ids"]) for s in specs)==2048
args=argparse.Namespace(model_dir=str(a.target),draft_dir=str(a.draft),max_seq=4352,
                        draft_quant="w8a8",max_batch=16,concurrency=16,max_n_per_req=16,
                        max_total_nodes=256,m_mma_max=64,no_w8_draft_head=False)
eng,tok=_build_engine(args)
for i,cap in enumerate((32,160,160,160)):
    result=eng.serve_requests(specs,concurrency=16,max_tokens=cap,eos_token_id=-1,tok=tok,
                              budget_mode="shared_budget",n_per_req=4,total_budget=64,min_nodes=2,
                              use_graph=True,verbose=False)
    assert len(result["rows"])==20 and result["summary"]["total_output_tokens"]==20*cap
    assert all(row["completion_tokens"]==cap for row in result["rows"])
    path=a.out/(f"fixed-{i}.json" if i else "warm.json")
    tmp=path.with_suffix(".part")
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(path)
    print("RESULT",path.stem,result["summary"]["aggregate_tok_s"],result["summary"]["wall_s"],
          sorted(eng._draft_graphs),sorted(eng._verify_graphs),flush=True)

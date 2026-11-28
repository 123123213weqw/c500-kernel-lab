"""Validate matching open-loop inputs/outputs and pair per-seed performance."""
from pathlib import Path
from collections import Counter
from statistics import median
from hashlib import sha256
import json
import re
import sys

d=Path(sys.argv[1])
def boxed(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    else:
        # The chat prompt already opens thinking, so absence of a closing
        # marker means no final answer, even if the unfinished reasoning has a box.
        return None
    values = []
    for m in re.finditer(r"\\boxed\s*\{", text):
        level = 1
        for i in range(m.end(), len(text)):
            level += (text[i] == "{") - (text[i] == "}")
            if not level:
                values.append(text[m.end():i])
                break
    return values[-1] if values else None

def normalize(text):
    text = re.sub(r"\s+|[{}$]|\\(?:left|right)", "", text.replace("−", "-"))
    if re.fullmatch(r"[+-]?\d+,[+-]?\d+", text) and not re.fullmatch(r"[+-]?\d{1,3},\d{3}", text):
        return tuple(sorted(int(x) for x in text.split(",")))
    return text.replace(",", "")


fixture=(d/"trace.json").read_bytes()
h=sha256(fixture).hexdigest()
traces=json.loads(fixture)["traces"]
assert sorted(traces)==["31","47","89"]
assert len({tuple((len(q["prompt_ids"]),q["max_output_tokens"]) for q in v) for v in traces.values()})==3
assert len({str(sorted(Counter((len(q["prompt_ids"]),q["max_output_tokens"]) for q in v).items())) for v in traces.values()})==1
perf={}
fields=("wall_s","output_tok_s","p95_ttft_ms","p95_e2e_ms","p95_short_ttft_ms","p95_long_ttft_ms",
        "p95_queue_ms","p95_host_submit_lag_ms","p95_nonempty_chunk_gap_ms","peak_native_waiting",
        "peak_outstanding_client","num_preemptions")
for mode,cap in (("k1",4096),("k1",2048),("dynamic",4096),("longcap",4096)):
    label=f"{mode}-{cap}"
    ds={}
    for seed,w in traces.items():
        file=d/mode/f"seed-{seed}-{cap}.json"
        if not file.is_file():
            continue
        j=json.loads(file.read_text())
        r=j["summary"]
        rows={x["id"]:x for x in j["rows"]}
        assert r["trace_sha256"]==h and r["scheduled_token_cap"]==cap and r["fixed_output"]
        assert len(w)==len(rows)==r["n"]==32
        assert sum(q["max_output_tokens"] for q in w)==r["total_output_tokens"]
        for q in w:
            x=rows[q["id"]]
            assert x["completion_tokens"]==len(x["output_ids"])==q["max_output_tokens"]
            assert x["input_tokens"]==len(q["prompt_ids"])
            assert x["planned_arrival_ms"]==q["arrival_ms"]
            assert x["submit_ms"]>=x["planned_arrival_ms"]
            assert x["first_schedule_ms"]>=x["submit_ms"]
        ds[seed]={**{k:r[k] for k in fields},"proposed_k":r["steps_by_proposed_k"],
                  "scheduled_batch":r["steps_by_scheduled_batch"],"chunks":r["chunks_by_output_size"]}
    warm=d/mode/f"warm-{cap}.json"
    warm_stats=json.loads(warm.read_text())["summary"] if warm.is_file() else None
    perf[label]={"seeds":ds,"median":{k:median(m[k] for m in ds.values()) for k in fields} if ds else None,
                 "warm_audit_graph_replays":None if warm_stats is None else warm_stats["actual_graph_replay_calls_in_audit"]}
for label in ("k1-2048","dynamic-4096","longcap-4096"):
    base=perf["k1-4096"]["seeds"]
    new=perf[label]["seeds"]
    pct={s:{k:100*(new[s][k]/base[s][k]-1) for k in ("output_tok_s","p95_ttft_ms","p95_e2e_ms","p95_short_ttft_ms","p95_long_ttft_ms") if base[s][k]>0}
         for s in sorted(base.keys()&new.keys())}
    perf[label]["paired_percent"]=pct

gold={}
for p in sorted((d/"dev").glob("*.jsonl")):
    for line in p.read_text().splitlines():
        v=json.loads(line)
        if v.get("answer") is not None:
            gold[v["id"]]=v["answer"]
quality={}
for mode,cap in (("k1",4096),("k1",2048),("longcap",4096)):
    path=d/mode/f"quality-{cap}.json"
    if not path.is_file():
        continue
    raw=json.loads(path.read_text())
    marks={}
    for row in raw["rows"]:
        if row["id"] in gold:
            x=boxed(row["text"])
            marks[row["id"]]=(x is not None and normalize(x)==normalize(gold[row["id"]]))
    assert len(marks)==8
    quality[f"{mode}-{cap}"]={"math":sum(marks.values()),"marks":marks,"run":raw["summary"]}
print(json.dumps({"fixture_sha256":h,"performance":perf,"quality":quality},indent=2,ensure_ascii=False))

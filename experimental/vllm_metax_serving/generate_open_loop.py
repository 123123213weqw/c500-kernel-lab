"""Freeze identical open-loop traffic for all vLLM engine arms."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from transformers import AutoTokenizer

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--model",type=Path,required=True)
p.add_argument("--out",type=Path,required=True)
a=p.parse_args()
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
shape_pairs=((96,32),(128,256),(512,96),(1024,64),
             (2048,48),(3072,32),(1024,256),(2048,384))
questions=("Describe ways to verify an observation at a weather station.",
           "Summarize the role of timestamps and measurement units in a research record.",
           "Explain practical methods to detect transcription errors and assess uncertainty.",
           "Discuss the tradeoffs between tabular and narrative field notes.")
def make(g, seed, idx, length):
    text=f"Study tag {seed}-{idx}. {questions[idx % len(questions)]} Write an extended technical explanation."
    end=tok.apply_chat_template([{"role":"user","content":"PLACEMARKu83L\n"+text}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    l,r=end.split("PLACEMARKu83L")
    encode=lambda x:tok.encode(x,add_special_tokens=False)
    l,r=encode(l),encode(r)
    pool=encode("Earlier field notes cover water levels, measurement intervals, archive review and calibration. "*550)
    n=length-len(l)-len(r)
    assert 0<=n<=len(pool)
    return l+pool[:n]+r

traces={}
for seed in (31,47,89):
    g=random.Random(seed)
    shapes=list(shape_pairs)*4
    g.shuffle(shapes)
    # Light lead-in, burst, steadier arrivals. Wall time is not coupled to completions.
    arrival=0.
    rows=[]
    for i,(length,cap) in enumerate(shapes):
        if i:
            lo,hi=(.55,1.08) if i<6 else ((.10,.27) if i<24 else (.31,.82))
            arrival+=g.uniform(lo,hi)
        ids=make(g,seed,i,length)
        assert len(ids)==length
        rows.append({"id":f"r{seed}-{i:02d}","arrival_ms":round(arrival*1000),
                     "prompt_ids":ids,"max_output_tokens":cap})
    assert Counter((len(r["prompt_ids"]),r["max_output_tokens"]) for r in rows)==Counter(shapes)
    traces[str(seed)]=rows
a.out.write_text(json.dumps({"shape_pairs":shape_pairs,"traces":traces},separators=(",",":"))+"\n")
print("TRACE_SHA256",hashlib.sha256(a.out.read_bytes()).hexdigest(),flush=True)

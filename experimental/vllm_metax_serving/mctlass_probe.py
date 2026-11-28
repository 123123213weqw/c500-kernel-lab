"""C500 hardware probe: real BF16 weight projections, synthetic activations."""
import argparse
import json
from pathlib import Path
from statistics import median
import sys

p=argparse.ArgumentParser(description=__doc__)
for k in ("model","pkg","out"):
    p.add_argument("--"+k,type=Path,required=True)
a=p.parse_args()
assert a.model.joinpath("VERIFIED").is_file()
sys.path.insert(0,str(a.pkg))
import torch
import torch.nn.functional as F
from safetensors import safe_open
from c500_specpilot_engine import MctlassW8A8Linear
assert torch.cuda.device_count()==1 and "C500" in torch.cuda.get_device_name()
index=json.loads((a.model/"model.safetensors.index.json").read_text())["weight_map"]

def weight(part):
    key=f"model.language_model.layers.0.mlp.{part}_proj.weight"
    with safe_open(a.model/index[key],framework="pt",device="cpu") as f:
        z=f.get_tensor(key).to("cuda")
    assert z.dtype==torch.bfloat16
    return z

def capture(fn):
    for _ in range(4):
        out=fn()
    torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out=fn()
    g.replay()
    torch.cuda.synchronize()
    return g,out

def measure(g,iters=40):
    e1,e2=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    e1.record()
    for _ in range(iters):
        g.replay()
    e2.record(); e2.synchronize()
    return 1000*e1.elapsed_time(e2)/iters

torch.manual_seed(901)
profile=[]
with torch.inference_mode():
    for proj in ("gate_up","down"):
        w=torch.cat([weight("gate"),weight("up")],dim=0) if proj=="gate_up" else weight("down")
        quant=MctlassW8A8Linear(w)
        for m in (16,32,64):
            x=torch.randn(m,w.shape[1],device="cuda",dtype=torch.bfloat16)*.05
            A,oa=capture(lambda:F.linear(x,w))
            B,ob=capture(lambda:quant.forward(x))
            f,g=oa.float(),ob.float()
            assert f.isfinite().all().item() and g.isfinite().all().item()
            err=torch.linalg.vector_norm(g-f).item()/max(torch.linalg.vector_norm(f).item(),1e-9)
            pairs=[]
            for i in range(5):
                timing={}
                for k,graph in ((("bf16",A),("w8a8",B)) if i%2==0 else (("w8a8",B),("bf16",A))):
                    timing[k]=measure(graph)
                pairs.append(timing)
            row={"projection":proj,"layer":0,"m":m,"weight_shape":list(w.shape),"rel_l2":err,"pairs_us":pairs,
                 "bf16_us":median(r["bf16"] for r in pairs),"w8a8_us":median(r["w8a8"] for r in pairs),
                 "speed_ratio":median(r["bf16"]/r["w8a8"] for r in pairs)}
            profile.append(row)
            print(row,flush=True)
            del A,B,oa,ob,f,g,x
        del quant,w
a.out.write_text(json.dumps(profile,indent=2)+"\n")

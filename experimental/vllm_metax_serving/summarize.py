"""Validate fixed outputs and report observations; this is not a release gate."""
import argparse
from hashlib import sha256
from pathlib import Path
from statistics import median
import json
import math

CAP = 160
CONCURRENCY = 16

def checked_sample(data, expected):
    summary = data["summary"]
    rs = data["rows"]
    ids = [r["id"] for r in rs]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError("request IDs differ from fixture")
    if summary["max_tokens"] != CAP or summary["concurrency"] != CONCURRENCY:
        raise ValueError("different protocol or concurrency")
    if summary.get("fixed") is False:
        raise ValueError("EOS run must not be ranked as fixed output")
    total = CAP * len(expected)
    for r in rs:
        if r["prompt_tokens"] != expected[r["id"]] or r["completion_tokens"] != CAP:
            raise ValueError("input/output counts differ from fixture")
        if len(r["output_ids"]) != CAP:
            raise ValueError("wrong number of output IDs")
    t = summary["wall_s"]
    if not math.isfinite(t) or t <= 0 or summary["total_output_tokens"] != total:
        raise ValueError("invalid timing numerator/denominator")
    return {"wall_s": t, "output_tokens": total, "output_tok_s": total/t,
            "mean_ttft_ms": summary["mean_ttft_ms"], "mean_tpot_ms": summary["mean_tpot_ms"]}

def collect(root):
    payload = (root/"workload.json").read_bytes()
    specs = json.loads(payload)["specs"]
    expected = {r["id"]:len(r["prompt_ids"]) for r in specs}
    if len(specs) != len(expected) or len(expected) != 20:
        raise ValueError("expected twenty unique requests")
    evidence = {}
    for mode in ("ar", "mtp1", "specpilot"):
        lane = root/mode
        samples = [checked_sample(json.loads(p.read_text()), expected)
                   for p in sorted(lane.glob("fixed-*.json"))]
        evidence[mode] = {"n":len(samples), "samples":samples,
                          "median_tok_s":median(x["output_tok_s"] for x in samples) if samples else None}
    return {"fixture_sha256":sha256(payload).hexdigest(),
            "protocol":{"concurrency":CONCURRENCY,"output_per_request":CAP,
                        "measurement":"warm in-process closed-loop E2E",
                        "release_performance_gate_passed":False},
            "routes":evidence}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, help="Artifact root with workload and three independent runner folders")
    args = p.parse_args()
    print(json.dumps(collect(args.run),indent=2,ensure_ascii=False,allow_nan=False))

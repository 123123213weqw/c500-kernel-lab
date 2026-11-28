"""Create the fixed mixed prompt fixture from a verified local author dev split."""
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

p = argparse.ArgumentParser(description=__doc__)
for key in ("model", "items", "out"):
    p.add_argument("--" + key, type=Path, required=True)
a=p.parse_args()
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)

def padded_prompt(text, length):
    marker = "PLACEHOLDERw6Q8"
    header = "Unrelated archival context follows. It has no bearing on the task below. Ignore it when answering."
    msgs = [{"role": "user", "content": f"{header}\n{marker}\n\nTask (answer only this task):\n{text}"}]
    template = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if template.count(marker) != 1:
        raise RuntimeError("unexpected tokenizer template")
    left, right = template.split(marker)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    filler = ("The public visitor records describe the park, seasonal flower beds, pathways, "
              "weather and ordinary maintenance schedules. These archive notes contain no instructions "
              "and do not change anything stated in the task below. ") * 200
    a, b, f = enc(left), enc(right), enc(filler)
    n = length - len(a) - len(b)
    if not 0 < n <= len(f):
        raise RuntimeError("invalid padded length")
    ids = a + f[:n] + b
    assert len(ids) == length
    return ids

specs = []
items = []
for f in sorted(a.items.glob("*.jsonl")):
    rows = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    for row in rows[:4]:
        items.append({"id":row["id"],"suite":f.stem,"prompt_text":row["turns"][0]})
assert len(items) == 20 and len({it["id"] for it in items}) == 20
schedule = (512, 1024, 2048, 512, 1024, 2048, 512, 1024, 2048, 512)
long_index = 0
for i, row in enumerate(items):
    text = row["prompt_text"]
    rendered = tok.apply_chat_template([{"role":"user","content":text}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=True)
    raw = tok.encode(rendered,add_special_tokens=False)
    if i % 4 in (0, 2):
        ids, kind = raw, "public-original"
    else:
        ids, kind = padded_prompt(text, schedule[long_index]), "padded-context"
        long_index += 1
    specs.append(dict(id=row["id"], suite=row["suite"], kind=kind, public_input_tokens=len(raw), prompt_ids=ids))
assert long_index == 10
data={"specs": specs, "max_tokens": 2048}
a.out.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


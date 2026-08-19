"""Dependency-free syntax and publication hygiene audit (no GPU imports)."""
import ast,pathlib,re,json,hashlib,sys
root=pathlib.Path(__file__).resolve().parent.parent
patterns=[r'(?<![A-Za-z0-9.])(?:\d{1,3}\.){3}\d{1,3}\b',r'/Users/[A-Za-z0-9_]+/',r'/data/'+'codex-bench/',r'root\+vm-[A-Za-z0-9]+',r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',r'(?i)(?:password|passwd)\s*[:=]\s*[\"\x27][^\"\x27]{3,}']
failures=[];files=[];npy=0
for p in sorted(root.rglob('*')):
 if not p.is_file() or any(z in p.parts for z in ['.git','__pycache__']):continue
 rel=p.relative_to(root).as_posix()
 if p.suffix in ['.pt','.pth','.safetensors','.whl','.deb','.pem','.key'] or p.name.startswith('.env') or p.stat().st_size>2**20:failures.append((rel,'forbidden artifact'))
 text=p.read_text()
 if p.suffix=='.py':ast.parse(text,filename=rel);npy+=1
 for pattern in patterns:
  matches=re.findall(pattern,text)
  if any(v not in {'3.3.0.15','3.5.3.20','3.5.3.9','1.33.5.0'} for v in matches):failures.append((rel,'private endpoint/path/key pattern'))
 files.append({'path':rel,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
print(json.dumps({'files':len(files),'python_ASTs':npy,'failures':failures},indent=2))
if failures:sys.exit(1)

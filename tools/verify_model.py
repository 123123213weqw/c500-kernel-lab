"""Read-only checkpoint integrity check; CPU only, no downloads or GPU imports."""
import argparse,hashlib,json,pathlib

def verify(model,manifest):
 model=pathlib.Path(model).resolve();rows=manifest['verified_files'];seen=set();result=[]
 assert model.is_dir() and rows
 for row in rows:
  name=pathlib.PurePosixPath(row['path'])
  assert not name.is_absolute() and '..' not in name.parts and str(name) not in seen,'Unsafe or duplicate manifest path'
  seen.add(str(name));p=(model/str(name)).resolve();assert p.is_relative_to(model),'Symlink escapes checkpoint root'
  assert p.is_file() and p.stat().st_size==row['bytes'],'Missing file or size mismatch'
  h=hashlib.sha256()
  with p.open('rb') as f:
   for block in iter(lambda:f.read(8<<20),b''):h.update(block)
  assert h.hexdigest()==row['sha256'],'Checkpoint SHA256 mismatch'
  result.append(str(name))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=pathlib.Path);p.add_argument('--manifest',required=True,type=pathlib.Path);a=p.parse_args()
 names=verify(a.model,json.loads(a.manifest.read_text()));print(json.dumps({'all_sha_match':True,'verified_files':len(names)},indent=2))

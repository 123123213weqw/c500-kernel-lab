"""Own-process-group deadline and single-GPU lock; standard library only."""
import argparse,datetime,fcntl,json,os,pathlib,signal,subprocess,time

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--deadline',required=True,help='UTC ISO8601 or Unix seconds; at most one hour ahead')
 p.add_argument('--workdir',required=True,type=pathlib.Path)
 p.add_argument('--shared-lock',action='append',default=[],type=pathlib.Path)
 p.add_argument('command',nargs=argparse.REMAINDER)
 a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
 if not cmd:p.error('command required after --')
 try:deadline=float(a.deadline)
 except ValueError:deadline=datetime.datetime.fromisoformat(a.deadline.replace('Z','+00:00')).timestamp()
 if not time.time()<deadline<=time.time()+3600:p.error('deadline expired or more than one hour away')
 root=a.workdir.resolve();root.mkdir(parents=True,exist_ok=True)
 locks=[]
 for path in [root/'controller.lock']+a.shared_lock:
  h=path.open('a' if path==root/'controller.lock' else 'r');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(h)
 env=dict(os.environ,C500_DEADLINE=str(deadline),CUDA_VISIBLE_DEVICES='0',MACA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
 for key,sub in {'XDG_CACHE_HOME':'cache','PIP_CACHE_DIR':'pip-cache','TMPDIR':'tmp','TORCH_EXTENSIONS_DIR':'torch-extensions','TORCHINDUCTOR_CACHE_DIR':'inductor-cache','VLLM_CACHE_ROOT':'vllm-cache','TRITON_CACHE_DIR':'triton-cache'}.items():
  path=root/sub;path.mkdir(exist_ok=True);env[key]=str(path)
 stop=False
 def signal_stop(*_):
  nonlocal stop
  stop=True
 old={s:signal.signal(s,signal_stop) for s in [signal.SIGTERM,signal.SIGINT]}
 start=time.time();proc=subprocess.Popen(cmd,env=env,start_new_session=True)
 reason='exit'
 try:
  while proc.poll() is None:
   if stop or time.time()>=deadline-5:
    reason='signal' if stop else 'deadline';os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL)
    break
   time.sleep(.1)
  rc=proc.wait()
 finally:
  if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
  for s,h in old.items():signal.signal(s,h)
 # A child can exit while leaving descendants in its own process group.
 # This group was created by us; never signal a caller-provided arbitrary PID.
 leftover_group=False
 try:
  os.killpg(proc.pid,0);leftover_group=True;os.killpg(proc.pid,signal.SIGTERM)
  time.sleep(.2)
  try:os.killpg(proc.pid,signal.SIGKILL)
  except ProcessLookupError:pass
 except ProcessLookupError:pass
 record={'leftover_own_group_cleanup':leftover_group,'argv':cmd,'pid':proc.pid,'deadline_unix':deadline,'started_unix':start,'finished_unix':time.time(),'returncode':proc.returncode,'reason':reason,'gpu_idle_requires_separate_read_only_audit':True}
 with (root/'launches.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 return rc if reason=='exit' else 124
if __name__=='__main__':raise SystemExit(main())

import copy,importlib.util,pathlib,unittest,json,subprocess,sys,tempfile,time
ROOT=pathlib.Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('qualify',ROOT/'tools/qualify.py');q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
def gate(b=1,l=13):
 return dict(batch=b,input_len=l,logits_finite=True,full_vocab_prefill_logits_exact=True,full_vocab_decode_logits_exact=True,complete_ids_exact=True,cache_exact={k:True for k in q.FIELDS},fresh_request_probes=[{k:True for k in ['complete_ids_exact','full_vocab_logits_exact','all_caches_exact','all_logits_finite']} for _ in range(2)])
class Contracts(unittest.TestCase):
 def test_exact_gate(self):self.assertTrue(q.gate_pass(gate()))
 def test_logits_not_just_ids(self):
  g=gate();g['full_vocab_decode_logits_exact']=False;self.assertFalse(q.gate_pass(g))
 def test_unchecked_cache_rejected(self):
  g=gate();g['cache_exact']['conv_states']=None;self.assertFalse(q.gate_pass(g))
 def test_missing_request_probe_rejected(self):
  g=gate();g['fresh_request_probes'].pop();self.assertFalse(q.gate_pass(g))
 def fixture(self):
  rows=[]
  for mode,t in [('native',2.),('segmented',1.)]:
   for i in range(3):rows.append(dict(batch=1,mode=mode,repeat=i,input_len=128,output_len=128,output_tokens=128,e2e_s=t,e2e_output_tokens_per_s=128/t,input_sha256='a',complete_output_sha256='b',all_ids_exact_vs_warmup=True,detokenized_text_exact_vs_warmup=True,dispatch_counts={'fallback':0,'fused_row_major':0 if mode=='native' else 10},peak_allocated_gib=52))
  return rows,[gate(),gate(3,127),gate(17,32)],[dict(batch=1,mode=m,raw_full_vocab_finite_calls=128) for m in ['native','segmented']]
 def test_paired_ratio(self):
  self.assertEqual(q.summarize(*self.fixture(),'segmented')[0]['speedup'],2.)
 def test_fewer_than_three_rejected(self):
  r,g,w=self.fixture();r.pop()
  with self.assertRaises(AssertionError):q.summarize(r,g,w,'segmented')
 def test_fallback_rejected(self):
  r,g,w=self.fixture();r[-1]['dispatch_counts']['fallback']=1
  with self.assertRaises(AssertionError):q.summarize(r,g,w,'segmented')
 def test_regression_is_preserved(self):
  r,g,w=self.fixture()
  for x in r:
   x['e2e_s']=1. if x['mode']=='native' else 2.;x['e2e_output_tokens_per_s']=128/x['e2e_s']
  self.assertEqual(q.summarize(r,g,w,'segmented')[0]['speedup'],.5)
 def test_missing_full_finite_warmup_rejected(self):
  r,g,w=self.fixture();w[-1]['raw_full_vocab_finite_calls']=127
  with self.assertRaises(AssertionError):q.summarize(r,g,w,'segmented')
 def test_changed_actual_outputs_rejected(self):
  r,g,w=self.fixture();r[-1]['all_ids_exact_vs_warmup']=False
  with self.assertRaises(AssertionError):q.summarize(r,g,w,'segmented')
 def test_nonfinite_timing_rejected(self):
  r,g,w=self.fixture();r[-1]['e2e_s']=float('nan')
  with self.assertRaises(AssertionError):q.summarize(r,g,w,'segmented')
 def test_stale_dispatch_counter_rejected(self):
  r,g,w=self.fixture()
  for x in r:x['dispatch_counts']['fused_row_major']=10
  with self.assertRaises(AssertionError):q.summarize(r,g,w,'segmented')
 def test_different_repeat_indices_rejected(self):
  r,g,w=self.fixture();r[-1]['repeat']=4
  with self.assertRaises(AssertionError):q.summarize(r,g,w,'segmented')
 def test_different_inputs_rejected(self):
  r,g,w=self.fixture();r[-1]['input_sha256']='different'
  with self.assertRaises(AssertionError):q.summarize(r,g,w,'segmented')
 def test_wrong_token_count_rejected(self):
  r,g,w=self.fixture();r[-1]['output_tokens']=256
  with self.assertRaises(AssertionError):q.summarize(r,g,w,'segmented')
 def test_timeout_owns_only_child_group(self):
  with tempfile.TemporaryDirectory() as d:
   unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
   self.addCleanup(lambda: unrelated.terminate() if unrelated.poll() is None else None)
   p=subprocess.run([sys.executable,str(ROOT/'tools/launch_bounded.py'),'--deadline',str(time.time()+5.5),'--workdir',d,'--',sys.executable,'-c','import time;time.sleep(30)'],capture_output=True,text=True,timeout=10)
   self.assertEqual(p.returncode,124,p.stderr);self.assertIsNone(unrelated.poll())
   unrelated.terminate();unrelated.wait(timeout=3)
   rec=json.loads((pathlib.Path(d)/'launches.jsonl').read_text());self.assertEqual(rec['reason'],'deadline')
 def test_expired_deadline_starts_nothing(self):
  with tempfile.TemporaryDirectory() as d:
   p=subprocess.run([sys.executable,str(ROOT/'tools/launch_bounded.py'),'--deadline','1','--workdir',d,'--',sys.executable,'-c','raise Exception("must not run")'],capture_output=True,text=True)
   self.assertNotEqual(p.returncode,0);self.assertFalse((pathlib.Path(d)/'launches.jsonl').exists())
if __name__=='__main__':unittest.main()

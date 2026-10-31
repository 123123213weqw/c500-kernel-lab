import ast,hashlib,json,pathlib,unittest
ROOT=pathlib.Path(__file__).resolve().parent.parent
class ReducedTrafficPublication(unittest.TestCase):
 def test_measured_source_hashes(self):
  folder=ROOT/'experimental/fused_reduce_v2';c=json.loads((folder/'FROZEN_B48.json').read_text())
  self.assertEqual(c['config']['block_v'],128);self.assertEqual(c['config']['warps'],4)
  for name,h in c['source_sha256'].items():self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),h)
 def test_strategy_imports_exist(self):
  tree=ast.parse((ROOT/'benchmarks/e2e_reduce_v2.py').read_text())
  names={a.name for node in ast.walk(tree) if isinstance(node,ast.Import) for a in node.names}
  self.assertIn('exact_gdn_reduce_v2',names);self.assertIn('reduce_graph_engine_v2',names)
  for name in ['exact_gdn_reduce_v2','reduce_graph_engine_v2']:
   self.assertTrue((ROOT/'experimental/fused_reduce_v2'/(name+'.py')).is_file())
 def test_no_broad_default_or_small_batch_claim(self):
  c=json.loads((ROOT/'experimental/fused_reduce_v2/FROZEN_B48.json').read_text())
  self.assertIs(c['not_default'],True);self.assertEqual(c['qualified_shape'],dict(batch=48,input_len=128,output_len=128))
if __name__=='__main__':unittest.main()

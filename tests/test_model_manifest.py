import unittest,tempfile,pathlib,importlib.util,hashlib
ROOT=pathlib.Path(__file__).resolve().parent.parent
s=importlib.util.spec_from_file_location('verify_model',ROOT/'tools/verify_model.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
class Manifest(unittest.TestCase):
 def fixture(self,d):
  root=pathlib.Path(d);(root/'sample.bin').write_bytes(b'checkpoint fixture')
  return root,{'verified_files':[{'path':'sample.bin','bytes':18,'sha256':hashlib.sha256(b'checkpoint fixture').hexdigest()}]}
 def test_hash(self):
  with tempfile.TemporaryDirectory() as d:
   r,m=self.fixture(d);self.assertEqual(v.verify(r,m),['sample.bin'])
 def test_modified_weights_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r,m=self.fixture(d);(r/'sample.bin').write_bytes(b'CHECKPOINT FIXTURE')
   with self.assertRaises(AssertionError):v.verify(r,m)
 def test_traversal_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r,m=self.fixture(d);m['verified_files'][0]['path']='../sample.bin'
   with self.assertRaises(AssertionError):v.verify(r,m)
 def test_duplicate_path_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r,m=self.fixture(d);m['verified_files']*=2
   with self.assertRaises(AssertionError):v.verify(r,m)

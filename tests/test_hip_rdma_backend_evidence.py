"""Reject incomplete numerical/lifecycle evidence from the live backend test."""
import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from run_hip_rdma_graph import verify_backend_artifact


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        return {'status':'passed','mode':'ring4','rank':0,'run_id':'a'*32,
                **{k:True for k in ('exact','finite','unsupported_shape_falls_back','graphs_destroyed','backend_closed','pynccl_destroyed','cpu_group_destroyed')},
                'operations_per_graph':32,'iterations':32,'samples':3,'warmup':4,'graph_output_checks':224,
                'wall_samples_us':[56.,57.,58.],'hip_event_samples_us':[55.,56.,57.],
                'parity_proof':{'banks':32,'elements_per_rank':10240,
                    **{k:True for k in ('exact_eager','exact_graph','validated_against_active_pynccl','repeat_bank0','graph_destroyed')},
                    'rank_results':[{'rank':r,'errors':[]} for r in range(4)]}}
    def valid(self,a): return verify_backend_artifact(a,{'rank':0,'run_id':'a'*32},32,3)
    def test_complete_evidence_passes(self): self.assertTrue(self.valid(self.fixture()))
    def test_identity_cleanup_and_nonfinite_time_rejected(self):
        for key,value in [('rank',1),('run_id','b'*32),('backend_closed',False),
                          ('graph_output_checks',223),('hip_event_samples_us',[56.,float('nan'),56.])]:
            with self.subTest(key=key):
                a=self.fixture();a[key]=value;self.assertFalse(self.valid(a))
    def test_any_rank_or_execution_mode_failure_rejected(self):
        for key in ('exact_eager','exact_graph','validated_against_active_pynccl'):
            a=self.fixture();a['parity_proof'][key]=False;self.assertFalse(self.valid(a))
        a=self.fixture();a['parity_proof']['rank_results'][3]['errors']=[{'mismatches':1}]
        self.assertFalse(self.valid(a))
        a=self.fixture();a['parity_proof']['rank_results'][3]['rank']=2
        self.assertFalse(self.valid(a))


if __name__=='__main__': unittest.main()

"""CPU checks for the merged CFD and graph-ablation switches."""
import unittest
from vf_fixture import install_source
install_source()
import torch
from pieef.models.graph import DynamicDirectionalGraphLayer

class ReleaseExtensionChecks(unittest.TestCase):
    def test_self_only_preserves_parameters_and_attention_is_identity(self):
        torch.manual_seed(13)
        common=dict(d_model=8,score_policy='cosine_temperature',use_direction_bias=False)
        all_nodes=DynamicDirectionalGraphLayer(**common,neighbour_policy='all')
        self_only=DynamicDirectionalGraphLayer(**common,neighbour_policy='self_only')
        self_only.load_state_dict(all_nodes.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in all_nodes.parameters()),sum(p.numel() for p in self_only.parameters()))
        x=torch.randn(2,3,8);power=torch.rand(2,3,1)
        self_only(x,power,power,torch.zeros(3,3),torch.zeros(3,3,2),torch.ones(2,2))
        torch.testing.assert_close(self_only.last_attn_weights,torch.eye(3).expand(2,-1,-1),rtol=0,atol=0)

    def test_local_uniform_direction_reproduces_farm_vector_alignment(self):
        coords=torch.tensor([[0.,0.],[400.,100.],[850.,-80.]])
        relative=coords[:,None]-coords[None,:]
        farm=torch.tensor([[.6,.8],[0.,1.]])
        local=farm[:,None,:].expand(-1,3,-1)
        a=DynamicDirectionalGraphLayer.direction_alignment(farm,relative)
        b=DynamicDirectionalGraphLayer.direction_alignment(local,relative)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        opposed=torch.tensor([[[1.,0.],[-1.,0.]]])
        pair=DynamicDirectionalGraphLayer.direction_alignment(opposed,relative[:2,:2])
        self.assertEqual(float(pair[0,0,1,0]),0.)
        self.assertEqual(float(pair[0,1,0,0]),0.)

if __name__=='__main__':unittest.main()

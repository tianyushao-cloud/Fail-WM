import unittest
import numpy as np
import torch
from futureworlds.reliable_pair import select_pair
from futureworlds.objective import compute_grpo_outcome_advantage

class ReliablePairChecks(unittest.TestCase):
 def base(self):
  return np.array([[[.95,.85,1],[.80,.60,1],[.9,.75,1],[.9,.8,1]],
                   [[.96,.86,1],[.81,.61,1],[.7,.45,1],[.9,.8,1]]])
 def test_bad_third_candidate_does_not_discard_good_pair(self):
  r,s=select_pair(self.base())
  self.assertEqual(s["chosen"]["winner"],0);self.assertEqual(s["chosen"]["loser"],1)
  np.testing.assert_array_equal(r,[1,-1,0,0])
  terminal=torch.zeros(4,80);terminal[:,-1]=torch.tensor(r)
  adv,_=compute_grpo_outcome_advantage(terminal,torch.ones_like(terminal),np.zeros(4,dtype=int))
  self.assertTrue(torch.equal(adv[2:],torch.zeros_like(adv[2:])))
  self.assertGreater(adv[0,0],0);self.assertLess(adv[1,0],0)
 def test_permutation_equivariance_and_role_swap(self):
  x=self.base();r,_=select_pair(x);p=[2,0,3,1]
  rp,_=select_pair(x[:,p]);np.testing.assert_array_equal(rp,r[p])
  rr,_=select_pair(x[::-1]);np.testing.assert_array_equal(rr,r)
 def test_tradeoff_and_reversal_abstain(self):
  x=np.array([[[.9,.8,1],[.95,.6,1]],[[.9,.8,1],[.95,.6,1]]])
  self.assertTrue(select_pair(x)[1]["rejected"])
  x=np.array([[[.9,.8,1],[.8,.6,1]],[[.8,.6,1],[.9,.8,1]]])
  self.assertTrue(select_pair(x)[1]["rejected"])
 def test_all_bad_or_identical_abstain(self):
  x=np.array([[[.9,.4,1],[.8,.2,1]],[[.9,.4,1],[.8,.2,1]]])
  self.assertTrue(select_pair(x)[1]["rejected"])
  self.assertTrue(select_pair(np.ones((2,4,3)))[1]["rejected"])
 def test_invalid_scores(self):
  x=self.base();x[0,0,0]=float("nan")
  with self.assertRaises(ValueError):select_pair(x)


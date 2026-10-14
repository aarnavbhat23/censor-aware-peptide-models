import unittest
import math
import numpy as np
import pandas as pd
import torch
from likelihood import observed_nll
from observations import parse_mic
from benchmark import purged_masks


class CensoringTests(unittest.TestCase):
    def test_micro_and_mass(self):
        for text in ['>64 µM','>64 μM','>64 uM']:
            result=parse_mic(text);self.assertEqual(result.lower,math.log10(64));self.assertTrue(math.isinf(result.upper))
        with self.assertRaises(ValueError):parse_mic('64 µg/mL')

    def test_units(self):self.assertEqual(parse_mic('1000 nM'),parse_mic('1 uM'))

    def test_tail_extreme_is_finite(self):
        mean=torch.tensor([-50.],requires_grad=True)
        loss=observed_nll(mean,torch.ones(1),torch.zeros(1),torch.tensor([float('inf')])).sum()
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(mean.grad).all())

    def test_exact_density(self):
        nll=observed_nll(torch.zeros(1),torch.ones(1),torch.zeros(1),torch.zeros(1))
        self.assertAlmostEqual(nll.item(),.5*math.log(2*math.pi),places=6)

    def test_survival_half(self):
        nll=observed_nll(torch.zeros(1),torch.ones(1),torch.zeros(1),torch.tensor([float('inf')]))
        self.assertAlmostEqual(nll.item(),math.log(2),places=6)

    def test_publication_sequence_purge(self):
        data=pd.DataFrame({'paper':['test','validation','train','train','validation'], 'sequence':['A','B','A','C','A']})
        train,val,test=purged_masks(data,'test','validation')
        np.testing.assert_array_equal(train,[False,False,False,True,False])
        np.testing.assert_array_equal(val,[False,True,False,False,False])


if __name__=='__main__':unittest.main()

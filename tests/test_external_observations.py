import math
import unittest
import pandas as pd
from chembl_benchmark import observation, masks


class ExternalObservationContracts(unittest.TestCase):
    def row(self, **values):
        return {
            "standard_units": "nM",
            "standard_type": "IC50",
            "standard_value": 1000,
            "standard_relation": ">",
            **values,
        }

    def test_explicit_unit_and_inequality(self):
        lo, hi, kind = observation(self.row())
        self.assertEqual((lo, kind), (0.0, "right"))
        self.assertEqual(hi, math.inf)
        with self.assertRaises(ValueError):
            observation(self.row(standard_units="ug/mL"))

    def test_upper_value_remains_interval(self):
        lo, hi, kind = observation(
            self.row(standard_relation="=", standard_upper_value=10000)
        )
        self.assertEqual((lo, hi, kind), (0.0, 1.0, "interval"))
        with self.assertRaises(ValueError):
            observation(self.row(standard_relation="=", standard_upper_value=500))

    def test_shared_molecule_purged(self):
        frame = pd.DataFrame(
            [
                {"document": f"d{i}", "molecule": m}
                for i in range(40)
                for m in ["shared", f"unique{i}"]
            ]
        )
        train, val, test = masks(frame, 901)
        self.assertFalse((frame.molecule[train] == "shared").any())
        self.assertFalse(set(frame.molecule[train]) & set(frame.molecule[val | test]))
        self.assertFalse(set(frame.molecule[val]) & set(frame.molecule[test]))

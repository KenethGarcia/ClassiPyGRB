import unittest
import numpy as np
import pandas as pd
from . import Data
from ClassiPyGRB import SWIFT
from importlib import resources


class TestData(unittest.TestCase):
    swift = SWIFT(res=10000, root_path=None)

    def test_query(self):
        df = self.swift.obtain_data(name='GRB180115A')
        with resources.path(Data, 'GRB180115A_sn5_10s.h5') as file:
            df2 = pd.read_hdf(file, key='GRB180115A')
        self.assertTrue(df.equals(df2))

    def test_query_invalid_name(self):
        with self.assertRaises(RuntimeError):
            self.swift.obtain_data(name='GRB010101')

    def test_download_invalid_name(self):
        with self.assertRaises(RuntimeError):
            self.swift.single_download(name='GRB010101')

    def test_limits_intervals(self):
        limits = self.swift.duration_limits(name='GRB061210')
        self.assertListEqual([list(item) for item in limits], [['GRB061210', '-0.004', '89.392']])
        limits = self.swift.duration_limits(name='GRB061210', t=50)
        self.assertListEqual([list(item) for item in limits], [['GRB061210', '6.576', '56.004']])
        limits1, limits2 = self.swift.duration_limits(name=('GRB061210', 'GRB060614'), t=50)
        self.assertListEqual([list(item) for item in limits1], [['GRB061210', '6.576', '56.004']])
        self.assertListEqual([list(item) for item in limits2], [['GRB060614', '21.116', '64.352']])
        limits = self.swift.duration_limits(name='GRB061210000', t=50)
        self.assertListEqual([list(item) for item in limits], [])

    def test_durations(self):
        dur = self.swift.total_durations(names='GRB060614')
        assert np.allclose(dur, np.array([180.576]))
        dur = self.swift.total_durations(names=['GRB061210'])
        assert np.allclose(dur, [89.396])
        dur = self.swift.total_durations(names=('GRB061210', 'GRB060614'), t=50)
        assert np.allclose(dur, [49.428, 43.236])
        with self.assertRaises(RuntimeError):
            self.swift.total_durations(names='GRB010101')

    def test_redshifts(self):
        z = self.swift.redshifts(name='GRB181020A')
        self.assertListEqual([list(item) for item in z], [['GRB181020A', '2.938']])
        z1, z2 = self.swift.redshifts(name=['GRB220611A', 'GRB220521A'])
        self.assertListEqual([list(item) for item in [z1]], [['GRB220611A', '2.3608']])
        self.assertListEqual([list(item) for item in [z2]], [['GRB220521A', '5.6']])
        z = self.swift.redshifts(name=['GRB010101'])
        self.assertListEqual([list(item) for item in z], [['GRB010101', None]])


if __name__ == "__main__":
    unittest.main()

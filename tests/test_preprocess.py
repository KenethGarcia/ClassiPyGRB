import unittest
import numpy as np
import pandas as pd
from ClassiPyGRB import SWIFT, _tools


class TestData(unittest.TestCase):
    swift = SWIFT(res=64)
    t = np.linspace(0, np.pi, 1000)
    df = pd.DataFrame({'Time(s)': t, 'sin(x)': np.sin(t), 'cos(x)': np.cos(t), 'x': t, 'y': np.ones_like(t)})

    def test_lc_limiter(self):
        df = self.swift.lc_limiter(name='GRB060510A')
        self.assertTrue(isinstance(df, tuple))
        self.assertTrue(df[0] == 'GRB060510A')
        self.assertTrue(df[1] == -6.752)
        self.assertTrue(df[2] == 16.748)
        self.assertTrue(df[3] == 'Only zeros')
        df = self.swift.lc_limiter(name='GRB050925', t=100)
        self.assertTrue(isinstance(df, tuple))
        self.assertTrue(df[0] == 'GRB050925')
        self.assertTrue(df[1] == -0.036)
        self.assertTrue(df[2] == 0.068)
        self.assertTrue(df[3] == 'Length=2')
        limits = self.swift.duration_limits(name='GRB061210')
        self.assertListEqual([list(item) for item in limits], [['GRB061210', '-0.004', '89.392']])
        df = self.swift.lc_limiter(name='GRB061210', t=100)
        self.assertTrue(isinstance(df, pd.DataFrame))
        # assert if the first element of Time column is higher than -0.004 and the last element is lower than 89.392
        self.assertTrue(df['Time(s)'].iloc[0] > -0.004)
        self.assertTrue(df['Time(s)'].iloc[-1] < 89.392)

    def test_lc_normalize(self):
        df_norm = self.swift.lc_normalize(self.df.copy(), base=1)
        self.assertTrue(self.df.shape == df_norm.shape)
        self.assertTrue(np.allclose(df_norm['Time(s)'], self.df['Time(s)']))
        self.assertTrue(np.allclose(df_norm['sin(x)'], self.df['sin(x)'] / 2))
        self.assertTrue(np.allclose(df_norm['cos(x)'], self.df['cos(x)'] / 2))
        self.assertTrue(np.allclose(df_norm['x'], self.df['x'] / 2))
        self.assertTrue(np.allclose(df_norm['y'], self.df['y'] / 2))
        df_norm = self.swift.lc_normalize(self.df.copy(), base=3)
        factor = np.square(np.pi)/2
        self.assertTrue(self.df.shape == df_norm.shape)
        self.assertTrue(np.allclose(df_norm['Time(s)'], self.df['Time(s)']))
        self.assertTrue(np.allclose(df_norm['sin(x)'], self.df['sin(x)'] / factor))
        self.assertTrue(np.allclose(df_norm['cos(x)'], self.df['cos(x)'] / factor))
        self.assertTrue(np.allclose(df_norm['x'], self.df['x'] / factor))
        self.assertTrue(np.allclose(df_norm['y'], self.df['y'] / factor))
        df_norm = self.swift.lc_normalize(self.df.copy())
        self.assertTrue(self.df.shape == df_norm.shape)
        self.assertTrue(np.allclose(df_norm['Time(s)'], self.df['Time(s)']))
        self.assertTrue(np.allclose(df_norm['sin(x)'], self.df['sin(x)'] / np.pi))
        self.assertTrue(np.allclose(df_norm['cos(x)'], self.df['cos(x)'] / np.pi))
        self.assertTrue(np.allclose(df_norm['x'], self.df['x'] / np.pi))
        self.assertTrue(np.allclose(df_norm['y'], self.df['y'] / np.pi))

    def test_zero_pad(self):
        with self.assertRaises(ValueError):
            self.swift.zero_pad(self.df.copy(), 1000)
        with self.assertRaises(ValueError):
            self.swift.zero_pad(self.df.copy(), 100)
        df_pad = self.swift.zero_pad(self.df.copy(), 1001)
        self.assertEqual(df_pad.shape[0], 1001)
        self.assertFalse('Time(s)' in df_pad.columns)
        self.assertTrue(np.allclose(df_pad.iloc[-1], np.zeros(4)))
        df_pad = self.swift.zero_pad(self.df.copy(), 1002)
        self.assertEqual(df_pad.shape[0], 1002)
        self.assertFalse('Time(s)' in df_pad.columns)
        self.assertTrue(np.allclose(df_pad.iloc[-1], np.zeros(4)))
        self.assertTrue(np.allclose(df_pad.iloc[-2], np.zeros(4)))
        self.assertFalse(np.allclose(df_pad.iloc[-3], np.zeros(4)))

    def test_concatenate(self):
        df = self.swift.concatenate(self.df.copy())
        self.assertEqual(df.shape[0], 5000)
        self.assertTrue(np.allclose(df[:1000], self.df['Time(s)']))
        self.assertTrue(np.allclose(df[1000:2000], self.df['sin(x)']))
        self.assertTrue(np.allclose(df[2000:3000], self.df['cos(x)']))
        self.assertTrue(np.allclose(df[3000:4000], self.df['x']))
        self.assertTrue(np.allclose(df[4000:5000], self.df['y']))
        df = self.swift.concatenate(np.asarray(self.df.copy()))
        self.assertEqual(df.shape[0], 5000)
        self.assertTrue(np.allclose(df[:1000], self.df['Time(s)']))
        self.assertTrue(np.allclose(df[1000:2000], self.df['sin(x)']))
        self.assertTrue(np.allclose(df[2000:3000], self.df['cos(x)']))
        self.assertTrue(np.allclose(df[3000:4000], self.df['x']))
        self.assertTrue(np.allclose(df[4000:5000], self.df['y']))
        df = self.swift.obtain_data('GRB060614')
        df_con = self.swift.concatenate(df.copy())
        self.assertEqual(df_con.shape[0], df.shape[0]*11)
        self.assertTrue(np.allclose(df_con[:df.shape[0]], df['Time(s)']))
        self.assertTrue(np.allclose(df_con[df.shape[0]:2*df.shape[0]], df['15-25keV']))

    def test_dft_spectrum(self):
        df = self.swift.zero_pad(self.df.copy(), 10000)
        df = self.swift.concatenate(df)
        df_dft = self.swift.dft_spectrum(df)
        self.assertEqual(df_dft.shape[0], 20000)

    def test_interpolation(self):
        times = np.linspace(0, 2 * np.pi, 10)  # Defining times
        cols = np.array([times, np.sin(times), np.cos(times), np.power(times, 3 / 2), np.square(times), np.sqrt(times)])

        df = pd.DataFrame({
            "times": cols[0],
            "sin": cols[1],
            "cos": cols[2],
            "times^3": cols[3],
            "times^2": cols[4],
            "sqrt(times)": cols[5]
        })

        new_times = np.linspace(1, 5, 100)
        inter_data = self.swift.grb_interpolate(data=df, new_time=new_times, kind='cubic', pack_num=5)
        self.assertEqual(inter_data.shape[0], 100)
        self.assertEqual(inter_data.shape[1], 6)
        self.assertTrue(np.allclose(inter_data['times'], new_times, atol=1e-2, rtol=1e-2))
        self.assertTrue(np.allclose(inter_data['sin'], np.sin(new_times), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.allclose(inter_data['cos'], np.cos(new_times), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.allclose(inter_data['times^3'], np.power(new_times, 3 / 2), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.allclose(inter_data['times^2'], np.square(new_times), atol=1e-2, rtol=1e-2))
        self.assertTrue(np.allclose(inter_data['sqrt(times)'], np.sqrt(new_times), atol=1e-2, rtol=1e-2))

    def test_noise_reduction(self):
        df = self.swift.obtain_data('GRB060614')
        with self.assertRaises(TypeError):
            self.swift.noise_reduction_fabada('GRB060614', '0')
        df_noise = self.swift.noise_reduction_fabada('GRB060614', save_data=False)
        self.assertTrue(np.allclose(df_noise['Time(s)'], df['Time(s)']))
        for col in df.columns[1::2]:
            self.assertEqual(df[col].shape[0], df_noise[col].shape[0])
            sig1 = _tools.estimate_noise(np.asarray([df[col]]))
            sig2 = _tools.estimate_noise(np.asarray([df_noise[col]]))
            self.assertTrue(np.abs(sig2) <= np.abs(sig1))


if __name__ == "__main__":
    unittest.main()

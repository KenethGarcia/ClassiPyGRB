import unittest
import numpy as np
import pandas as pd
from importlib import resources
from ClassiPyGRB import SWIFT, summary_tables


class TestSWIFT(unittest.TestCase):
    def test_init_default(self):
        swift = SWIFT(root_path='/home')
        self.assertEqual(swift.res, 64)
        self.assertEqual(swift.end, '64ms')
        self.assertEqual(swift.n_bands, (1, 2, 3, 4))
        self.assertEqual(swift.root_path, '/home')
        self.assertEqual(swift.data_path, '/home/Data')
        self.assertEqual(swift.original_data_path, '/home/Data/Original_Data')
        self.assertEqual(swift.noise_data_path, '/home/Data/Noise_Filtered_Data')
        self.assertEqual(swift.results_path, '/home/Results')
        self.assertEqual(swift.bands_selected, ['Time(s)', '15-25keV', '25-50keV', '50-100keV', '100-350keV'])

    def test_init_custom(self):
        swift = SWIFT('/home/Documents', 10000, [5], '/home/Documents/Data', '/home/Documents/Data/Original_Data',
                      '/home/Documents/Data/Noise_Filtered_Data', '/home/Documents/Results')
        self.assertEqual(swift.res, 10000)
        self.assertEqual(swift.end, 'sn5_10s')
        self.assertEqual(swift.n_bands, [5])
        self.assertEqual(swift.root_path, '/home/Documents')
        self.assertEqual(swift.data_path, '/home/Documents/Data')
        self.assertEqual(swift.original_data_path, '/home/Documents/Data/Original_Data')
        self.assertEqual(swift.noise_data_path, '/home/Documents/Data/Noise_Filtered_Data')
        self.assertEqual(swift.results_path, '/home/Documents/Results')
        self.assertEqual(swift.bands_selected, ['Time(s)', '15-350keV'])

    def test_init_invalid_res(self):
        with self.assertRaises(ValueError):
            SWIFT(root_path='/home', res=3)

    def test_init_invalid_n_bands(self):
        with self.assertRaises(ValueError):
            SWIFT(root_path='/home', n_bands=[0, 1, 2])

    def test_changing_column_names(self):
        swift = SWIFT()
        swift.column_labels = ('time', '15-25', '15-25', '25-50', '25-50', '50-100', '50-100', '100-350', '100-350',
                               '15-350', '15-350')
        swift.__init__(n_bands=(1, 2))
        self.assertEqual(swift.bands_selected, ['time', '15-25', '25-50'])

    def test_summary_table(self):
        swift = SWIFT()
        expected_columns = ['GRBname', 'Trig_ID', 'Trig_time_met', 'Trig_time_UTC', 'RA_ground', 'DEC_ground',
                            'Image_position_err', 'Image_SNR', 'T90', 'T90_err', 'T50', 'T50_err',
                            'Evt_start_sincetrig', 'Evt_stop_sincetrig', 'pcode', 'Trigger_method', 'XRT_detection',
                            'comment']
        with resources.open_text(summary_tables, 'summary_general.txt') as summary:
            table = np.genfromtxt(summary, delimiter="|", dtype=str, unpack=True, autostrip=True)
        expected_output = pd.DataFrame()
        for i in range(len(table)):
            expected_output[expected_columns[i]] = table[i]
        output = swift.summary_table()
        assert isinstance(output, pd.DataFrame), f"Expected a pandas DataFrame, but got {type(output)}."
        assert output.equals(expected_output), f"The output DataFrame does not match the expected output.\nExpected:" \
                                               f"\n{expected_output}\nActual:\n{output}\n"
        assert list(output.columns) == expected_columns, f"The column names in the output DataFrame do not match the " \
                                                         f"expected column names.\nExpected:\n{expected_columns}\n" \
                                                         f"Actual:\n{list(output.columns)}\n"


if __name__ == '__main__':
    unittest.main()

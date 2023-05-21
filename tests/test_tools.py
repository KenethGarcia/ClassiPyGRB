import unittest
import numpy as np
import moviepy.editor as mpy
from ClassiPyGRB import SWIFT


class TestData(unittest.TestCase):
    swift = SWIFT(res=64)
    example = 'GRB060614'
    x = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

    def test_get_flux(self):
        flux = self.swift.get_flux(self.example, band=3, t=None, limits=[-1, 100])
        self.assertTrue(isinstance(flux, (int, float)))
        self.assertTrue(np.isclose(flux, 7.12, atol=1e-3))
        with self.assertRaises(RuntimeError):
            self.swift.get_flux('GRB060611', band=3, t=None, limits=[-1, 100])

    def test_hardness_ratio(self):
        hr = self.swift.hardness_ratio(self.example)
        self.assertTrue(isinstance(hr, (int, float)))
        self.assertTrue(np.isclose(hr, 0.547, atol=1e-3))
        with self.assertRaises(RuntimeError):
            self.swift.hardness_ratio('GRB060611')

    def test_tsne(self):
        pos = self.swift.perform_tsne(self.x, perplexity=1, learning_rate=100, n_iter=1000)
        self.assertTrue(isinstance(pos, np.ndarray))
        self.assertTupleEqual(pos.shape, (4, 2))
        pos = self.swift.perform_tsne(self.x, perplexity=1, learning_rate=1000, n_iter=1000, library='opentsne')
        self.assertTrue(isinstance(pos, np.ndarray))
        self.assertTupleEqual(pos.shape, (4, 2))

    def test_convergence_animation(self):
        clip = self.swift.convergence_animation(self.x, n_iter=1000, fps=1, filename=None, perplexity=1)
        self.assertTrue(isinstance(clip, mpy.VideoClip))
        with self.assertRaises(TypeError):
            self.swift.convergence_animation(self.x, n_iter=1000, fps=1, filename=None, pp=1)
            self.swift.convergence_animation(self.x, n_iter=1000, fps=1, perplexity=1, library='opentsne')

    def test_tsne_animation(self):
        clip = self.swift.tsne_animation(self.x, fps=2, filename=None, iterable='learning_rate',
                                         perplexity=1, learning_rate=np.arange(10, 1000, 75))
        self.assertTrue(isinstance(clip, mpy.VideoClip))
        with self.assertRaises(KeyError):
            self.swift.tsne_animation(self.x, n_iter=1000, fps=1, filename=None, pp=1)
        with self.assertRaises(ValueError):
            self.swift.tsne_animation(self.x, n_iter=1000, fps=1, perplexity=1, library='opentsne')
        self.swift.tsne_animation(self.x, n_iter=1000, fps=1, perplexity=[1, 2, 3], library='opentsne')
        self.assertTrue(isinstance(clip, mpy.VideoClip))


if __name__ == '__main__':
    unittest.main()

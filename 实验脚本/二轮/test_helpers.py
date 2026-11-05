import unittest
import numpy as np
try:
    from helpers import fit_scale, blend_scores, rank_metrics
except ImportError:
    from helpers import fit_scale, blend_scores
    rank_metrics = None


class ScoreTests(unittest.TestCase):
    def test_blend_uses_validation_scale_for_other_candidates(self):
        self.assertIsNotNone(fit_scale, 'validation-fitted blend is not implemented')
        a = np.array([[0., 2.]])
        b = np.array([[2., 6.]])
        scales = (fit_scale(a), fit_scale(b))
        got = blend_scores(np.array([[4., 0.]]), np.array([[10., 2.]]), 0.25, scales)
        np.testing.assert_allclose(got, [[3., -1.]])

    def test_constant_scores_remain_finite(self):
        self.assertIsNotNone(fit_scale, 'validation-fitted blend is not implemented')
        a = np.ones((1, 3))
        scale = fit_scale(a)
        self.assertTrue(np.isfinite(blend_scores(a, a, .5, (scale, scale))).all())

    def test_full_rank_excludes_history_but_keeps_targets(self):
        self.assertIsNotNone(rank_metrics, 'full-catalog validation is not implemented')
        scores = np.array([.9, .8, .7, .6])
        known = np.array([True, True, True, False])
        ap, ndcg = rank_metrics(scores, [1, 3], known, k=2)
        self.assertAlmostEqual(ap, 1.)
        self.assertAlmostEqual(ndcg, 1.)


if __name__ == '__main__':
    unittest.main()

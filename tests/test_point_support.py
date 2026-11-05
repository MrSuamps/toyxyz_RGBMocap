"""Pointe single support: confidence dropouts of a lone toe-box support are closed and the stance keeps one anchor."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mocap.correction import prepare_contacts  # noqa: E402

N = 90


def clip():
    """Left foot planted throughout, drifting 0.4 cm/frame in X; right foot lifts at frame 20. Left confidence drops 40-47."""
    probes = np.zeros((N, 4, 3), np.float32)
    x = .004 * np.arange(N)
    probes[:, 0, 0], probes[:, 1, 0] = x, x
    probes[:, 1, 2] = .15
    probes[:, 2, 0], probes[:, 3, 0], probes[:, 3, 2] = .3, .3, .15
    lift = np.clip(np.arange(N) - 20, 0, None) * .03
    probes[:, 2:, 1] += lift[:, None]
    probabilities = np.full((N, 4), .9, np.float32)
    probabilities[20:, 2:] = 0
    probabilities[40:48, :2] = .1
    return probes, probabilities


class PointSupportTest(unittest.TestCase):
    def left(self, plan):
        return [(s['start'], s['end']) for s in plan[5] if s['foot'] == 0]

    def test_without_hold_the_dropout_splits_the_stance(self):
        probes, probabilities = clip()
        # Also split every 12 cm of drift (anchor_drift).
        self.assertEqual(self.left(prepare_contacts(probes, probabilities, 30.)), [(0, 31), (31, 40), (48, 78), (78, 90)])

    def test_lone_pointe_support_is_one_stance_pinned_where_it_stands_alone(self):
        probes, probabilities = clip()
        hold = np.zeros((N, 2), bool)
        hold[:, 0] = True
        active, anchors, support, *_ = prepare_contacts(probes, probabilities, 30., hold=hold)
        plan = prepare_contacts(probes, probabilities, 30., hold=hold)
        self.assertEqual(self.left(plan), [(0, 90)])
        self.assertTrue(active[:, 0].all())
        self.assertTrue(np.allclose(anchors[:, 0, [0, 2]], anchors[0, 0, [0, 2]]))
        first_alone = int(np.flatnonzero(~active[:, 1])[0])
        self.assertTrue(np.allclose(anchors[first_alone, 0, [0, 2]], support[first_alone, 0, [0, 2]], atol=1e-6))

    def test_a_raised_toe_box_keeps_its_dropout(self):
        probes, probabilities = clip()
        probes[40:48, :2, 1] += .05
        hold = np.ones((N, 2), bool)
        self.assertEqual(self.left(prepare_contacts(probes, probabilities, 30., hold=hold))[:1], [(0, 40)])


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from mocap.cli.anatomy import elbow_twist, swing_twist, unit

AXIS = np.array([0.0, -1.0, 0.0])
SHOULDER, ELBOW, WRIST = 0, 1, 2


def rest(off_plane_deg=20.0):
    """Upper arm along +x, forearm tilted off the hinge plane (perpendicular to AXIS) like the SMPL rest arm."""
    angle = np.radians(off_plane_deg)
    return np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.3 + 0.25 * np.cos(angle), 0.25 * np.sin(angle), 0.0]])


def turn(axis, degrees, count=1):
    return np.repeat(Rotation.from_rotvec(np.radians(degrees) * unit(axis)).as_matrix()[None], count, 0)


class ElbowTwistTest(unittest.TestCase):
    def test_pure_hinge_bend_reads_no_twist(self):
        points = rest()
        for bend in (30.0, 90.0, 150.0):
            local = turn(AXIS, bend)
            np.testing.assert_allclose(elbow_twist(local, points, SHOULDER, ELBOW, WRIST, AXIS), 0.0, atol=1e-6)
        forearm = unit(points[WRIST] - points[ELBOW])
        self.assertGreater(abs(swing_twist(turn(AXIS, 150.0), forearm)[1][0]), 20.0)

    def test_twist_about_the_forearm_survives_a_bend(self):
        points = rest()
        forearm = unit(points[WRIST] - points[ELBOW])
        for twist in (-70.0, 40.0, 95.0):
            local = turn(AXIS, 120.0) @ turn(forearm, twist)
            np.testing.assert_allclose(elbow_twist(local, points, SHOULDER, ELBOW, WRIST, AXIS), twist, atol=1e-6)


if __name__ == '__main__':
    unittest.main()

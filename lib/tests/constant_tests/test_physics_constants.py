import unittest

from ...constants.physical_constants import (
    AIR_DENSITY_AT_SEA_LEVEL,
    GAS_CONSTANT,
    GRAVITATIONAL_ACCELERATION,
    SPEED_OF_LIGHT,
    STANDARD_ATMOSPHERE
)


class TestTimeConstants(unittest.TestCase):
    def test_time_constants(self):
        self.assertEqual(AIR_DENSITY_AT_SEA_LEVEL, 1.225)
        self.assertEqual(GAS_CONSTANT, 8.314459848)
        self.assertEqual(GRAVITATIONAL_ACCELERATION, 9.80665)
        self.assertEqual(SPEED_OF_LIGHT, 2.99792458e8)
        self.assertEqual(STANDARD_ATMOSPHERE, 1.01325)

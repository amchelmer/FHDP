import unittest

from ...constants.time_constants import (
    DAYS_PER_WEEK,
    DAYS_PER_YEAR,
    HOURS_PER_DAY,
    HOURS_PER_WEEK,
    MINUTES_PER_DAY,
    MINUTES_PER_HOUR,
    MONTHS_PER_YEAR,
    SECONDS_PER_DAY,
    SECONDS_PER_HOUR,
    SECONDS_PER_MINUTE,
    WEEKDAYS_PER_WEEK,
    WEEKEND_DAYS_PER_WEEK,
)


class TestTimeConstants(unittest.TestCase):
    def test_time_constants(self):
        self.assertEqual(DAYS_PER_WEEK, 7.)
        self.assertEqual(DAYS_PER_YEAR, 365.)
        self.assertEqual(HOURS_PER_DAY, 24.)
        self.assertEqual(HOURS_PER_WEEK, 168.)
        self.assertEqual(MINUTES_PER_DAY, 1440.)
        self.assertEqual(MINUTES_PER_HOUR, 60.)
        self.assertEqual(MONTHS_PER_YEAR, 12.)
        self.assertEqual(SECONDS_PER_DAY, 86400.)
        self.assertEqual(SECONDS_PER_HOUR, 3600.)
        self.assertEqual(SECONDS_PER_MINUTE, 60.)
        self.assertEqual(WEEKDAYS_PER_WEEK, 5.)
        self.assertEqual(WEEKEND_DAYS_PER_WEEK, 2.)

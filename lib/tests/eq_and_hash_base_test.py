import unittest


class EqAndHashBaseTest(unittest.TestCase):
    """
    Base test class for testing implementations of __hash__, __eq__ and __ne__.
    """

    def assert_eq_and_hash_implemented_correctly(self, instance_creator, different_instance_creator):
        self.assert_hash_implemented_correctly(instance_creator, different_instance_creator)
        self.assert_eq_implemented_correctly(instance_creator, different_instance_creator)

    def assert_hash_implemented_correctly(self, instance_creator, different_instance_creator):
        instance_1, instance_2 = instance_creator(), instance_creator()
        self.assert_instances_hash_equal(instance_1, instance_2)

        different_instance = different_instance_creator()
        self.assert_instances_hash_not_equal(instance_1, different_instance)

    def assert_eq_implemented_correctly(self, instance_creator, different_instance_creator):
        instance_1 = instance_creator()
        self.assert_instance_equal_to_self(instance_1)

        instance_2 = instance_creator()
        self.assert_instances_equal(instance_1, instance_2)

        different_instance = different_instance_creator()
        self.assert_instances_not_equal(instance_1, different_instance)

    def assert_instance_equal_to_self(self, instance):
        self.assert_instances_equal(instance, instance)

    def assert_instances_hash_equal(self, instance_1, instance_2):
        self.assertEqual(hash(instance_1), hash(instance_2))

    def assert_instances_hash_not_equal(self, instance_1, different_instance):
        self.assertNotEqual(hash(instance_1), hash(different_instance))

    def assert_instances_equal(self, instance_1, instance_2):
        self.assertEqual(instance_1, instance_2)
        self.assertEqual(instance_2, instance_1)

        self.assertTrue(instance_1 == instance_2)
        self.assertTrue(instance_2 == instance_1)

        self.assertFalse(instance_1 != instance_2)
        self.assertFalse(instance_2 != instance_1)

    def assert_instances_not_equal(self, instance_1, different_instance):
        self.assertNotEqual(instance_1, different_instance)
        self.assertNotEqual(different_instance, instance_1)

        self.assertFalse(instance_1 == different_instance)
        self.assertFalse(different_instance == instance_1)

        self.assertTrue(instance_1 != different_instance)
        self.assertTrue(different_instance != instance_1)

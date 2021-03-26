"""
Utilities for testing the model as a whole.
"""


from unittest import TestCase


class TestModel(TestCase):

    def setUpClass(self):
        """
        Avoid redundant, time-consuming, equivalent setups that instantiate
        the model when testing it across the different methods, that can use
        a common instance.
        """
        pass
"""
.
"""


from unittest import TestCase


class TestLayers(TestCase):
    """
    Test each single layer kind independently.
    """
    pass


class TestModel(TestCase):
    """
    Test the model as a whole.
    """

    def setUpClass(self):
        """
        Avoid redundant, time-consuming, equivalent setups that instantiate
        the model when testing it across the different methods, that can use
        a common instance.
        """
        pass

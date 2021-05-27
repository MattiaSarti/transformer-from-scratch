"""
Reproducible tests to inherit from.
"""


from transformer.training_and_inference.reproducibility import (
    make_results_reproducible
)


class ReproducibleTestLayer:  # pylint: disable=too-few-public-methods
    """
    Common setups for reproducible tests.
    """

    def setUp(self):  # pylint: disable=no-self-use,invalid-name
        """
        Setup executed before every method (test) for reproducible results.
        """
        make_results_reproducible()

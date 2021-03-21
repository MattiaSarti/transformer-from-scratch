from torch.optim import Adam


class OptimizerHandler:
    """
    Hangling an optimizer with the defined learning rate schedule.
    """
    def __init__(self, optimizer: Adam, n_warmup_steps: int,
                 amplification_factor: float, model_hidden_dimension:
                 int) -> None:
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.amplification_factor = amplification_factor
        self.model_hidden_dimension = model_hidden_dimension
        self._training_step_number = 0
        self._learning_rate = 0

    def get_learning_rate(self, step: int = None) -> float:
        """
        Return the learning rate at the input step according to the Noam
        learning rate trend.
        """
        if step is not None:
            step = self._training_step_number
        # reconstructing Noam trend as (amplified, by a scalar) minimum
        # between a line passing through origin with positive slope - initial
        #  trend kept until warm-up is over - and a decreasing hyperbola -
        # trend maintained since warm-up is over to the end of training -
        # tending to 0 after ∞ steps - see my notes for demonstration that
        # their intersection corresponds to a step number equal to
        # n_warmup_steps:
        return self.amplification_factor * (
            (self.model_hidden_dimension ** (-0.5)) *
            min(
                (step ** (-0.5)),  # decreasing hyperbola
                (step * (self.n_warmup_steps ** (-1.5)))  # warm-up line
            )
        )

    def run_weight_update_step(self) -> None:
        """
        Update model parameters, in addition to learning rate value and
        current training step counter.
        """
        self._training_step_number += 1
        self._learning_rate = self.get_learning_rate(step=self.
                                                     _training_step_number)
        # updating the learning rate for all the parameters the optimizer
        # is assigned to:
        for parameters in self.optimizer.param_groups:
            parameters['lr'] = self._learning_rate
        # updating the trainable model parameters:
        self.optimizer.step()
        # cleaning gradients of trainable weights, for the next iteration:
        self.optimizer.zero_grad()

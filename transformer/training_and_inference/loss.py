"""
Loss functions and utilities for handling them.
"""


from torch import nonzero, Tensor  # noqa: E501 pylint: disable=E0611
from torch.nn import KLDivLoss, Module


class LabelSmoothedLoss(Module):
    """
    Layer to be stacked on top of the model of interest during training,
    carrying out label smoothing first and then loss computation, turning
    each target token id into smoothed probability distributions before
    feeding them, toghether with the model output, to the KL-divergence
    loss computation instead of one-hot target probability distributions.
    Inputs predictions and lables are assumed as flattened along samples
    (sequences), thus labels have shape: (# outputs, vocabulary size),
    fusing together outputs from both the same and different sequences,
    without considering positions. Label smoothing penalizes the model
    when outputting too "confident" predictions, when predicting a too high
    probability on the most likely token: the higher this probability over
    a reasonable value (for which the loss reaches its minimum), the higher
    the loss becomes, even if less gently than if this probability were lower
    than the value yielding the loss minimum. Label smoothing aims at avoiding
    overfitting, it is a form of regularization.
    """

    def __init__(self, softmax_dimension: int, padding_token: int,
                 smoothing_factor: float) -> None:
        super(LabelSmoothedLoss, self).__init__()
        self.softmax_dimension = softmax_dimension
        self.padding_token = padding_token
        # factor of which:
        self.smoothing_factor = smoothing_factor
        # factor of which:
        self.confidence = 1.0 - smoothing_factor
        # fraction of redistributed probability assigned to each one of the
        # non-target tokens in the vocabulary (padding token excluded)
        self.redistributed_probability_each = smoothing_factor /\
            (softmax_dimension - 2)

        # loss criterion - requiring inputs as log-probabilities:
        self.loss_criterion = KLDivLoss(reduction='sum', log_target=False)
        # predictions expected as log-probabilities, labels as probabilities

        # initialization of label distributions:
        self.smoothed_tgt_distributions = None
        # TODO: understand if just to inspect label distributions - not used

    def forward(self, predicted_log_probabilities: Tensor, tgt_tokens:
                Tensor) -> Tensor:
        """
        Forward propagation.
        """
        # NOTE: assertions are avoided to speed up training:
        # assert x.size(1) == self.softmax_dimension

        # creating a tensor with the same shape as the input one but filled
        # with constant values, evenly distributed over all vocabulary tokens,
        # equal to the smoothing factor divided by the number of tokens in the
        # vocabulaty except the padding token and the target token, which does
        # not have to receive a re-distribution of the original target one-hot
        # unitary probability distribution:
        smoothed_tgt_distributions = predicted_log_probabilities.detach()\
            .data.clone()
        # NOTE: necessary to make the smoothed label distributions not require
        # backpropagation (i.e. gradient computations and weight update)
        smoothed_tgt_distributions.fill_(self.redistributed_probability_each)

        # NOTE: avoided the original, redundant division, it can be done just
        # once during initialization to speed up training:
        # smoothed_tgt_distributions.fill_(self.smoothing_factor /
        #                                 (self.softmax_dimension - 2))

        # filling the probability values of the target tokens in the smoothed
        # distributions with the remaining probability portion of the
        # originally unitary, target one after its partial redistribution over
        # the other tokens of the vocabulary:
        smoothed_tgt_distributions.scatter_(
            dim=1,
            index=tgt_tokens.data.unsqueeze(dim=1),  # 1D -> 2D
            value=self.confidence
        )

        # resetting the padding token probability to 0, as the smoothed
        # probability amount is redistributed only on meaningful tokens and
        # not on the padding one when smoothing labels:
        smoothed_tgt_distributions[:, self.padding_token] = 0

        # for outputs whose target corresponds to the padding token:
        mask_indices = nonzero(tgt_tokens.data == self.padding_token)
        if mask_indices.dim() > 0:  # TODO: '.dim()' is always 2
            # not considering predictions over samples where the target token
            # is the padding one by setting all the probability values over
            # the vocabulary of those predictions to 0:
            smoothed_tgt_distributions.index_fill_(
                dim=0,
                index=mask_indices.squeeze(),  # TODO: '.squeeze()' redundant?
                value=0.0
            )

        # TODO: understand if just to inspect label distributions - not used
        self.smoothed_tgt_distributions = smoothed_tgt_distributions

        # returning loss value by considering the smoothed label distributions
        # instead of the one-hot labels as targets:
        return self.loss_criterion(
            input=predicted_log_probabilities,
            target=smoothed_tgt_distributions
        )

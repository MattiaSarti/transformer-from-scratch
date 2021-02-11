from time import time
from typing import Callable

from torch import nn, nonzero, Tensor
from torch.optim import Adam

from model.attention import allowed_positions_to_attend


def data_iterator() -> Callable:
    """
    """
    pass


def execute_training_epoch(data_iterator: Callable, model: nn.Module,
                           loss_computer: None) -> float:
    # TODO: update data type of loss computer
    """
    Execute a single epoch of model training.
    """
    tic = time()
    cumulative_loss = 0
    cumulative_n_tokens_done = 0
    # for each mini-batch:
    for i, mini_batch in enumerate(data_iterator):
        # forward propagation:
        output = model(
            mini_batch.src_tokens,
            mini_batch.tgt_input_tokens,
            mini_batch.src_mask, mini_batch.tgt_mask
            # TODO: understand why it uses model.forward, the docs suggest
            # using __call__ instead - AND - specify args
        )
        loss = loss_computer(
            output=output,
            tgt_expected_tokens=mini_batch.tgt_expected_tokens,
            xxx=mini_batch.actual_n_target_tokens
        )
        cumulative_loss += loss
        cumulative_n_tokens_done += mini_batch.actual_n_target_tokens
        # displaying loss every 50 mini-batches:
        if i % 50 == 1:
            toc = time()
            print("Mini-batches done: {i} - Loss for the current mini-" +
                  "batch: {l} - Average time per mini-batch [s]: {t}".format(
                    i=i, l=round(loss, 4), t=round(((toc-tic) / 50), 1)))
            tic = time()
    # returning the average loss across all the tokens of all the
    # mini-batches:
    return cumulative_loss / cumulative_n_tokens_done


class LabelSmoothedLoss(nn.Module):
    """
    Layer to be stacked on top of the model of interest during training,
    carrying out label smoothing first and then loss computation, turning the
    one-hot target probability distributions of each position into smoothed
    probability distributions before feeding them, toghether with the model
    output, to the KL-divergence loss computation.
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
        self.loss_criterion = nn.KLDivLoss(reduction='sum')

        # initialization of label distributions:
        self.smoothed_tgt_distributions = None
        # TODO: understand if just to inspect label distributions - not used

    def forward(self, x: Tensor, tgt_tokens: Tensor) -> Tensor:

        # NOTE: assertions are avoided to speed up training:
        # assert x.size(1) == self.softmax_dimension

        # creating a tensor with the same shape as the input one but filled
        # with constant values, evenly distributed over all vocabulary tokens,
        # equal to the smoothing factor divided by the number of tokens in the
        # vocabulaty except the padding token and the target token, which does
        # not have to receive a re-distribution of the original target one-hot
        # unitary probability distribution:
        smoothed_tgt_distributions = x.clone()
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
            index=tgt_tokens.unsqueeze(dim=1),
            value=self.confidence
        )

        # resetting the padding token probability to 0, as the smoothed
        # probability amount is redistributed only on meaningful tokens and
        # not on the padding one when smoothing labels:
        smoothed_tgt_distributions[:, self.padding_token] = 0

        # for outputs whose target corresponds to the padding token:
        mask_indices = nonzero(tgt_tokens == self.padding_token)
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

        # TODO: check if necessary:
        smoothed_tgt_distributions.requires_grad = False

        # returning loss value by considering the smoothed label distributions
        # instead of the one-hot labels as targets:
        return self.loss_criterion(
            input=x,
            target=smoothed_tgt_distributions
        )


class LossComputer:
    """
    Take care of loss computation, backpropagation and weight update during a
    single training iteration.
    """
    def __init__(self, final_softmax_layer: nn.Module, criterion,
                 # TODO: add 'criterion' data type
                 optimizer: Adam = None) -> None:
        self.final_softmax_layer = final_softmax_layer
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x: Tensor, y: Tensor, norm) -> float:
        # TODO: add 'norm' data type AND check returned data type

        # computing final softmax log-probabilities:
        x = self.final_softmax_layer(x)
        # computing loss value:
        loss = self.criterion(

        )
        # backpropagating gradient to trainable weights:
        loss.backward()
        if self.optimizer:
            # updating trainable weights:
            self.optimizer.step()
            # cleaning gradients of trainable weights, for next iteration:
            self.optimizer.zero_grad()
        # TODO
        return loss.item() * norm


class MiniBatch:
    """
    Mini-batch of samples.
    """
    @staticmethod
    def build_mask(tgt_tokens: Tensor, padding_token: int) -> Tensor:
        """
        Build masks of target positions allowed to be attended by the decoder,
        position by position:
        """
        tgt_mask = (tgt_tokens != padding_token).unsqueeze(dim=-2)
        tgt_mask = tgt_mask and \
            allowed_positions_to_attend(n_positions=tgt_tokens.size(-1))\
            .type_as(tgt_mask)
        # NOTE: the original implementation had '&', which is the bit-wise
        # AND, in place of 'and', which is the logical AND... why? wasn't it
        # wrong?
        return tgt_mask

    def __init__(self, src_tokens: Tensor, tgt_tokens: Tensor = None,
                 padding_token: int = 0) -> None:
        # source inputs:
        self.src_tokens = src_tokens
        # all source positions are allowed to be attended, both by the
        # encoder and by decoder:
        self.src_mask = (src_tokens != padding_token).unsqueeze(dim=-2)
        # when target outputs specified:
        if tgt_tokens:
            self.tgt_input_tokens = tgt_tokens[:, :-1]  # excluding </s> token
            self.tgt_expected_tokens = tgt_tokens[:, 1:]  # excluding <s> token
            self.actual_n_target_tokens = \
                (self.tgt_input_tokens != padding_token).sum()
            # only target positions up to the current position are allowed to
            # be attended by the decoder, for each position:
            self.tgt_mask = self.build_mask(tgt_tokens,
                                            padding_token=padding_token)


class MiniBatchHandler:
    """
    """
    def __init__(self, max_n_src_tokens_in_minibatch: int,
                 max_n_tgt_tokens_in_minibatch: int) -> None:
        self.max_n_src_tokens_in_minibatch = max_n_src_tokens_in_minibatch
        self.max_n_tgt_tokens_in_minibatch = max_n_tgt_tokens_in_minibatch

    def get_current_minibatch_size(self, new, count: int):
        # TODO: add data type & understand why they add an unused, additional
        # argument called 'sofar'

        # resetting initial values when starting a new mini-batch size
        # monitoring (during construction):
        if count == 1:
            self.max_n_src_tokens_in_minibatch = 0
            self.max_n_tgt_tokens_in_minibatch = 0
        # :
        self.max_n_src_tokens_in_minibatch = max(
            self.max_n_src_tokens_in_minibatch,
            len()
        )
        self.max_n_tgt_tokens_in_minibatch = max(
            self.max_n_tgt_tokens_in_minibatch,
            len()
        )
        #
        src_tokens = count * self.max_n_src_tokens_in_minibatch
        tgt_tokens = count * self.max_n_tgt_tokens_in_minibatch
        return max(src_tokens, tgt_tokens)


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

    def get_learning_rate(self, step=None):
        """
        Return the learning rate at the input step according to the Noam
        learning rate trend.
        """
        if not step:
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

    def run_training_step(self):
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

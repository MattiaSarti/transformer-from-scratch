"""
Utilities for training a Transformer.
"""


from time import time
from typing import Generator

from numpy import int64 as numpy_int64
from numpy.random import randint
from torch import from_numpy, nonzero, Tensor
from torch.nn import KLDivLoss, Module
from torch.optim import Adam

from transformer.architecture.attention import allowed_positions_to_attend


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
            .clone()
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
            index=tgt_tokens.unsqueeze(dim=1),  # 1D -> 2D
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

        # returning loss value by considering the smoothed label distributions
        # instead of the one-hot labels as targets:
        return self.loss_criterion(
            input=predicted_log_probabilities,
            target=smoothed_tgt_distributions
        )


# TODO: this should not be a class, it just stores data after processing them
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
        tgt_mask = tgt_mask & (
            allowed_positions_to_attend(
                n_positions=tgt_tokens.size(-1)
            ).type_as(tgt_mask)
        )
        # NOTE: the original implementation had '&', which is the bit-wise
        # AND, in place of 'and', which is the logical AND... why? wasn't it
        # wrong?
        return tgt_mask

    def __init__(self, src_tokens: Tensor, padding_token: int,
                 tgt_tokens: Tensor = None) -> None:
        # source inputs:
        self.src_tokens = src_tokens
        # all source positions are allowed to be attended, both by the
        # encoder and by decoder:
        self.src_mask = (src_tokens != padding_token).unsqueeze(dim=-2)
        # when target outputs specified:
        if tgt_tokens is not None:
            self.tgt_input_tokens = tgt_tokens[:, :-1]  # excluding </s> token
            self.tgt_expected_tokens = tgt_tokens[:, 1:]  # excluding <s> token
            self.actual_n_target_tokens = \
                (self.tgt_expected_tokens != padding_token).sum().item()
            # only target positions up to the current position are allowed to
            # be attended by the decoder, for each position:
            self.tgt_mask = self.build_mask(self.tgt_input_tokens,
                                            padding_token=padding_token)
    # NOTE: understand why shapes of tgt masks are different from src masks


class MiniBatchHandler:
    """
    TODO
    """
    def __init__(self, max_n_src_tokens_in_mini_batch: int,
                 max_n_tgt_tokens_in_mini_batch: int) -> None:
        self.max_n_src_tokens_in_mini_batch = max_n_src_tokens_in_mini_batch
        self.max_n_tgt_tokens_in_mini_batch = max_n_tgt_tokens_in_mini_batch

    def get_current_mini_batch_size(self, new, count: int):
        """
        TODO
        """
        # TODO: add data type & understand why they add an unused, additional
        # argument called 'sofar'

        # resetting initial values when starting a new mini-batch size
        # monitoring (during construction):
        if count == 1:
            self.max_n_src_tokens_in_mini_batch = 0
            self.max_n_tgt_tokens_in_mini_batch = 0
        # :
        self.max_n_src_tokens_in_mini_batch = max(
            self.max_n_src_tokens_in_mini_batch,
            len()
        )
        self.max_n_tgt_tokens_in_mini_batch = max(
            self.max_n_tgt_tokens_in_mini_batch,
            len()
        )
        # TODO:
        src_tokens = count * self.max_n_src_tokens_in_mini_batch
        tgt_tokens = count * self.max_n_tgt_tokens_in_mini_batch
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

    def get_learning_rate(self, step: int = None):
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

    def run_weight_update_step(self):
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


class LossMinimizer:
    """
    Take care of loss computation, backpropagation and weight update during a
    single training iteration (for single mini-batch in a given epoch).
    """
    def __init__(self, final_log_softmax_layer: Module, criterion: Module,
                 optimizer_handler: OptimizerHandler = None) -> None:
        self.final_log_softmax_layer = final_log_softmax_layer
        self.criterion = criterion
        self.optimizer_handler = optimizer_handler

    def __call__(self, log_probabilities: Tensor, labels: Tensor,
                 n_mini_batch_tokens: int) -> float:
        """
        Compute loss from log-probabilities, and, if an optimizer handler is
        # passed, backpropagate gradient and update weights to minimize loss.
        """
        # TODO: check returned data type - float or Tensor?

        # computing final softmax log-probabilities:
        log_probabilities = self.final_log_softmax_layer(log_probabilities)
        # computing loss value, flattening all outputs of all sequences along
        # a unique dimension (with no changes on the computational graphs but
        # more efficiently for the subsequent computations, using 2D arrays),
        # i.e. sequences are flattened:
        loss = self.criterion(
            predicted_log_probabilities=log_probabilities.contiguous()\
                .view(
                    (-1, log_probabilities.size(-1))
                ),  # 3D -> 2D
            tgt_tokens=labels.contiguous().view(-1)  # 2D -> 1D
        ) / n_mini_batch_tokens
        # this normalization of the loss, which is returned by the criterion
        # as the sum of all the single loss values, by the number of tokens
        # in the mini-batch, which is analogous to the number of mini-batches
        # with sequences flattened and fused together, is carried out only to
        # backpropagate its gradient and to update weights more efficiently,
        # and it is reverted afterwards to return the original loss value

        # updating weights only when an optimizer is passed:
        if self.optimizer_handler is not None:

            # backpropagating gradient to trainable weights:
            loss.backward()
            # NOTE: this step is included in this 'if' statement
            # to execute it, i.e. to compute gradients, only when updating
            # weights, contrary to the original implementation
            # updating trainable weights:

            self.optimizer_handler.run_weight_update_step()
            # NOTE: gradients of trainable weights are cleaned, for the next
            # iteration, implicitly within this method call of the optimizer
            # handler and not explicitly here, contrary to the original
            # implementation

        # returning the loss value reverted back to the original,
        # un-normalized scale:
        return loss.item() * n_mini_batch_tokens


def copy_task_dataset_builder(
        vocabulary_size: int, mini_batch_size: int,
        n_mini_batches: int) -> Generator[MiniBatch, None, None]:
    """
    Build generator yielding dummy samples and labels for a toy source-target
    copy task.
    """
    sequence_length = 10  # same for all sequences, here

    for _ in range(n_mini_batches):
        # random token indices, excluding 0 because assumed to represent the
        # padding token:
        samples = from_numpy(
            randint(
                low=1,
                high=vocabulary_size,
                size=(mini_batch_size, sequence_length),
                dtype=numpy_int64
            )
        )
        # assuming all sequences start with the same token, an hypothetical
        # <s> token that can also be found in other positions of the sequences
        # in this toy task:
        samples[:, 0] = 1
        # yielding mini-batch made of identical source and target samples
        # (i.e. labels equal samples):
        yield MiniBatch(
            src_tokens=samples.detach().clone(),  # graph-detached, deep copy
            tgt_tokens=samples.detach().clone(),  # graph-detached, deep copy
            padding_token=0  # as assumed above
        )


def execute_training_epoch(
        dataset_iterator: Generator[MiniBatch, None, None], model: Module,
        loss_minimizer: LossMinimizer, verbose: bool = True) -> float:
    """
    Execute a single epoch of model training.
    """
    cumulative_loss = 0
    cumulative_n_tokens_done = 0
    tic = time()

    # for each mini-batch:
    for i, mini_batch in enumerate(dataset_iterator):

        # forward propagation:
        output = model.forward(
            # TODO: understand why it uses "model.forward()", despite the docs
            # suggest using "model.__call__()" by directly calling "model()"
            # instead
            src_tokens=mini_batch.src_tokens,
            tgt_tokens=mini_batch.tgt_input_tokens,
            src_mask=mini_batch.src_mask,
            tgt_mask=mini_batch.tgt_mask
        )

        # loss computation, backpropagation and weight update:
        loss = loss_minimizer(
            x=output,
            y=mini_batch.tgt_expected_tokens,
            n_mini_batch_tokens=mini_batch.actual_n_target_tokens
        )

        # remembering statistics:
        cumulative_loss += loss
        cumulative_n_tokens_done += mini_batch.actual_n_target_tokens

        if verbose:

            # displaying loss every 50 mini-batches:
            display_every = 50
            if i % display_every == 0:
                toc = time()
                print(("Mini-batches done: {n} - Loss for the current mini-" +
                       "batch: {l:.4f} - Average speed [tokens/s]: {t:.1f}")
                      .format(
                          n=(i + 1),
                          l=(loss / mini_batch.actual_n_target_tokens),
                          t=(cumulative_n_tokens_done / (toc-tic))
                      ))

    # returning the average loss across all the tokens of all the
    # mini-batches:
    return cumulative_loss / cumulative_n_tokens_done

from time import time
from typing import Callable

from .attention import allowed_positions_to_attend
from .base import *


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
            allowed_positions_to_attend(n_positions=tgt.size(-1))\
                .type_as(tgt_mask.data)
        # NOTE: the original implementation had '&', which is the bit-wise AND, in place of 'and', which is the logical AND... why? wasn't it wrong?
        return tgt_mask
    
    def __init__(self, src_tokens: Tensor, tgt_tokens: Tensor=None,
        padding_token: int=0) -> None:
        # source inputs:
        self.src_tokens = src_tokens
        # all source positions are allowed to be attended, both by the 
        # encoder and by decoder:
        self.src_mask = (src_tokens != padding_token).unsqueeze(dim=-2)
        # when target outputs specified:
        if tgt_tokens:
            self.tgt_input_tokens = tgt_tokens[:, :-1] # excluding </s> token
            self.tgt_expected_tokens = tgt_tokens[:, 1:] # excluding <s> token
            self.actual_n_target_tokens = \
                (self.tgt_input_tokens != pagging_token).data.sum()
            # only target positions up to the current position are allowed to
            # be attended by the decoder, for each position:
            self.tgt_mask = self.build_mask(tgt_tokens, \
                padding_token=padding_token)


class OptimizerHandler:
    """
    Hangling an optimizer with the defined learning rate schedule.
    """
    def __init__(self, optimizer: , warmup_factor: ,
        amplification_factor: , model_output_dimension: int): # TODO: add/specify data types
        self.optimizer = optimizer
        self.warmup_factor = warmup_factor
        self.amplification_factor = amplification_factor
        self.model_output_dimension = model_output_dimension
        self._training_step_number = 0
        self._learning_rate = 0

    def get_learning_rate(self, step=None):
        """
        Return the learning rate at the input step according to the Noam 
        learning rate trend.
        """
        if not step:
            step = self._training_step_number
        # TODO: reconstruct trend
        return self.amplification_factor * ((self.model_output_dimension ** (-0.5)) * min(step ** (-0.5), step * (self.warmup_factor ** (-15))))

    def run_training_step(self):
        """
        Update model parameters, in addition to learning rate value and 
        current training step counter.
        """
        self._training_step_number += 1
        self._learning_rate = get_learning_rate(step=\
            self._training_step_number)
        # updating the learning rate for all the parameters the optimizer 
        # is assigned to:
        for parameters in self.optimizer.param_groups:
            parameters['lr'] = self._learning_rate
        # updating the trainable model parameters:
        self.optimizer.step()


def data_iterator() -> Callable:
    pass


def execute_training_epoch(data_iterator: Callable, model: nn.Module, 
    loss_computer: ) -> float: # TODO: update data type of loss computer
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
            mini_batch.src_mask, mini_batch.tgt_mask # why it uses model.forward? the docs suggest using __call__ instead - AND - specify args
        )
        loss = loss_computer(
            output=output,
            tgt_expected_tokens=mini_batch.tgt_expected_tokens,
            mini_batch.actual_n_target_tokens
        )
        cumulative_loss += loss
        cumulative_n_tokens_done += mini_batch.actual_n_target_tokens
        # displaying loss every 50 mini-batches:
        if i % 50 == 1:
            toc = time()
            print("Mini-batches done: {i} - Loss for the current mini-" + \
                "batch: {l} - Average time per mini-batch [s]: {t}".format(\
                i=i, l=round(loss, 4), t=round(((toc-tic) / 50), 1)))
            tic = time()
    # returning the average loss across all the tokens of all the 
    # mini-batches:
    return cumulative_loss / cumulative_n_tokens_done
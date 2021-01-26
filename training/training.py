from time import time
from typing import Callable

from torch import nn, nonzero, tensor, Tensor
from torch.optim import Adam

from model.attention import allowed_positions_to_attend


def data_iterator() -> Callable:
    """
    """
    pass


def execute_training_epoch(data_iterator: Callable, model: nn.Module, 
    loss_computer: None) -> float: # TODO: update data type of loss computer
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
            xxx=mini_batch.actual_n_target_tokens
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


class LabelSmoothing(nn.Module):
    """
    Layer carrying out label smoothing, to be used on top of the model of 
    interest during training, before loss computation, to turn the one-hot
    target probability distributions of each position into smoothed 
    probability distributions.
    """
    def __init__(self, softmax_dimension: int, padding_token: int,
        smoothing_factor: float):
        super(LabelSmoothing, self).__init__()
        self.softmax_dimension = softmax_dimension
        self.padding_token = padding_token
        self.smoothing_factor = smoothing_factor
        self.confidence = 1 - smoothing_factor
        self.criterion = nn.KLDivLoss(size_average=False) # input as log-probabilities
        self.actual_label_distribution = None

    def forward(self, x: Tensor, target_tokens: Tensor):
        assert x.size(1) == self.softmax_dimension
        actual_label_distribution = target_tokens.data.clone() # TODO: understand why then copying data
        # TODO: ?
        actual_label_distribution.fill_(self.smoothing_factor / (self.softmax_dimension - 2))
        # TODO: ?
        actual_label_distribution.scatter(
            dim=1,
            index=target_tokens.data.unsqueeze(),
            src=self.confidence
        )
        # TODO: ?
        actual_label_distribution[:, self.padding_token] = 0
        # if some positions are padded (i.e. actual target sequence length 
        # lower than maximum sequence length):
        mask = nonzero(target_tokens == self.padding_token)
        if mask.dim() > 0:
            # not considering predictions aver padding tokens by setting 
            # those predictions to 0:
            actual_label_distribution.index_fill_(
                dim=0,
                index=mask.squeeze(),
                val=0.0
            )
        self.actual_label_distribution = actual_label_distribution
        return self.criterion( # TODO: write arg names
            x,
            tensor(actual_label_distribution, requires_grad=False)
        )
    
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
    def __init__(self, optimizer: Adam, n_warmup_steps: int, 
        amplification_factor: float, model_hidden_dimension: int):
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
            (self.model_hidden_dimension ** (-0.5)) * \
                min(
                    (step ** (-0.5)), # decreasing hyperbola
                    (step * (self.n_warmup_steps ** (-1.5))) # warm-up line
                )
        )

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


class MiniBatchHandler:
    """
    """
    def __init__(self, max_n_src_tokens_in_minibatch: int, max_n_tgt_tokens_in_minibatch: int):
        self.max_n_src_tokens_in_minibatch = max_n_src_tokens_in_minibatch
        self.max_n_tgt_tokens_in_minibatch = max_n_tgt_tokens_in_minibatch

    def get_current_minibatch_size(self, new, count: int): # TODO: add data type & understand why they add an unused, additional argument called 'sofar'
        # resetting initial values when starting a new mini-batch size 
        # monitoring (during construction):
        if count == 1:
            self.max_n_src_tokens_in_minibatch = 0
            self.max_n_tgt_tokens_in_minibatch = 0
        # 
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


class LossComputer:
    """
    Take care of loss computation, backpropagation and weight update during a
    single training iteration.
    """
    def __init__(self, final_softmax_layer: nn.Module, criterion: ,
        optimizer: Adam=None):
        self.final_softmax_layer = final_softmax_layer
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x: Tensor, y: Tensor, norm: )
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

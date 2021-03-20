"""
Utilities for training a Transformer.
"""


from time import time
from typing import Generator, List

from numpy import int64 as numpy_int64
from numpy.random import randint
from torch import cat as torch_cat, from_numpy, Tensor
from torch.nn import Module
from torch.nn.parallel import gather as parallel_gather, parallel_apply,\
    replicate as parallel_replicate, scatter as parallel_scatter

from transformer.architecture.attention import allowed_positions_to_attend
from transformer.training_and_inference.loss import LabelSmoothedLoss
from transformer.training_and_inference.optimizer import OptimizerHandler


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
                (self.tgt_expected_tokens != padding_token).data.sum().item()
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


class LossMinimizer:
    """
    Take care of loss computation, backpropagation and weight update during a
    single training iteration (for single mini-batch in a given epoch).
    """
    def __init__(self, final_log_softmax_layer: Module, criterion: Module,
                 optimizer_handler: OptimizerHandler = None) -> float:
        self.final_log_softmax_layer = final_log_softmax_layer
        self.criterion = criterion
        self.optimizer_handler = optimizer_handler

    def __call__(self, logits: Tensor, labels: Tensor, n_mini_batch_tokens:
                 int) -> float:
        """
        Compute loss from log-probabilities, and, if an optimizer handler is
        passed, backpropagate gradient and update weights to minimize loss.
        """
        # TODO: check returned data type - float or Tensor?

        # computing final softmax log-probabilities:
        log_probabilities = self.final_log_softmax_layer(logits)
        # computing loss value, flattening all outputs of all sequences along
        # a unique dimension (with no changes on the computational graphs but
        # more efficiently for the subsequent computations, using 2D arrays),
        # i.e. sequences are flattened:
        loss = self.criterion(
            predicted_log_probabilities=log_probabilities.contiguous().view(
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

            # backpropagating loss gradient to trainable weights:
            loss.backward()
            # NOTE: this step is included in this 'if' statement
            # to execute it, i.e. to compute gradients, only when updating
            # weights, contrary to the original implementation
            # updating trainable weights:

            # updating weights based on their respective gradients, eventually
            # cleaning such gradients for next iterations:
            self.optimizer_handler.run_weight_update_step()
            # NOTE: gradients of trainable weights are cleaned, for the next
            # iteration, implicitly within this method call of the optimizer
            # handler and not explicitly here, contrary to the original
            # implementation

        # returning the loss value reverted back to the original,
        # un-normalized scale:
        return loss.item() * n_mini_batch_tokens


class DataParallelLossMinimizer:
    """
    Take care of loss computation, backpropagation and weight update during a
    single training iteration (for single mini-batch in a given epoch) among
    different GPUS according to a data-parallel strategy.
    """
    def __init__(self, final_log_softmax_layer: Module, criterion: Module,
                 device_ids: List[int], chunk_size: int = 5, 
                 optimizer_handler: OptimizerHandler = None) -> None:
        self.final_log_softmax_layer = final_log_softmax_layer
        self.criterion = parallel_replicate(criterion, devices=device_ids)
        self.optimizer_handler = optimizer_handler
        self.device_ids = device_ids
        self.chunk_size = chunk_size

    def __call__(self, logits: Tensor, labels: Tensor, n_mini_batch_tokens:
                 int) -> float:
        """
        Compute loss from log-probabilities, and, if an optimizer handler is
        passed, backpropagate gradient and update weights to minimize loss.
        """

        # TODO: understand
        total = 0.0

        # replicating the last log-softmax layer on all the devices - note
        # that it is done here instead of during initialization (contrary to
        # the optimizer):
        # TODO: understand why it is not done during __init__ contrary to the
        # optimizer
        final_log_softmax_layer = parallel_replicate(
            self.final_log_softmax_layer,
            device=self.device_ids
        )

        # distributing the tensors of predicted log-probabilities and labels
        # among all the devices by splitting them along the first dimenstion,
        # that is the mini-batch size, so as to distribute the mini-batch
        # samples among parallel sub-mini batches, each one on a different
        # device:
        scattered_logits = parallel_scatter(
            inputs=logits,
            target_gpus=self.device_ids
        )
        scattered_labels = parallel_scatter(
            inputs=labels,
            target_gpus=self.device_ids
        )

        # initializing the different gradients with respect to the samples in
        # the respective sub-mini-batches, on the respective devices:
        gradients_from_devices = [[] for _ in scattered_logits]

        ###########################################################################################################

        for i in range(0, scattered_logits[0].dim(1), self.chunk_size):

            ... = [[tensor(l[:, i:i+self.chunk_size],
                           requires_grad=self.optimizer_handler is not None)]
                   for l in scattered_logits]

            scattered_log_probabilities = parallel_apply(
                self.final_log_softmax_layer,
                ...
            )

        ###########################################################################################################

            # computing gradients, for updating weights later, only when an
            # optimizer is passed:
            if self.optimizer_handler is not None:

                # backpropagating loss gradient to output log-probabilites - Chain
                # Rule is exploited to later backpropagate complete gradients up
                # to each weight, but gradients across different sub-mini-batches
                # on different devices need to be cumulated before:
                loss.backward()

        # computing gradients and updating weights only when an optimizer is
        # passed:
        if self.optimizer_handler is not None:

            # gathering all gradients from sub-mini-batches on different
            # devices and concatenating them along the first dimenstion, that
            # is the mini-batch size - now these are loss gradients with
            # respect to output log-probabilities - and completing Chain Rule,
            # backpropagating from these complete gradients to weights -
            # equivalently to backpropagating gradient with respect to weights
            # of the loss averaged on all samples in the whole mini-batch,
            # i.e. on all sub-mini-batches among different devices:
            gradients_from_devices = [
                # concatenating gradients along the 1st dimension TODO: i.e.?
                tensor(torch_cat(gradients_from_devices, dim=1),
                                 requires_grad=True)
                for g in gradients_from_devices
            ]
            log_probabilities.backward(
                gradient=paraller_gather(
                    outputs=gradients_from_devices,
                    target_device=self.device_ids[0]  # device used to compute
                )
            )

            # updating weights based on their respective gradients, eventually
            # cleaning such gradients for next iterations:
            self.optimizer_handler.run_weight_update_step()
            # NOTE: gradients of trainable weights are cleaned, for the next
            # iteration, implicitly within this method call of the optimizer
            # handler and not explicitly here, contrary to the original
            # implementation

        # returning the loss value reverted back to the original,
        # un-normalized scale:
        return loss.item() * n_mini_batch_tokens

        ###########################################################################################################
        
    # def __call__(self, out, targets, normalize):
        # total = 0.0
        # generator = nn.parallel.replicate(self.generator, 
        #                                         devices=self.devices)
        # out_scatter = nn.parallel.scatter(out, 
        #                                   target_gpus=self.devices)
        # out_grad = [[] for _ in out_scatter]
        # targets = nn.parallel.scatter(targets, 
        #                               target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i+chunk_size].data, 
                                    requires_grad=self.opt is not None)] 
                           for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss. 
            y = [(g.contiguous().view(-1, g.size(-1)), 
                  t[:, i:i+chunk_size].contiguous().view(-1)) 
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l = nn.parallel.gather(loss, 
                                   target_device=self.devices[0])
            l = l.sum()[0] / normalize
            total += l.data[0]

            # # Backprop loss to output of transformer
            # if self.opt is not None:
            #     l.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # # Backprop all loss through transformer.            
        # if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad, 
                                    target_device=self.devices[0])
            # o1.backward(gradient=o2)
            # self.opt.step()
            # self.opt.optimizer.zero_grad()
        # return total * normalize


def copy_task_dataset_builder(vocabulary_size: int, mini_batch_size: int,
                              n_mini_batches: int) -> Generator[MiniBatch,
                                                                None, None]:
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


def execute_training_epoch(dataset_iterator: Generator[MiniBatch, None, None],
                           model: Module, loss_minimizer: LossMinimizer,
                           verbose: bool = True) -> float:
    """
    Execute a single epoch of model training.
    """
    cumulative_loss = 0
    cumulative_n_tokens_done = 0
    tic = time()

    # for each mini-batch:
    for i, mini_batch in enumerate(dataset_iterator):

        # forward propagation:
        output_logits = model.forward(
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
            x=output_logits,
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

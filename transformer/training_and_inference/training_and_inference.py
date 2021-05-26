"""
Utilities for training a Transformer.
"""


from time import time
# from typing import Generator, List, Optional
from typing import Generator, Optional

# from torch import cat as torch_cat, Tensor
from torch import Tensor
from torch.nn import Module
# from torch.nn.parallel import gather as parallel_gather, parallel_apply,\
#     replicate as parallel_replicate, scatter as parallel_scatter

from transformer.training_and_inference.data import MiniBatch
from transformer.training_and_inference.optimizer import OptimizerHandler


class LossMinimizer:  # pylint: disable=too-few-public-methods
    """
    Take care of loss computation, backpropagation and weight update during a
    single training iteration (for single mini-batch in a given epoch).
    """

    def __init__(self, log_softmax_layer: Module, criterion: Module,
                 optimizer_handler: Optional[OptimizerHandler] = None)\
            -> float:
        self.final_log_softmax_layer = log_softmax_layer
        self.criterion = criterion
        self.optimizer_handler = optimizer_handler

    def __call__(self, logits: Tensor, labels: Tensor, n_mini_batch_tokens:
                 int) -> float:
        # TODO: check returned data type - float or Tensor?
        """
        Compute loss from log-probabilities, and, if an optimizer handler is
        passed, backpropagate gradient and update weights to minimize loss.
        """
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


# class DataParallelLossMinimizer:
#     """
#     Take care of loss computation, backpropagation and weight update during
#     a single training iteration (for single mini-batch in a given epoch)
#     among different GPUS according to a data-parallel strategy.
#     """

#     def __init__(self, log_softmax_layer: Module, criterion: Module,
#                  device_ids: List[int], chunk_size: int = 5,
#                  optimizer_handler: Optional[OptimizerHandler] = None)\
#             -> None:
#         self.final_log_softmax_layer = log_softmax_layer
#         self.criterion = parallel_replicate(criterion, devices=device_ids)
#         self.optimizer_handler = optimizer_handler
#         self.device_ids = device_ids
#         self.chunk_size = chunk_size

#     def __call__(self, logits: Tensor, labels: Tensor, n_mini_batch_tokens:
#                  int) -> float:
#         """
#         Compute loss from log-probabilities, and, if an optimizer handler is
#         passed, backpropagate gradient and update weights to minimize loss.
#         """
#         # TODO: understand
#         total = 0.0

#         # replicating the last log-softmax layer on all the devices - note
#         # that it is done here instead of during initialization (contrary to
#         # the optimizer):
#         # TODO: understand why it is not done during __init__ contrary to
#         # the optimizer
#         final_log_softmax_layer = parallel_replicate(
#             network=self.final_log_softmax_layer,
#             devices=self.device_ids
#         )

#         # distributing the tensors of predicted log-probabilities and labels
#         # among all the devices by splitting them along the first
#         # dimenstion, that is the mini-batch size, so as to distribute the
#         # mini-batch samples among parallel sub-mini batches, each one on a
#         # different device:
#         scattered_logits = parallel_scatter(
#             inputs=logits,
#             target_gpus=self.device_ids
#         )
#         scattered_labels = parallel_scatter(
#             inputs=labels,
#             target_gpus=self.device_ids
#         )

#         # initializing the different gradients with respect to the samples
#         # in the respective sub-mini-batches, on the respective devices:
#         gradients_from_devices = [[] for _ in scattered_logits]

#         # TODO: understand
#         for i in range(0, scattered_logits[0].dim(1), self.chunk_size):

#             chunk_output = [
#                 [tensor(logits[:, i:i+self.chunk_size],
#                         requires_grad=self.optimizer_handler is not None)]
#                 for logits in scattered_logits
#             ]

#             scattered_log_probabilities = parallel_apply(
#                 modules=final_log_softmax_layer,
#                 inputs=chunk_output
#             )

#             log_probs_and_labels = [
#                 (log_probs.contiguous().view(-1, log_probs.size(-1)),
#                  targets[:, i:i+self.chunk_size].contiguous().view(-1))
#                  for log_probs, targets in zip(scattered_log_probabilities,
#                                            scattered_labels)
#             ]

#             scattered_losses = parallel_apply(modules=self.criterion,
#                                               inputs=log_probs_and_labels)

#             loss = parallel_gather(outputs=scattered_losses,
#                                 target_device=self.device_ids[0])
#             loss = loss.sum()[0] / n_mini_batch_tokens
#             total_loss += loss.data[0]

#             # computing gradients, for updating weights later, only when an
#             # optimizer is passed:
#             if self.optimizer_handler is not None:

#                 # backpropagating loss gradient to output log-probabilites -
#                 # Chain Rule is exploited to later backpropagate complete
#                 # gradients up to each weight, but gradients across
#                 # different sub-mini-batches on different devices need to be
#                 # cumulated before:
#                 loss.backward()

#                 # TODO: understand
#                 for j, _ in enumerate(scattered_losses):
#                     gradients_from_devices[j].append(
#                         chunk_output[j][0].grad.data.clone()
#                     )

#         # computing gradients and updating weights only when an optimizer is
#         # passed:
#         if self.optimizer_handler is not None:

#             # gathering all gradients from sub-mini-batches on different
#             # devices and concatenating them along the first dimenstion,
#             # that is the mini-batch size - now these are loss gradients
#             # with respect to output log-probabilities - and completing
#             # Chain Rule, backpropagating from these complete gradients
#             # to weights - equivalently to backpropagating gradient with
#             # respect to weights of the loss averaged on all samples in
#             # the whole mini-batch, i.e. on all sub-mini-batches among
#             # different devices:
#             gradients_from_devices = [
#                 # concatenating gradients along the 1st dimension TODO i.e.?
#                 tensor(torch_cat(gradients_from_devices, dim=1),
#                        requires_grad=True)
#                 for g in gradients_from_devices
#             ]
#             log_probabilities.backward(
#                 gradient=parallel_gather(
#                     outputs=gradients_from_devices,
#                     target_device=self.device_ids[0]  # device computing
#                 )
#             )

#             # updating weights based on their respective gradients,
#             # eventually cleaning such gradients for next iterations:
#             self.optimizer_handler.run_weight_update_step()
#             # NOTE: gradients of trainable weights are cleaned, for the next
#             # iteration, implicitly within this method call of the optimizer
#             # handler and not explicitly here, contrary to the original
#             # implementation

#         # returning the loss value reverted back to the original,
#         # un-normalized scale:
#         return total_loss.item() * n_mini_batch_tokens


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
            logits=output_logits,
            labels=mini_batch.tgt_expected_tokens,
            n_mini_batch_tokens=mini_batch.actual_n_target_tokens
        )

        # remembering statistics:
        cumulative_loss += loss
        cumulative_n_tokens_done += mini_batch.actual_n_target_tokens

        if verbose:

            # displaying loss every tot. mini-batches:
            display_every = 50  # [n. mini-batches]
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

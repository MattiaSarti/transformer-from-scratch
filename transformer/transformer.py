"""
Transformer, with methods for initialization, training and inference.
"""


from copy import deepcopy
from typing import Tuple, TypedDict

from torch import cat as torch_cat, max as torch_max, ones as torch_ones,\
    Tensor
from torch.cuda import is_available as cuda_is_available
from torch.nn import DataParallel, Sequential
from torch.nn.init import xavier_uniform_
from torch.optim import Adam

from transformer.architecture.attention import allowed_positions_to_attend,\
    MultiHeadAttention
from transformer.architecture.base import LogSoftmax, PositionWiseFeedForward
from transformer.architecture.embedding import Embedder, PositionalEncoding
from transformer.architecture.encoder import Encoder, EncoderBlock,\
    EncoderBlockBuildingBlocks
from transformer.architecture.decoder import Decoder, DecoderBlock,\
    DecoderBlockBuildingBlocks
from transformer.architecture.seq2seq import Seq2Seq,\
    Seq2SeqBuildingBlocks

from transformer.training_and_inference.data import\
    dataset_builder_copy_task
from transformer.training_and_inference.optimizer import OptimizerHandler
from transformer.training_and_inference.training_and_inference import\
    execute_training_epoch, LabelSmoothedLoss, LossMinimizer


class HyperparameterDict(TypedDict):
    """
    Initialization hyperparemeters with their data types.
    """

    src_vocabulary_dimension: int
    tgt_vocabulary_dimension: int
    n_encoder_blocks: int
    n_decoder_blocks: int
    representation_dimension: int
    feedforward_dimension: int
    n_attention_heads: int
    max_sequence_length: int
    dropout_prob: float


class Transformer:
    """
    Transformer model handler, with methods for initialization, training and
    inference.
    """

    def __init__(
                self,
                src_vocabulary_dimension: int,
                tgt_vocabulary_dimension: int,
                n_encoder_blocks: int = 6,
                n_decoder_blocks: int = 6,
                representation_dimension: int = 512,
                feedforward_dimension: int = 2048,
                n_attention_heads: int = 8,
                max_sequence_length: int = 5000,
                dropout_prob: float = 0.1,
                path: str = ''
                ) -> None:

        # if path not given:
        if not path:

            # setting the hyperparameters:
            hyperparameters = locals()
            del hyperparameters['path']
            self._set_hyperparameters(hyperparameters=hyperparameters)

            # initializing the new model:
            self._build_model_architecture()

        # otherwise:
        else:

            pass

            # # checking no (possibly different) hyperparameters are specified:

            # # retrieving the existing hyperparameters:
            # self.src_vocabulary_dimension=\
            #     hyperparameters['src_vocabulary_dimension'],
            # self.tgt_vocabulary_dimension=\
            #     hyperparameters['tgt_vocabulary_dimension'],
            # self.n_encoder_blocks=hyperparameters['n_encoder_blocks'],
            # self.n_decoder_blocks=hyperparameters['n_decoder_blocks'],
            # self.token_representation_dimension=\
            #     hyperparameters['representation_dimension'],
            # self.feedforward_dimension=\
            #     hyperparameters['feedforward_dimension'],
            # self.n_attention_heads=hyperparameters['n_attention_heads'],
            # self.max_sequence_length=hyperparameters['max_sequence_length'],
            # self.dropout_prob=hyperparameters['dropout_prob']

            # # loading an existing model:
            # self.model = ...load

    def _build_model_architecture(self) -> None:
        """
        Initializing the Transformer model object instantiated with the
        architecture specified by the input hyperparameters, with newly
        initialized weights.
        """
        # building the architecture:

        # instantiating (some of) the base layers/blocks of the architecture:
        positional_encoding_layer = PositionalEncoding(
            token_representation_dimension=self.representation_dimension,
            dropout_prob=self.dropout_prob,
            max_sequence_length=self.max_sequence_length
        )
        multi_head_attention_later = MultiHeadAttention(
            n_attention_heads=self.n_attention_heads,
            token_representation_dimension=self.representation_dimension,
            dropout_prob=self.dropout_prob
        )
        feedforward_layer = PositionWiseFeedForward(
            token_representation_dimension=self.representation_dimension,
            feedforward_dimension=self.feedforward_dimension,
            dropout_prob=self.dropout_prob
        )
        log_softmax_layer = LogSoftmax(
            token_representation_dimension=self.representation_dimension,
            vocabulary_dimension=self.tgt_vocabulary_dimension
        )

        # composing some of the base layers to build the more complex ones:
        src_embedder = Sequential(
            Embedder(
                token_representation_dimension=self.representation_dimension,
                vocabulary_dimension=self.src_vocabulary_dimension
            ),
            deepcopy(positional_encoding_layer)
        )
        tgt_embedder = Sequential(
            Embedder(
                token_representation_dimension=self.representation_dimension,
                vocabulary_dimension=self.tgt_vocabulary_dimension
            ),
            deepcopy(positional_encoding_layer)
        )
        base_encoder_block = EncoderBlock(
            building_blocks=EncoderBlockBuildingBlocks(
                self_multi_head_attention_layer=deepcopy(
                    multi_head_attention_later),
                fully_connected_layer=deepcopy(feedforward_layer),
            ),
            feature_dimension=self.representation_dimension,
            dropout_prob=self.dropout_prob
        )
        encoder = Encoder(
            base_block=base_encoder_block,
            n_clones=self.n_encoder_blocks
        )
        base_decoder_block = DecoderBlock(
            building_blocks=DecoderBlockBuildingBlocks(
                self_multi_head_attention_layer=deepcopy(
                    multi_head_attention_later),
                source_multi_head_attention_layer=deepcopy(
                    multi_head_attention_later),
                fully_connected_layer=deepcopy(feedforward_layer)
            ),
            feature_dimension=self.representation_dimension,
            dropout_prob=self.dropout_prob
        )
        decoder = Decoder(
            base_block=base_decoder_block,
            n_clones=self.n_decoder_blocks
        )

        # instantiating the whole seq2seq encoder-decoder model:
        building_blocks = Seq2SeqBuildingBlocks(
            encoder=encoder,
            decoder=decoder,
            src_embedder=src_embedder,
            tgt_embedder=tgt_embedder,
            log_softmax_layer=log_softmax_layer
        )
        model = Seq2Seq(
            building_blocks=building_blocks
        )

        # initializing the parameters:

        # for each layer's parameter set:
        for parameter in model.parameters():
            # TODO: explain why:
            if parameter.dim() > 1:
                # parameters initialized following Xavier initialization:
                xavier_uniform_(parameter)

        self.model = model

    def _set_hyperparameters(self, hyperparameters: HyperparameterDict)\
            -> None:
        """
        Remembering the hyperparameters.
        """
        for key, value in hyperparameters.items():
            setattr(self, key, value)

    def predict(
                self,
                src_sequences: Tensor,
                src_masks: Tensor,
                tgt_bos_token: int,
                decoding_method: str = 'greedy',
                gpu_if_possible: bool = True
                ) -> Tensor:
        """
        Predict target token sequences from source token sequences.
        """
        # selecting the device handling computations:
        if gpu_if_possible:
            # employing a GPU if possible:
            device = 'cuda:0' if cuda_is_available() else 'cpu'
        else:
            device = 'cpu'

        # moving model parameters and buffers to such device:
        self.model.to(device)

        # moving inputs to such device:
        src_sequences = src_sequences.to(device)
        src_masks = src_masks.to(device)
        print()

        # switching to inference mode:
        self.model.eval()

        if decoding_method == 'greedy':

            # greedy decoding:

            # computing encoder outputs, i.e. encoded representations of
            # source tokens - from dimensionality (samples, tokens) to
            # dimensionality (samples, tokens, features):
            src_encoded_tokens = self.model.encode(
                src_tokens=src_sequences,
                src_mask=src_masks
            )

            # initializing predicted output sequences:
            cumulative_tgt_sequences = torch_ones((1, 1), requires_grad=False)\
                .fill_(value=tgt_bos_token).type_as(src_sequences)

            # for each target position, the respective token is sequentially
            # predicted, given the decoder auto-regressive predictive nature -
            # for all sequences at the same time:
            for _ in range(self.max_sequence_length - 1):

                # computing logits - from dimensionality (samples, tokens,
                # features) to dimensionality (samples, tokens, features):
                next_token_logits = self.model.decode(
                    src_encoded_tokens=src_encoded_tokens,
                    src_mask=src_masks,
                    tgt_tokens=cumulative_tgt_sequences,
                    tgt_mask=allowed_positions_to_attend(
                        # positions to attend equal computed target tokens:
                        n_positions=cumulative_tgt_sequences.size(1)
                    ).to(device)
                )

                # turning the logits of next (last) tokens in the sequences
                # into log-probabilities - from dimensionality (samples,
                # tokens, features) to dimensionality (samples, features):
                next_token_log_probabilities = self.model.log_softmax_layer(
                    next_token_logits[:, -1]  # next (last) tokens
                )

                # discretizing probabilities to predicted tokens - from
                # dimensionality (samples, features) to dimensionality
                # (samples):
                next_tokens = torch_max(next_token_log_probabilities,
                                        dim=1).indices[0]

                # concatenating the newly predicted tokens to the sequences of
                # already predicted tokens:
                cumulative_tgt_sequences = torch_cat(
                    (
                        cumulative_tgt_sequences,
                        torch_ones((1, 1)).type_as(src_sequences).fill_(
                            next_tokens
                        )
                    ),
                    dim=1
                )
                # FIXME: shapes not understood

                # TODO: truncate the different predicted sequences in the
                # mini-batch from their respective first padding token on

            return cumulative_tgt_sequences

        raise NotImplementedError("Unavailable decoding method: "
                                  + decoding_method)

    def train_on_toy_copy_task(
                self,
                n_epochs: int,
                epoch_samples: int,
                mini_batch_size: int,
                label_smoothing_factor: float,
                learning_rate_n_warmup_steps: int = 4000,
                learning_rate_amplification_factor: float = 2,
                adam_betas: Tuple[float, float] = (0.9, 0.98),
                adam_epsilon: float = 1e-9,
                gpu_if_possible: bool = True
                ) -> None:
        """
        Training the model on a toy task: copying the source sentence, with an
        identical target. Teacher forcing is applied, avoiding sequential
        decoding and improving training convergence in spite of accuracy
        at inference time (with actual auto-regressive outputs).
        """
        assert self.src_vocabulary_dimension == self\
            .tgt_vocabulary_dimension, "For this toy task, the source and"\
            + " target vocabularies have to be shared."

        # for this toy task, the padding token index has to be 0:
        padding_token = 0

        criterion = LabelSmoothedLoss(
            softmax_dimension=self.tgt_vocabulary_dimension,
            padding_token=padding_token,
            smoothing_factor=label_smoothing_factor
        )

        # selecting the device handling computations:
        if gpu_if_possible:
            # employing a GPU if possible:
            device = 'cuda:0' if cuda_is_available() else 'cpu'
        else:
            device = 'cpu'

        # moving the model parameters and buffers to such device:
        self.model.to(device)

        # moving the criterion parameters and buffers to such device:
        criterion.to(device)

        # NOTE: this way, the optimizer handler operates on objects that
        # are already on the device of interest, so it does not have to be
        # moved there

        optimizer_handler = OptimizerHandler(
            optimizer=Adam(
                params=self.model.parameters(),
                lr=0,  # as learning rate is customized externally
                betas=adam_betas,
                eps=adam_epsilon
            ),
            n_warmup_steps=learning_rate_n_warmup_steps,
            amplification_factor=learning_rate_amplification_factor,
            model_hidden_dimension=self.representation_dimension
        )

        # for each training epoch:
        for epoch in range(n_epochs):

            print('-' * 60)
            print("Epoch " + str(epoch + 1) + "/" + str(n_epochs))

            # switching to training mode:
            self.model.train()

            # executing a training epoch:
            _ = execute_training_epoch(
                dataset_iterator=dataset_builder_copy_task(
                    sequence_length=self.max_sequence_length,
                    vocabulary_size=self.src_vocabulary_dimension,
                    mini_batch_size=mini_batch_size,
                    n_mini_batches=(int(epoch_samples / mini_batch_size)),
                    gpu_if_possible=gpu_if_possible
                ),
                model=self.model,
                loss_minimizer=LossMinimizer(
                    final_log_softmax_layer=self.model.log_softmax_layer,
                    criterion=criterion,
                    optimizer_handler=optimizer_handler
                ),
                verbose=True
            )

            # back to inference mode:
            self.model.eval()

            # evaluating performances:
            loss = execute_training_epoch(
                dataset_iterator=dataset_builder_copy_task(
                    sequence_length=self.max_sequence_length,
                    vocabulary_size=self.src_vocabulary_dimension,
                    mini_batch_size=mini_batch_size,
                    n_mini_batches=(int(n_epochs / mini_batch_size) + 1),
                    gpu_if_possible=gpu_if_possible
                ),
                model=self.model,
                loss_minimizer=LossMinimizer(
                    final_log_softmax_layer=self.model.log_softmax_layer,
                    criterion=criterion,
                    # no backpropagation and weight update:
                    optimizer_handler=None
                ),
                verbose=False
            )
            print("Average Loss per Token: {l:.3f}".format(l=loss))

        print('-' * 60)

    def train_on_IWSLT(
                self,
                n_epochs: int = 10,
                mini_batch_size: int = 12000,
                label_smoothing_factor: float = 0.1,
                learning_rate_n_warmup_steps: int = 2000,
                learning_rate_amplification_factor: float = 1,
                adam_betas: Tuple[float, float] = (0.9, 0.98),
                adam_epsilon: float = 1e-9
                ) -> None:
        """
        Training the model on an IWSLT 2016 TED talks: the { German -> English }
        translation task.
        """
        raise NotImplementedError
    #     # identifying GPU devices used to parallelize operations:
    #     device_ids = [0, 1, 2, 3]

    #     # ensuring the mini-batch size is greater than (or at least equal to)
    #     # the number of GPUs employed:
    #     assert mini_batch_size >= len(device_ids), "Mini-batch size must " +\
    #         "be greater than (at least equal to) the number of GPUs employed."

    #     training_iterator, validation_iterator,\
    #          padding_token = dataset_builder_IWSLT_task(

    #     )

    #     criterion = LabelSmoothedLoss(
    #         softmax_dimension=self.tgt_vocabulary_dimension,
    #         padding_token=padding_token,
    #         smoothing_factor=label_smoothing_factor
    #     )

    #     # moving the model parameters and buffers to the main GPU:
    #     self.model.to('cuda:0')

    #     # moving the criterion parameters and buffers to the main GPU:
    #     criterion.to('cuda:0')

    #     optimizer_handler = OptimizerHandler(
    #         optimizer=Adam(
    #             params=self.model.parameters(),
    #             lr=0,  # as learning rate is customized externally
    #             betas=adam_betas,
    #             eps=adam_epsilon
    #         ),
    #         n_warmup_steps=learning_rate_n_warmup_steps,
    #         amplification_factor=learning_rate_amplification_factor,
    #         model_hidden_dimension=self.representation_dimension
    #     )

    #     # NOTE: the optimizer handler operates on objects that are already
    #     # on GPU(s), so it does not need to be moved there

    #     # NOTE: the model must have its parameters and buffers on
    #     # device_ids[0] before parallelizing it, and so does it now

    #     # parallelizing the model, i.e. copying the model parameters across
    #     # the chosen GPU devices and allowing to split inputs across such
    #     # devices splitting them along the first dimension, i.e. the
    #     # mini-batch dimension - during forward propagation, the model is
    #     # independently replicated on each device, with each replica
    #     # handling the respective portion of input samples, while, during
    #     # backpropagation, gradients from each replica are summed into the
    #     # "original" model, converging all the mini-batch information on a
    #     # single device for weight update:
    #     parallelized_model = DataParallel(self.model, device_ids=device_ids)

    #     # for each training epoch:
    #     for epoch in range(n_epochs):

    #         print('-' * 60)
    #         print("Epoch " + str(epoch + 1) + "/" + str(n_epochs))

    #         # switching to training mode:
    #         parallelized_model.train()

    #         # executing a training epoch:
    #         _ = execute_training_epoch(
    #             dataset_iterator=training_iterator,
    #             model=parallelized_model,
    #             loss_minimizer=DataParallelLossMinimizer(
    #                 final_log_softmax_layer=self.model.log_softmax_layer,
    #                 criterion=criterion,
    #                 device_ids=device_ids,
    #                 optimizer_handler=optimizer_handler
    #             ),
    #             verbose=True
    #         )

    #         # back to inference mode:
    #         parallelized_model.eval()

    #         # evaluating performances:
    #         loss = execute_training_epoch(
    #             dataset_iterator=validation_iterator,
    #             model=parallelized_model,
    #             loss_minimizer=DataParallelLossMinimizer(
    #                 final_log_softmax_layer=self.model.log_softmax_layer,
    #                 criterion=criterion,
    #                 device_ids=device_ids,
    #                 # no backpropagation and weight update:
    #                 optimizer_handler=None
    #             ),
    #             verbose=False
    #         )
    #         print("Average Loss per Token: {l:.3f}".format(l=loss))

    #     print('-' * 60)

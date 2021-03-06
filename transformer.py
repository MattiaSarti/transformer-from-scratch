from copy import deepcopy
from typing import Tuple

from torch import cat as torch_cat
from torch import Tensor
from torch import max as torch_max
from torch import ones as torch_ones
from torch.nn import Module, Sequential
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
from transformer.architecture.seq2seq import EncoderDecoder,\
    Seq2SeqBuildingBlocks

from transformer.training.training import copy_task_dataset_builder,\
    execute_training_epoch, LabelSmoothedLoss, LossMinimizer, OptimizerHandler


class Transformer:
    """
    """
    def __init__(
        self,
        src_vocabulary_dimension: int,
        tgt_vocabulary_dimension: int,
        n_encoder_blocks: int = 6,
        n_decoder_blocks: int = 6,
        token_representation_dimension: int = 512,
        feedforward_dimension: int = 2048,
        n_attention_heads: int = 8,
        max_sequence_length: int = 5000,
        dropout_prob: float = 0.1,
        path: str = ''
    ) -> None:
        # if path not given, initializing a new model:
        if not path:
            self.hyperparameters = {
                'src_vocabulary_dimension': src_vocabulary_dimension,
                'tgt_vocabulary_dimension': tgt_vocabulary_dimension,
                'n_encoder_blocks': n_encoder_blocks,
                'n_decoder_blocks': n_decoder_blocks,
                'representation_dimension':
                    token_representation_dimension,
                'feedforward_dimension': feedforward_dimension,
                'n_attention_heads': n_attention_heads,
                'max_sequence_length': max_sequence_length,
                'dropout_prob': dropout_prob
            }
            self.model = self.build_model_architecture(
                **self.hyperparameters
            )
        # otherwise, loading existing model:
        else:
            pass
            # self.model = ...load
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

    def build_model_architecture(
        self,
        src_vocabulary_dimension: int,
        tgt_vocabulary_dimension: int,
        n_encoder_blocks: int,
        n_decoder_blocks: int,
        representation_dimension: int,
        feedforward_dimension: int,
        n_attention_heads: int,
        max_sequence_length: int,
        dropout_prob: float
    ) -> Module:
        """
        Return a Transformer model object instantiated with the architecture
        specified by the input hyperparameters, with newly initialized
        weights.
        """

        # building the architecture:

        # instantiating (some of) the base layers/blocks of the architecture:
        positional_encoding_layer = PositionalEncoding(
            token_representation_dimension=representation_dimension,
            dropout_prob=dropout_prob,
            max_sequence_length=max_sequence_length
        )
        multi_headed_attention_later = MultiHeadAttention(
            n_attention_heads=n_attention_heads,
            token_representation_dimension=representation_dimension,
            dropout_prob=dropout_prob
        )
        feedforward_layer = PositionWiseFeedForward(
            token_representation_dimension=representation_dimension,
            feedforward_dimension=feedforward_dimension,
            dropout_prob=dropout_prob
        )
        log_softmax_layer = LogSoftmax(
            token_representation_dimension=representation_dimension,
            vocabulary_dimension=tgt_vocabulary_dimension
        )

        # composing some of the base layers to build the more complex ones:
        src_embedder = Sequential(
            Embedder(
                token_representation_dimension=representation_dimension,
                vocabulary_dimension=src_vocabulary_dimension
            ),
            deepcopy(positional_encoding_layer)
        )
        tgt_embedder = Sequential(
            Embedder(
                token_representation_dimension=representation_dimension,
                vocabulary_dimension=tgt_vocabulary_dimension
            ),
            deepcopy(positional_encoding_layer)
        )
        base_encoder_block = EncoderBlock(
            building_blocks=EncoderBlockBuildingBlocks(
                self_multi_headed_attention_layer=deepcopy(
                    multi_headed_attention_later),
                fully_connected_layer=deepcopy(feedforward_layer),
            ),
            feature_dimension=representation_dimension,
            dropout_prob=dropout_prob
        )
        encoder = Encoder(
            base_block=base_encoder_block,
            n_clones=n_encoder_blocks
        )
        base_decoder_block = DecoderBlock(
            building_blocks=DecoderBlockBuildingBlocks(
                self_multi_headed_attention_layer=deepcopy(
                    multi_headed_attention_later),
                source_multi_headed_attention_layer=deepcopy(
                    multi_headed_attention_later),
                fully_connected_layer=deepcopy(feedforward_layer)
            ),
            feature_dimension=representation_dimension,
            dropout_prob=dropout_prob
        )
        decoder = Decoder(
            base_block=base_decoder_block,
            n_clones=n_encoder_blocks
        )

        # instantiating the whole seq2seq encoder-decoder model:
        building_blocks = Seq2SeqBuildingBlocks(
            encoder=encoder,
            decoder=decoder,
            src_embedder=src_embedder,
            tgt_embedder=tgt_embedder,
            log_softmax_layer=log_softmax_layer
        )
        model = EncoderDecoder(
            building_blocks=building_blocks
        )

        # initializing the parameters:

        # for each layer's parameter set:
        for parameter in model.parameters():
            # TODO: explain why:
            if parameter.dim() > 1:
                # parameters initialized following Xavier initialization:
                xavier_uniform_(parameter)

        return model

    def train_on_toy_copy_task(
                self,
                epoch_samples: int,
                mini_batch_size: int,
                n_epochs: int,
                padding_token: int,
                label_smoothing_factor: float,
                learning_rate_n_warmup_steps: int = 4000,
                learning_rate_amplification_factor: float = 2,
                adam_betas: Tuple[float, float] = (0.9, 0.98),
                adam_epsilon: float = 1e-9
            ) -> None:
        """
        Execute the whole training of the model.
        """
        assert self.hyperparameters['src_vocabulary_dimension'] == self\
            .hyperparameters['tgt_vocabulary_dimension'], "For this toy"\
            + " task, the source and target vocabularies have to be shared."
        assert padding_token == 0, "For this toy task, the padding token"\
            + " index has to be 0."
        assert self.hyperparameters['max_sequence_length'] == 10, "For this"\
            + " toy task, the maximum sequence length has to be 10."

        criterion = LabelSmoothedLoss(
            softmax_dimension=self.hyperparameters[
                                   'tgt_vocabulary_dimension'],
            padding_token=padding_token,
            smoothing_factor=label_smoothing_factor
        )
        optimizer_handler = OptimizerHandler(
            optimizer=Adam(
                params=self.model.parameters(),
                lr=0,  # as learning rate is customized externally
                betas=adam_betas,
                eps=adam_epsilon
            ),
            n_warmup_steps=learning_rate_n_warmup_steps,
            amplification_factor=learning_rate_amplification_factor,
            model_hidden_dimension=self.hyperparameters[
                                        'representation_dimension']
        )

        # for each training epoch:
        for epoch in range(n_epochs):

            print('-' * 60)
            print("Epoch " + str(epoch + 1) + "/" + str(n_epochs))

            # switching to training mode:
            self.model.train()

            # executing a training epoch:
            _ = execute_training_epoch(
                dataset_iterator=copy_task_dataset_builder(
                    vocabulary_size=self.hyperparameters[
                                         'src_vocabulary_dimension'],
                    mini_batch_size=mini_batch_size,
                    n_mini_batches=(int(epoch_samples / mini_batch_size))
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
                dataset_iterator=copy_task_dataset_builder(
                    vocabulary_size=self.hyperparameters[
                                        'src_vocabulary_dimension'],
                    mini_batch_size=mini_batch_size,
                    n_mini_batches=(int(n_epochs / mini_batch_size) + 1)
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

    def predict(
                self,
                src_sequences: Tensor,
                src_masks: Tensor,
                tgt_bos_token: Tensor,
                decoding_method: str = 'greedy'
            ) -> Tensor:
        """
        Predict target token sequences from source token sequences.
        """
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
            for _ in range(self.hyperparameters['max_sequence_length'] - 1):

                # computing logits - from dimensionality (samples, tokens,
                # features) to dimensionality (samples, tokens, features):
                next_token_logits = self.model.decode(
                    src_encoded_tokens=src_encoded_tokens,
                    src_mask=src_masks,
                    tgt_tokens=cumulative_tgt_sequences,
                    tgt_mask=allowed_positions_to_attend(
                        # positions to attend equal computed target tokens:
                        n_positions=cumulative_tgt_sequences.size(1)
                    )
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
                        torch_ones((1, 1)).type_as(src_sequences)
                                          .fill_(next_tokens)
                    ),
                    dim=1
                )
                # FIXME: shapes not understood

            return cumulative_tgt_sequences

        elif False:  # TODO

            pass

        else:

            raise Exception("Unknown decoding method for prediction: "
                            + decoding_method)

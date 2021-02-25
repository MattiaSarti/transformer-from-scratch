from copy import deepcopy
from math import log, sqrt

from numpy import ones as np_ones
from numpy import tril
from torch import arange as torch_arange
from torch import cos as torch_cos
from torch import exp as torch_exp
from torch import from_numpy, matmul, nn, tensor, Tensor
from torch import ones as torch_ones
from torch import sin as torch_sin
from torch import zeros as torch_zeros
from torch.autograd import Variable
from torch.nn import functional as F

from .attention import MultiHeadedAttention
from .base import PositionWiseFeedForward
from .embedding import PositionalEncoding
from .encoder import Encoder, EncoderBlock
from .decoder import Decoder, DecoderBlock
from training.training import LabelSmoothedLoss, OptimizerHandler


class EncoderDecoder(nn.Module):
    """
    Base architecture for encoder-decoder sequence-to-sequence Transformer
    models.
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, src_embedder:
                 nn.Module, tgt_embedder: nn.Module, log_softmax_layer:
                 nn.Module) -> None:
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.log_softmax_layer = log_softmax_layer
        # (not used for building the architecture!! maybe because it is only
        # uesd at inference time, while the training loss starts from its
        # inputs to optimize performances)

    def forward(self, src_tokens: Tensor, tgt_tokens: Tensor, src_mask:
                Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Process both masked source and target sequences.
        """
        return self.decode(
            src_encoded_tokens=self.encode(
                src_tokens=src_tokens,
                src_mask=src_mask
            ),
            src_mask=src_mask,
            tgt_tokens=tgt_tokens,
            tgt_mask=tgt_mask
        )

    def encode(self, src_tokens: Tensor, src_mask: Tensor) -> Tensor:
        """
        """
        return self.encoder(
            x=self.src_embedder(src_tokens),
            mask=src_mask
        )

    def decode(self, src_encoded_tokens: Tensor, src_mask: Tensor,
               tgt_tokens: Tensor, tgt_mask: Tensor) -> Tensor:
        """
        """
        return self.decoder(
            x=self.tgt_embedder(tgt_tokens),
            src_encoded_tokens=src_encoded_tokens,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )


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
                'token_representation_dimension':
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
            #     hyperparameters['token_representation_dimension'],
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
        token_representation_dimension: int,
        feedforward_dimension: int,
        n_attention_heads: int,
        max_sequence_length: int,
        dropout_prob: float
    ) -> nn.Module:
        """
        Return a Transformer model object instantiated with the architecture
        specified by the input hyperparameters, with newly initialized
        weights.
        """

        # building the architecture:

        # instantiating base layers:
        multi_headed_attention_later = MultiHeadedAttention(
            n_attention_heads=n_attention_heads,
            token_representation_dimension=token_representation_dimension,
            dropout_prob=dropout_prob
        )
        feedforward_layer = PositionWiseFeedForward(
            token_representation_dimension=token_representation_dimension,
            feedforward_dimension=feedforward_dimension,
            dropout_prob=dropout_prob
        )
        positional_encoding_layer = PositionalEncoding(
            token_representation_dimension=token_representation_dimension,
            dropout_prob=dropout_prob,
            max_sequence_length=max_sequence_length
        )
        # composing base layers to build the whole model architecture:
        model = EncoderDecoder(
            encoder=Encoder(
                base_block=EncoderBlock(
                    feature_dimension=token_representation_dimension,
                    self_multi_headed_attention_layer=deepcopy(
                        multi_headed_attention_later),
                    fully_connected_layer=deepcopy(feedforward_layer),
                    dropout_prob=dropout_prob
                ),
                n_clones=n_encoder_blocks
            ),
            decoder=Decoder(
                base_block=DecoderBlock(
                    feature_dimension=token_representation_dimension,
                    self_multi_headed_attention_layer=deepcopy(
                        multi_headed_attention_later),
                    source_multi_headed_attention_layer=deepcopy(
                        multi_headed_attention_later),
                    fully_connected_layer=deepcopy(feedforward_layer),
                    dropout_prob=dropout_prob
                ),
                n_clones=n_encoder_blocks
            ),
            src_embedder=nn.Sequential(
                Embedder(
                    token_representation_dimension=
                        token_representation_dimension,
                    vocabulary_dimension=src_vocabulary_dimension
                ),
                deepcopy(positional_encoding_layer)
            ),
            tgt_embedder=nn.Sequential(
                Embedder(
                    token_representation_dimension=
                        token_representation_dimension,
                    vocabulary_dimension=tgt_vocabulary_dimension
                ),
                deepcopy(positional_encoding_layer)
            ),
            log_softmax_layer=LogSoftmax(
                token_representation_dimension=token_representation_dimension,
                vocabulary_dimension=tgt_vocabulary_dimension
            )
        )

        # initializing the parameters:

        # for each layer's parameter set:
        for parameter in model.parameters():
            # TODO: explain why:
            if parameter.dim() > 1:
                # parameters initialized following Xavier initialization:
                nn.init.xavier_uniform(parameter)

        return model

    def train(self, n_epochs: int, padding_token: int,
              label_smoothing_factor: int, learning_rate_n_warmup_steps: int,
              learning_rate_amplification_factor: float) -> None:
        """
        Execute the whole training of the model.
        """

        criterion = LabelSmoothedLoss(
            softmax_dimension=self.\
                hyperparameters['tgt_vocabulary_dimension'],
            padding_token=padding_token,
            smoothing_factor=label_smoothing_factor
        )
        optimizer = OptimizerHandler(
            optimizer=Adam(

            ),
            n_warmup_steps=learning_rate_n_warmup_steps, 
            amplification_factor=learning_rate_amplification_factor,
            model_hidden_dimension=self.\
                hyperparameters['token_representation_dimension']
        )

        # for each training epoch:
        for epoch in range(n_epochs):
            # switching to training mode:
            self.model.train()
            # :
            execute_training_epoch(

            )
            # back to inference mode:
            self.model.eval()
            

    def predict(self, src_sequence: Tensor, src_mask: Tensor, tgt_bos_token:
                Tensor, decoding_method: str='greedy') -> Tensor:
        """
        Predict a single source token sequence as an output, target token 
        sequence.
        """
        # switching to inference mode:
        self.model.eval()

        if decode_method == 'greedy':

            # greedy decoding:

            # computing hidden (i.e. encoded) representations of source 
            # tokens:
            src_encoded_tokens=self.model.encode(
                src_tokens=src_sequence,
                src_mask=src_mask
            )
            # initializing predicted output sequence:
            cumulative_tgt_sequence = torch_ones((1, 1), requires_grad=False)\
                .fill_(value=tgt_bos_token).type_as(src_sequence)
            # for each target position, the respective token in sequentially
            # predicted, given the decoder auto-regressive prediction nature:
            for i in range(self.hyperparameters['max_sequence_length'] - 1):
                self.model.decode(
                    src_encoded_tokens=src_encoded_tokens,
                    src_mask=src_mask,
                    tgt_tokens=cumulative_tgt_sequence,
                    tgt_mask=allowed_positions_to_attend(n_positions=)
                )

            return

        elif False:

            pass

        else:

            raise Exception("Unknown decoding method for prediction: " + decoding_method)

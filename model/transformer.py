from .attention import *
from .base import *
from .embedding import *
from .encoder import *
from .decoder import *


class EncoderDecoder(nn.Module):
    """
    Base architecture for encoder-decoder sequence-to-sequence Transformer 
    models.
    """
    def __init__(
        self, encoder, decoder, src_embedder, tgt_embedder, log_softmax_layer
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.log_softmax_layer = log_softmax_layer # (not used for building the architecture!! maybe because it is only uesd at inference time, while the training loss starts from its inputs to optimize performances)

    def forward(self, src_sequence, tgt_sequence, src_mask, tgt_mask):
        """
        Process both masked source and target sequences.
        """
        return self.decode(
            self.encode(src_sequence, src_mask),
            src_mask,
            tgt_sequence,
            tgt_mask
        )
    
    def encode(self, src_sequence, src_mask):
        """
        """
        return self.encoder(
            self.src_embedder(src_sequence),
            src_mask
        )
    
    def decode(self, memory, src_mask, tgt_sequence, tgt_mask):
        """
        """
        return self.decoder(
            self.tgt_embedder(tgt_sequence),
            memory,
            src_mask,
            tgt_mask
        )


class Transformer:
    """
    """
    def __init__(self, path: str=''):
        # if not path given, initializing a new model:
        if not path:
            pass
            # self.model = self.build_model_architecture(
            #     ...
            # )
        else:
            pass
            # self.model = ...load

    def build_model_architecture(
        self,
        src_vocabulary_dimension: int,
        tgt_vocabulary_dimension: int,
        n_encoder_blocks: int=6,
        n_decoder_blocks: int=6,
        token_representation_dimension: int=512,
        feedforward_dimension: int=2048,
        n_attention_heads: int=8,
        max_sequence_length: int=5000,
        dropout_prob: float=0.1
    ) -> nn.Module:
        """
        Return a Transformer model object instantiated with the architecture 
        specified by the input hyperparameters, with newly initialized weights.
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
                    self_multi_headed_attention_layer=\
                        deepcopy(multi_headed_attention_later),
                    fully_connected_layer=deepcopy(feedforward_layer),
                    dropout_prob=dropout_prob
                ),
                n_clones=n_encoder_blocks
            ),
            decoder=Decoder(
                base_block=DecoderBlock(
                    feature_dimension=token_representation_dimension,
                    self_multi_headed_attention_layer=\
                        deepcopy(multi_headed_attention_later),
                    source_multi_headed_attention_layer=\
                        deepcopy(multi_headed_attention_later),
                    fully_connected_layer=deepcopy(feedforward_layer),
                    dropout_prob=dropout_prob
                ),
                n_clones=n_encoder_blocks
            ),
            src_embedder=nn.Sequential(
                Embedder(
                    token_representation_dimension=\
                        token_representation_dimension,
                    vocabulary_dimension=src_vocabulary_dimension
                ),
                deepcopy(positional_encoding_layer)
            ),
            tgt_embedder=nn.Sequential(
                Embedder(
                    token_representation_dimension=\
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

        # initializing the hyperparameters:

        # for each layer's parameter set:
        for parameter in model.parameters():
            # TODO: explain why:
            if parameter.dim() > 1:
                # parameters initialized following Xavier initialization:
                nn.init.xavier_uniform(parameter)

        return model

    def train(self):
        pass

    def predict(self):
        pass
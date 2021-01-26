from copy import deepcopy
from math import log, sqrt
from typing import Type

from numpy import ones as np_ones
from numpy import tril
from torch import arange as torch_arange
from torch import cos as torch_cos
from torch import cos as torch_exp
from torch import from_numpy, matmul, nn, tensor, Tensor
from torch import ones as torch_ones
from torch import sin as torch_sin
from torch import zeros as torch_zeros
from torch.autograd import Variable
from torch.nn import functional as F


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
        self.log_softmax_layer = log_softmax_layer # (not used for building the architecture!! maybe because it is only uesd at inference time, while the training loss starts from its inputs to optimized performances)

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


##############################################################################


class LogSoftmax(nn.Module):
    """
    Linear layer followed by log-softmax activation function.
    """
    def __init__(self, token_representation_dimension: int, vocabulary_dimension: int):
        super(LogSoftmax, self).__init__()
        self.linear_layer = nn.Linear(in_features=token_representation_dimension,
            out_features=vocabulary_dimension)

    def forward(self, x: Type[Tensor]) -> Type[Tensor]:
        return F.log_softmax(self.linear_layer(x), dim=-1)


##############################################################################


def get_clones(module_to_be_cloned, n_clones):
    """
    Produce 'n' "architecturally" identical layers, without shared parameters.
    """
    return nn.ModuleList(
        [deepcopy(module_to_be_cloned) for _ in range(n_clones)]
    )


##############################################################################


class Encoder(nn.Module):
    """
    Whole encoder, composed of repeated encoder blocks which do not share 
    parameters.
    """
    def __init__(self, base_block, n_clones):
        super(Encoder, self).__init__()
        self.layers = get_clones(
            module_to_be_cloned=base_block,
            n_clones=n_clones
        )
        self.normalization_layer = LayerNorm(base_block.feature_dimension) # TODO: see TODO below

        def forward(self, x: Type[Tensor], mask: Type[Tensor]) \
            -> Type[Tensor]:
            # forwarding inputs throught all encoder blocks:
            for layer in self.layers:
                x = layer(x=x, mask=mask)
            return self.normalization_layer(x) # TODO: understand why this last, additional normalization and why it is not to be masked


##############################################################################


class LayerNorm(nn.Module):
    """
    Layer-normalization layer.
    """
    def __init__(self, feature_dimension: int, epsilon: float=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(data=torch_ones(feature_dimension))
        self.beta = nn.Parameter(data=torch_zeros(feature_dimension))
        self.epsilon = epsilon

    def forward(self, x: Type[Tensor]) -> Type[Tensor]:
       mean = x.mean(dim=-1, keepdim=True)
       std = x.std(dim=-1, keepdim=True)
       return ((x - mean) / (std + self.epsilon)) * self.alpha + self.beta


##############################################################################


class ResidualConnectionAndLayerNorm(nn.Module):
    """
    Residual connection around a base layer, after dropout is applied to the 
    base layer's output, eventually followed by layer-normalization.
    """
    def __init__(self, feature_dimension: int, dropout_prob: float):
        super(ResidualConnectionAndLayerNorm, self).__init__()
        self.layer_normalization_layer = LayerNorm(feature_dimension=\
            feature_dimension)
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        
    def forward(self, x, base_layer: Type[nn.Module]):
        return self.layer_normalization_layer(x + self.dropout_layer(base_layer(x)))
    # TODO: understand why norm is applied first instead of last in their implementation: "Note for code simplicity the norm is first as opposed to last."


##############################################################################


class EncoderBlock(nn.Module):
    """
    Core encoder block, composed of, from inputs to outputs:
    - multi-headed self-attention layer;
    - residual connection;
    - layer-normalization layer;
    - fully-connected (feed-forward) layer;
    - residual connection;
    - layer-normalization layer.
    """
    def __init__(self, feature_dimension: int,
        self_multi_headed_attention_layer: Type[nn.Module], 
        fully_connected_layer: Type[nn.Module], dropout_prob: float):
        super(EncoderBlock, self).__init__()
        self.feature_dimension = feature_dimension 
        self.self_multi_headed_attention_layer = \
            self_multi_headed_attention_layer
        self.fully_connected_layer = fully_connected_layer
        self.residual_connection_blocks = get_clones(
            module_to_be_cloned=ResidualConnectionAndLayerNorm(
                feature_dimension=feature_dimension,
                dropout_prob=dropout_prob
            ),
            n_clones=2
        )

    def forward(self, x: Type[Tensor], mask: Type[Tensor]):
        # self-attention, towards encoder token positions themselves, followed
        # by residual connection and layer normalization:
        x = self.residual_connection_blocks[0](
            x,
            lambda x: self.self_multi_headed_attention_layer(
                query_tokens=x,
                key_or_value_tokens=x,
                mask=mask
            )
        )
        # fully-connected (feed-forward) layer followed by residual connection
        # and layer normalization:
        return self.residual_connection_blocks[0](x, \
            self.fully_connected_layer)


##############################################################################


class Decoder(nn.Module):
    """
    Whole decoder, composed of repeated decoder blocks which do not share 
    parameters.
    """
    def __init__(self, base_block, n_clones):
        super(Decoder, self).__init__()
        self.layers = get_clones(
            module_to_be_cloned=base_block,
            n_clones=n_clones
        )
        self.normalization_layer = LayerNorm(base_block.feature_dimension) # TODO: see TODO below

        def forward(self, x: Type[Tensor], src_encoded_tokens: Type[Tensor],
            src_mask: Type[Tensor], tgt_mask: Type[Tensor]) -> Type[Tensor]:
            # forwarding inputs throught all decoder blocks:
            for layer in self.layers:
                x = layer(x=x, src_encoded_tokens=src_encoded_tokens, \
                    src_mask=src_mask, tgt_mask=tgt_mask)
            return self.normalization_layer(x) # TODO: understand why this last, additional normalization and why it is not to be masked


##############################################################################


class DecoderBlock(nn.Module):
    """
    Core decoder block, composed of, from inputs to outputs:
    - multi-headed self-attention layer;
    - residual connection;
    - layer-normalization layer;
    - multi-headed source-attention layer;
    - residual connection;
    - layer-normalization layer;
    - fully-connected (feed-forward) layer;
    - residual connection;
    - layer-normalization layer.
    """
    def __init__(self, feature_dimension: int, 
        self_multi_headed_attention_layer: Type[nn.Module],
        source_multi_headed_attention_layer: Type[nn.Module],
        fully_connected_layer: Type[nn.Module], dropout_prob: float):
        super(DecoderBlock, self).__init__()
        self.feature_dimension = feature_dimension 
        self.self_multi_headed_attention_layer = \
            self_multi_headed_attention_layer
        self.source_multi_headed_attention_layer = \
            source_multi_headed_attention_layer
        self.fully_connected_layer = None
        self.residual_connection_blocks = get_clones(
            module_to_be_cloned=ResidualConnectionAndLayerNorm(
                feature_dimension=feature_dimension,
                dropout_prob=dropout_prob
            ),
            n_clones=3
        )

    def forward(self, x: Type[Tensor], src_encoded_tokens: Type[Tensor],
        src_mask: Type[Tensor], tgt_mask: Type[Tensor]):
        # self-attention, towards decoder token positions themselves, followed
        # by residual connection and layer normalization:
        x = self.residual_connection_blocks[0](
            x,
            lambda x: self.self_multi_headed_attention_layer(
                query_tokens=x,
                key_or_value_tokens=x,
                mask=tgt_mask
            )
        )
        # source-attention, towards (final representations of) encoder token 
        # positions, followed by residual connection and layer normalization:
        x = self.residual_connection_blocks[1](
            x,
            lambda x: self.source_multi_headed_attention_layer(
                query_tokens=x,
                key_or_value_tokens=src_encoded_tokens,
                mask=src_mask
            )
        )
        # fully-connected (feed-forward) layer followed by residual connection
        # and layer normalization:
        return self.residual_connection_blocks[2](x, \
            self.fully_connected_layer)


##############################################################################


def allowed_positions_to_attend(n_positions):
    """
    Create masks showing source positions allowed to be attended by each target 
    position.
    """
    mask_shape = (1, n_positions, n_positions)
    masks = tril(np_ones(mask_shape), k=0).astype('uint8')
    return from_numpy(masks)
     # TODO: double-check my mplementation is correct, as I've implemented it in a more optimized way - but the visual output confirms as well ✓✓


##############################################################################


def scaled_dot_product_attention(queries: Type[Tensor], keys: Type[Tensor], 
    values: Type[Tensor], mask: Type[Tensor]=None, dropout_layer: \
    Type[nn.Module]=None) -> Type[Tensor]:
    """
    Return result of scaled dot-product attention operation:
    - equation: Attention(Q, K, V) = softmax(QK_T / √dk)V , with dropout applied
        only right after softmax application, during training.
    """
    # computing scores resembling each key's importance for each considered 
    # query, scaling by √dk, i.e. the square root of the feature dimension of 
    # the query vector, in order to counteract the variance increase of with 
    # the query-key dot-product, that would saturate softmax and vanish its 
    # gradient:
    scores = matmul(queries, keys.transpose(dim0=-2, dim1=-1)) \
        / sqrt(queries.size(-1)) # TODO: understand tensor dimensions
    
    # if input masked:
    if mask:
        # replacing all values of the token positions under the mask - i.e. 
        # whose values are not to be considered when composing the outputs by 
        # making a weighted average of values - with minus infinity, so as to 
        # let them completely lose their significance after softmax 
        # application (because tending to 0, i.e. the lowest probability 
        # achievable after normalization):
        scores = scores.mask_fill(mask=(mask == 0), value=-1e9)
    
    # NOTE: I believe that the scaling factor above - i.e. √dk - should be 
    # corrected as well when there is a mask

    # computing normalized attention weights, i.e. attention probabilities, 
    # of all tokens (each in a different position) toward all tokens - 
    # softmax is applied for (along) each query, to see the importance of 
    # each key (i.e. each token position) to it:
    normalized_attention_weights = F.softmax(scores, dim=-1) # TODO: explain why dim=-1 instead of -2

    # if dropout:
    if dropout_layer:
        # the function output giving normalized attention weights (attention 
        # probabilities) is substituted by an output of dropped-out attention 
        # probabilities, i.e. some probabilities are reset to 0 at random:
        normalized_attention_weights = dropout_layer()

    # computing each token output feature as a weighted average of the values 
    # of the tokens in all positions, where weights (each representing the 
    # attention of a token in a given position to a token in another given 
    # position) are normalized attention scores, i.e. attention probabilities 
    # - note how feature vectors of tokens in different positions are averaged
    # "feature-wise", in this weighted average, maintaining the representation
    # meaning of each feature and not mixing different feature values:
    return matmul(normalized_attention_weights, values), \
        normalized_attention_weights
        # returning also either the normalized attention weights or a 
        # dropped-out version of them


##############################################################################


class MultiHeadedAttention(nn.Module):
    """
    Multi-Headed Attention layer.
    """
    def __init__(self, n_attention_heads: int, 
        token_representation_dimension: int, dropout_prob: float):
        assert ((token_representation_dimension % n_attention_heads) == 0)
        super(MultiHeadedAttention, self).__init__()
        # keys and values feature dimensionality:
        self.key_or_value_dimension = \
            token_representation_dimension // n_attention_heads
        self.n_attention_heads = n_attention_heads
        # layers for linearly projecting tokens into keys, queries and values,
        # respectively - the first three ones - and for merging information
        #  from different heads - the fourth one:
        self.projection_layers = get_clones(
            module_to_be_cloned=nn.Linear(
                in_features=token_representation_dimension,
                out_features=token_representation_dimension # TODO: understand why it is not 'key_or_value_dimension'
            ),
            n_clones=4
        )
        self.normalized_attention_weights = None
        self.dropout_layer = nn.Dropout(p=dropout_prob)
        

        def forward(self, query_tokens: Type[Tensor], key_or_value_tokens: 
            Type[Tensor], mask: Type[Tensor]=None) -> Type[Tensor]:

            # if input masked:
            if mask:
                # for applying the same mask to all heads:
                mask = mask.unsqueeze(dim=1) # TODO: understand why on axis -1

            # computing queries, keys and values as linear projections of
            # tokens' own features:
            queries, keys, values = [
                layer(x) for layer, x in zip(self.projection_layers, \
                    (query_tokens, key_or_value_tokens, key_or_value_tokens))
            ]

            # computing scaled dot-product attention - separately for each 
            # head:
            x, self.normalized_attention_weights = \
                scaled_dot_product_attention(
                    queries=queries,
                    keys=keys,
                    values=values,
                    mask=mask,
                    dropout_layer=self.dropout_layer
                )

            # concatenating results from all different heads along feature 
            # dimension, after adjusting tensor shape properly:
            x = x.transpose(dim0=1, dim1=2).contiguous() \
                .view(
                    *[
                        query_tokens.dim(0),
                        -1,
                        self.n_attention_heads * self.key_or_value_dimension
                    ]
                ) # TODO: understand dimensions
            
            # final fully-connected linear combination of information from 
            # different heads:
            return self.projection_layers[-1](x)


##############################################################################


class PositionWiseFeedForward(nn.Module):
    """
    Point-wise feed-forward (fully-connected) layer:
    - parameters (weights) are shared among different positions of the same 
        layer, but not among different layers;
    - equation: FFN(x) = max(0, xW1 + b1)W2 + b2 , with dropout applied only right
        after ReLU (i.e. max(..., 0)) application, during training.
    """
    def __init__(self,
                 token_representation_dimension: int,
                 feedforward_dimension: int,
                 dropout_prob: float):
        super(PositionWiseFeedForward, self).__init__()
        self.linear_layer_1 = nn.Linear(
            in_features=token_representation_dimension,
            out_features=feedforward_dimension
        )
        self.linear_layer_2 = nn.Linear(
            in_features=feedforward_dimension,
            out_features=token_representation_dimension
        )
        self.dropout_layer = nn.Dropout(dropout_prob)

        def forward(self, x: Type[Tensor]) -> Type[Tensor]:
            return self.linear_layer_2(
                self.dropout_layer(F.relu(self.linear_layer_1(x)))
            )


##############################################################################


class Embedder(nn.Module):
    """
    Embedding layer that, besides pure embedding, additionally carries out the
    (element-wise) multiplication of the embedded feature vector by the square 
    root of the embedding dimension size.
    """
    def __init__(self,
                 vocabulary_dimension: int,
                 token_representation_dimension: int):
        super(Embedder, self).__init__()
        self.core_embedding_layer = nn.Embedding(
            num_embeddings=vocabulary_dimension,
            embedding_dim=token_representation_dimension
        )
        self.token_representation_dimension = token_representation_dimension

        def forward(self, x: Type[Tensor]) -> Type[Tensor]:
            return self.core_embedding_layer(x) * \
                sqrt(self.token_representation_dimension)


##############################################################################


class PositionalEncoding(nn.Module):
    """
    Positional encoding layer, adding position information to feature values
    of input embeddings and eventually applying dropout.
    """
    def __init__(self, token_representation_dimension: int, dropout_prob: 
        float, max_sequence_length: int):
        super(PositionalEncoding, self).__init__()
        self.dropout_layer = nn.Dropout(p=dropout_prob)

        # defining positional signals added to embeddings:

        positional_signals = torch_zeros(
            (max_sequence_length, token_representation_dimension),
            requires_grad=False
        )
        positions = torch_arange(
            start=0,
            end=max_sequence_length,
            requires_grad=False
        ).unsqueeze(dim=1)
        wave_inputs = positions * torch_exp(
            torch_arange(start=0, end=token_representation_dimension, step=2)\
                * (-log(10000.0) / token_representation_dimension)
        ) # ✓ see demonstration on my notes ▢
        # interleaving sinusoidal and cosinusoidal components along feature 
        # dimension (starting with sine):
        positional_signals[:, 0::2] = torch_sin(wave_inputs)
        positional_signals[:, 1::2] = torch_cos(wave_inputs)
        positional_signals = positional_signals.unsqueeze(dim=0)

        self.register_buffer = ('positional_signals', positional_signals) # TODO: understand if redundant with requires_grad=False

    def forward(self, x):
        return self.dropout_layer(
            x + self.positional_signals
        )


##############################################################################
##############################################################################

##############################################################################
##############################################################################


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
    ) -> Type[nn.Module]:
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
    

##############################################################################

# Notes:
# pw ff the equation is applied to each position separately and identically
# is mask pf Type[Tensor] or of type something like List[List[bool]]?


if __name__ == '__main__':

    # my_model = build_transformer_architecture(
    #     src_vocabulary=10000,
    #     tgt_vocabulary=10000
    # )
    pass
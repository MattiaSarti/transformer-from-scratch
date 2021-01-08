from copy import deepcopy
from math import sqrt
from typing import Type

from torch import nn
from torch import Tensor
from torch.nn import functional as F


class EncoderDecoder(nn.Module):
    """
    base architecture for encoder-decoder sequence-to-sequence Transformer 
    models
    """
    def __init__(
        self, encoder, decoder, src_embedder, tgt_embedder, generator
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedder = src_embedder
        self.tgt_embedder = tgt_embedder
        self.generator = generator # (not used for building the architecture)

    def forward(self, src_sequence, tgt_sequence, src_mask, tgt_mask):
        """
        process both masked source and target sequences
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


class LogSoftmaxLayer(nn.Module):
    """
    linear layer followed by log-softmax activation function
    """
    def __init__(self, token_representation_dimension: int, vocabulary_dimension: int):
        super(LogSoftmaxLayer, self).__init__()
        self.linear_layer = nn.Linear(in_features=token_representation_dimension,
            out_features=vocabulary_dimension)

    def forward(self, x: Type[Tensor]) -> Type[Tensor]:
        return F.log_softmax(self.linear_layer(x), dim=-1)


##############################################################################


def get_clones(module_to_be_cloned, n_clones):
    """
    produce 'n' "architecturally" identical layers, without shared parameters
    """
    return nn.ModuleList(
        [deepcopy(module_to_be_cloned) for _ in range(n_clones)]
    )


##############################################################################


class Encoder(nn.Module):
    """
    core encoder block
    """
    def __init__(self, base_layer, n_clones):
        super(Encoder, self).__init__()
        self.layers = clones(
            module_to_be_cloned=base_layer,
            n_clones=n_clones
        )
        self.normalization_layer = LayerNorm(layer.size)

        def forward(self, x, mask):
            for layer in self.layers:
                x = layer(x, mask)
            return self.normalization_layer(x)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x) # TODO: understand why it is not to be masked (correctly defined, though, not requiring mask during forward method call)


##############################################################################


class LayerNorm(nn.Module):
    """
    layer normalization layer
    """
    def __init__(self, feature_dimension, epsilon):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(data=torch.ones(feature_dimension))
        self.beta = nn.Parameter(data=torch.zeros(feature_dimension))
        self.epsilon = epsilon

    def forward(self, x):
       mean = x.mean(dim=-1, keepdim=True)
       std = x.std(dim=-1, keepdim=True)
       return ((x - mean) / (std + self.epsilon)) * self.alpha + self.beta


##############################################################################


class MultiHeadedAttention(nn.Module):
    pass


##############################################################################


class PositionWiseFeedForward(nn.module):
    """
    point-wise feed-forward (fully-connected) layer:
    parameters (weights) are shared among different positions of the same 
        layer, but not among different layers;
    equation: FFN(x) = max(0, xW1+b1)W2 + b2 , with dropout applied only right
        after 'max' application during training;
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

        def forward(self, x):
            return self.linear_layer_2(
                dropout_layer(F.relu(self.linear_layer_1(x)))
            )


##############################################################################


class EmbeddingLayer(nn.Module):
    """
    embedding layer that, besides pure embedding, additionally carries out the
    (element-wise) multiplication of the embedded feature vector by the square 
    root of the embedding dimension size
    """
    def __init__(self,
                 vocabulary_dimension: int,
                 token_representation_dimension: int):
        super(EmbeddingLayer, self).__init__():
        self.core_embedding_layer = nn.Embedding(
            num_embeddings=vocabulary_dimension,
            embedding_dim=token_representation_dimension
        )
        self.token_representation_dimension = token_representation_dimension

        def forward(self, x):
            return self.core_embedding_layer(x) * \
                sqrt(elf.token_representation_dimension)


##############################################################################


class PositionalEncoding(nn.Module):
    pass



##############################################################################
##############################################################################

##############################################################################
##############################################################################


def build_transformer_architecture(
    src_vocabulary: int,
    tgt_vocabulary: int,
    n_encoder_blocks: int=6,
    n_decoder_blocks: int=6,
    token_representation_dimension: int=512,
    feedforward_dimension: int=2048,
    n_attention_heads: int=8,
    dropout_prob: float=0.1
) -> :
    """
    Return a Transformer model object instantiated with the architecture 
    specified by the input hyperparameters, with newly initialized weights.
    """

    # building the architecture:

    # instantiating base layers:
    multi_headed_attention_later = MultiHeadedAttention(
        
    )
    feedforward_layer = PositionWiseFeedForward(
        
    )
    positional_encoding_layer = PositionalEncoding(

    )
    # composing base layers to build the whole model architecture:
    model = EncoderDecoder(
        encoder=Encoder(
            EncoderLayer(
                token_representation_dimension,
                deepcopy(multi_headed_attention_later),
                deepcopy(feedforward_layer),
                dropout_prob
            ),
            n_encoder_blocks
        ),
        decoder=Decoder(
            DecoderLayer(
                token_representation_dimension,
                deepcopy(multi_headed_attention_later),
                deepcopy(multi_headed_attention_later),
                deepcopy(feedforward_layer),
                dropout_prob
            ),
            n_encoder_blocks
        ),
        src_embedder=nn.Sequential(
            EmbeddingLayer(
                token_representation_dimension,
                src_vocabulary
            ),
            deepcopy(positional_encoding_layer)
        ),
        tgt_embedder=nn.Sequential(
            EmbeddingLayer(
                token_representation_dimension,
                tgt_vocabulary
            ),
            deepcopy(positional_encoding_layer)
        ),
        generator=Generator(
            token_representation_dimension,
            tgt_vocabulary
        )
    )

    # initializing the hyperparameters:

    # for each layer's parameter set:
    for parameter in model.parameters:
        # TODO: explain why:
        if parameter.dim() > 1:
            # parameters initialized following Xavier initialization:
            nn.init.xavier_uniform(parameter)

    return model
    

##############################################################################

# Notes:
# pw ff the equation is applied to each position separately and identically


if __name__ == '__main__':

    my_model make_model(
        src_vocabulary=10000,
        tgt_vocabulary=10000
    )
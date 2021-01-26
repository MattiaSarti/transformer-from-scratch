from .base import *


def allowed_positions_to_attend(n_positions: int) -> Tensor:
    """
    Create masks showing source positions allowed to be attended by each target 
    position.
    """
    mask_shape = (1, n_positions, n_positions)
    masks = tril(np_ones(mask_shape), k=0).astype('uint8')
    return from_numpy(masks)
     # TODO: double-check my mplementation is correct, as I've implemented it in a more optimized way - but the visual output confirms as well ✓✓


def scaled_dot_product_attention(queries: Tensor, keys: Tensor, 
    values: Tensor, mask: Tensor=None, dropout_layer: \
    nn.Module=None) -> Tensor:
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
        

        def forward(self, query_tokens: Tensor, key_or_value_tokens: Tensor,
            mask: Tensor=None) -> Tensor:

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
import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size: int):
        """
        Initializes the InputEmbedding module with the given d_model and vocab_size.
        
        Parameters:
            d_model (int): The dimension of the model.
            vocab_size (int): The size of the vocabulary.
        """
        super().__init__()
        self.d_model = d_model
        self.vocabsize = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass of the InputEmbedding module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The embedded and scaled output tensor.
        """
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, sequence_length: int, dropout: float)-> None:
        """
        Initializes the PositionalEncoding module with the given d_model, sequence_length, and dropout.
        
        Parameters:
            d_model (int): The dimension of the model.
            sequence_length (int): The length of the input sequence.
            dropout (float): The dropout rate.
        """
        super().__init__()
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(dropout)

        #creating matrix of shape (sequence_length, d_model)
        pe = torch.zeros(sequence_length, d_model)
        #creating a vector of shape (sequence_length, 1) with values from 0 to sequence_length
        position = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)
        denom_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #Applying the sine to even positions and cosine to odd positions
        pe[:,0::2] = torch.sin(position * denom_term)
        pe[:,1::2] = torch.cos(position * denom_term)

        pe = pe.unsqueeze(0) # adding a batch dimension (1, sequence_length, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying positional encoding.
        """
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        """
        Initializes the LayerNormalization module with the given epsilon value, alpha as a multiplier, and bias as an adder.
        
        Parameters:
            eps (float): The epsilon value for numerical stability.

        Returns:
            None
        """
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplier
        self.bias = nn.Parameter(torch.zeros(1)) # Adder

    def forward(self,x):
        """
        A function that computes the layer normalization of the input tensor x.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The normalized output tensor.
        """
        mean = x.mean(dim =-1, keepdim=True)
        std = x.std(dim =-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float ) -> None:
        """
        Initializes the FeedForwardBlock module with the given d_model, d_ff, and dropout values.
        
        Parameters:
            d_model (int): The dimension of the model.
            d_ff (int): The dimension of the feedforward layer.
            dropout (float): The dropout rate.
        
        Returns:
            None
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # First linear layer with d_model inputs and d_ff outputs, w1 and b1 
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # Second linear layer with d_ff inputs and d_model outputs, w2 and b2

    def forward(self, x):
        """
        Applies forward pass through the feedforward block.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the feedforward block.
        """
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        """
        Initializes the MultiHeadAttentionBlock module with the given d_model, n_heads, and dropout values.

        Args:
            d_model (int): The dimension of the model.
            n_heads (int): The number of attention heads.
            dropout (float): The dropout rate.

        Raises:
            AssertionError: If the number of heads is not divisible by the d_model.

        Returns:
            None
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, "Number of heads is not divisible by d_model"
        self.d_k = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) # wk
        self.w_v = nn.Linear(d_model, d_model) # wv
        self.w_o = nn.Linear(d_model, d_model) # wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(self, key, value, query, mask, dropout):
        """
        Compute the attention scores between the query, key, and value tensors.

        Args:
            key (torch.Tensor): The key tensor of shape (batch_size, seq_len, d_model).
            value (torch.Tensor): The value tensor of shape (batch_size, seq_len, d_model).
            query (torch.Tensor): The query tensor of shape (batch_size, seq_len, d_model).
            dropout (float, optional): The dropout rate. Defaults to None.

        Returns:
            torch.Tensor: The attention scores tensor of shape (batch_size, n_heads, seq_len, seq_len).
        """
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask=None):
        """
        Applies the forward pass of the MultiHeadAttentionBlock module.

        Args:
            q (torch.Tensor): The query tensor of shape (batch_size, seq_len, d_model).
            k (torch.Tensor): The key tensor of shape (batch_size, seq_len, d_model).
            v (torch.Tensor): The value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): The mask tensor of shape (batch_size, seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        query, key, value = self.w_q(q), self.w_k(k), self.w_v(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.n_heads, self.d_k).transpose(1,2) 
        key = key.view(key.shape[0], key.shape[1], self.n_heads, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.n_heads, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_heads * self.d_k)

        return self.w_o(x)
    
class ResidualConnectionLayer(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        """
        Initializes the ResidualConnectionLayer module with the given d_model and dropout values.

        Args:
            d_model (int): The dimension of the model.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super().__init__()
        self.norm = LayerNormalization()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Applies a residual connection to the input tensor `x` by adding the output of a sublayer to the input tensor.

        Parameters:
            x (torch.Tensor): The input tensor to apply the residual connection to.
            sublayer (torch.nn.Module): The sublayer to apply to the input tensor.

        Returns:
            torch.Tensor: The output tensor after applying the residual connection.
        """
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attn_block: MultiHeadAttentionBlock , feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Initializes the EncoderBlock module with the given self_attn_block, feed_forward_block, and dropout values.

        Args:
            self_attn_block (MultiHeadAttentionBlock): The multi-head attention block to be used in the encoder.
            feed_forward_block (FeedForwardBlock): The feed-forward block to be used in the encoder.
            dropout (float): The dropout rate to be applied in the encoder.

        Returns:
            None
        """
        super().__init__()
        self.self_attn_block = self_attn_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnectionLayer(dropout) for _ in range(2)])

    def forward(self, x, mask):
        """
        Applies the forward pass of the EncoderBlock module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): The mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model).
        """
        x = self.residual_connection[0](x, lambda x: self.self_attn_block(x, x, x, mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        """
        Initializes the Encoder module with the given layers and applies layer normalization.

        Args:
            layers (nn.ModuleList): A list of layers to be applied in the encoder.

        Returns:
            None
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        """
        Applies a sequence of layers to the input tensor `x` with the given mask.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor): The mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: The output tensor after applying all layers and applying layer normalization.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attn_block: MultiHeadAttentionBlock, src_attn_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Initializes the DecoderBlock module with the given self_attn_block, src_attn_block, feed_forward_block, and dropout values.

        Args:
            self_attn_block (MultiHeadAttentionBlock): The multi-head attention block for self-attention.
            src_attn_block (MultiHeadAttentionBlock): The multi-head attention block for source/cross attention.
            feed_forward_block (FeedForwardBlock): The feed-forward block for the decoder.
            dropout (float): The dropout rate to be applied in the decoder.

        Returns:
            None
        """
        super().__init__()
        self.self_attn_block = self_attn_block
        self.src_attn_block = src_attn_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnectionLayer(dropout) for _ in range(3)])

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        """
        Applies the forward pass of the DecoderBlock module.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            encoder_out (torch.Tensor): The output tensor from the encoder of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): The mask tensor for the source/cross attention of shape (batch_size, seq_len, seq_len).
            tgt_mask (torch.Tensor): The mask tensor for the self-attention of shape (batch_size, seq_len, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model) after applying the decoder block.
        """
        x = self.residual_connection[0](x, lambda x: self.self_attn_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.src_attn_block(x, encoder_out, encoder_out, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:    
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        Initializes the ProjectionLayer module with the given d_model and vocab_size.

        Args:
            d_model (int): The dimension of the model.
            vocab_size (int): The size of the vocabulary.

        Returns:
            None
        """
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Apply the forward pass of the ProjectionLayer module to the input tensor `x`.

        Parameters:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, vocab_size) after applying the projection layer and applying the log softmax function along the last dimension.
        """
        return torch.log_softmax(self.projection(x), dim=-1)
    
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_embed: PositionalEncoding, tgt_pos: PositionalEncoding ,projection_layer: ProjectionLayer) -> None:
        """
        Initializes the Transformer module with the given encoder, decoder, and projection_layer.

        Args:
            encoder (Encoder): The encoder module.
            decoder (Decoder): The decoder module.
            src_embed (InputEmbedding): The input embedding module for the source sequence.
            src_pos (PositionalEncoding): The positional encoding module for the source sequence.
            tgt_embed (PositionalEncoding): The positional encoding module for the target sequence.
            tgt_pos (PositionalEncoding): The positional encoding module for the target sequence.
            projection_layer (ProjectionLayer): The projection layer module.

        Returns:
            None
        """
        
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.tgt_embed = tgt_embed
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        """
        Encodes the source sequence using the encoder module.

        Args:
            src (torch.Tensor): The input tensor of shape (batch_size, seq_len) representing the source sequence.
            src_mask (torch.Tensor): The mask tensor of shape (batch_size, seq_len) representing the source sequence mask.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model) after encoding the source sequence.
        """
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(self.src_pos(src), src_mask)

    def decode(self, tgt, encoder_out, src_mask, tgt_mask):
        """
        Decodes the target sequence using the decoder module.

        Args:
            tgt (torch.Tensor): The input tensor of shape (batch_size, seq_len) representing the target sequence.
            encoder_out (torch.Tensor): The output tensor from the encoder module of shape (batch_size, seq_len, d_model).
            src_mask (torch.Tensor): The mask tensor for the source sequence of shape (batch_size, seq_len).
            tgt_mask (torch.Tensor): The mask tensor for the target sequence of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, d_model) after decoding the target sequence.
        """
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)

    def project(self, x):
        """
        Projects the input tensor `x` using the projection layer.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, seq_len, vocab_size) after projecting the input tensor.
        """
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int =512, N: int =6, heads: int =8, dropout: float =0.1, d_ff: int =2048  ) -> Transformer:
    """
    Builds a Transformer model for sequence-to-sequence translation.

    Args:
        src_vocab_size (int): The size of the source vocabulary.
        tgt_vocab_size (int): The size of the target vocabulary.
        src_seq_len (int): The maximum length of the source sequence.
        tgt_seq_len (int): The maximum length of the target sequence.
        d_model (int, optional): The dimensionality of the input embeddings and the transformer model. Defaults to 512.
        N (int, optional): The number of encoder and decoder blocks. Defaults to 6.
        heads (int, optional): The number of attention heads. Defaults to 8.
        dropout (float, optional): The dropout rate. Defaults to 0.1.
        d_ff (int, optional): The dimensionality of the feed-forward layer. Defaults to 2048.

    Returns:
        Transformer: The built Transformer model.

    Description:
        This function builds a Transformer model for sequence-to-sequence translation. It creates the embedding layers, positional encodings, encoder blocks, decoder blocks, encoder, decoder, and projection layer. The model is then initialized with Xavier uniform initialization.

        The embedding layers are created using the InputEmbedding and PositionalEncoding classes.

        The positional encodings are created using the PositionalEncoding class.

        The encoder blocks are created using the MultiHeadAttentionBlock, FeedForwardBlock, and EncoderBlock classes.

        The decoder blocks are created using the MultiHeadAttentionBlock, MultiHeadAttentionBlock, FeedForwardBlock, and DecoderBlock classes.

        The encoder and decoder are created using the Encoder and Decoder classes.

        The projection layer is created using the ProjectionLayer class.

        The Transformer model is created using the Transformer class.

        The model is then initialized with Xavier uniform initialization.

        Finally, the built Transformer model is returned.

    """
    
    # create the embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = PositionalEncoding(d_model, tgt_vocab_size)

    # create the positional encodings
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attn_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attn_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)        
    
    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attn_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        decoder_cross_attn_block = MultiHeadAttentionBlock(d_model, heads, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attn_block, decoder_cross_attn_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # create the transformer
    transformer = Transformer(encoder, decoder, src_embed, src_pos, tgt_embed, tgt_pos, projection_layer)

    #initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
from multi_head_attention.multi_head_attention import (
    MultiHeadAttention
)
from feed_forward_network.feed_forward import FeedForward
from layer_normalization.layernorm import LayerNorm
from residual_connection.residual_connection import ResidualConnection


class TransformerBlock:
    
    def __init__(self, embed_dim, num_heads):
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.multihead_att = MultiHeadAttention(self.embed_dim, self.num_heads)
        
        self.resid1 = ResidualConnection()
        
        self.layern1 = LayerNorm(self.embed_dim)
        
        self.feedf = FeedForward(self.embed_dim)
        
        self.resid2 = ResidualConnection()
        
        self.layern2 = LayerNorm(self.embed_dim)
    
    def forward(self, embeddings):
            
        attention_output = self.multihead_att.forward(embeddings)
        
        residual_output = self.resid1.forward(embeddings, attention_output)
        
        normalized = []
        
        for token in residual_output:
            
            normalized.append(self.layern1.forward(token))
        
        feedforward_output = []
        
        for token in normalized:
            
            feedforward_output.append(self.feedf.forward(token))
        
        second_residual_output = self.resid2.forward(normalized, feedforward_output)
        
        final_ouput = []
        
        for token in second_residual_output:
            
            final_ouput.append(self.layern2.forward(token))
            
        return final_ouput
from self_attention.self_attention import SelfAttention
from self_attention.linear_layer import Linear

class MultiHeadAttention:
    
    def __init__(self, embed_dim, num_heads):
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads"
            )
        self.head_dim = embed_dim // num_heads
            
        self.heads = []
        for _ in range(num_heads):
            self.heads.append(
                SelfAttention(self.head_dim)
            )
        
        self.output = Linear(embed_dim, embed_dim)
            
    def _split_heads_v1(self, embedding):
        heads = []
        row = []
        
        for value in embedding:
            row.append(value)
            if len(row) == self.head_dim:
                heads.append(row)
                row = []
        
        return heads

    def _split_heads_v2(self, embeddings):

        # Create one bucket for each head
        heads = []

        for _ in range(self.num_heads):
            heads.append([])

        # process every embedding
        for embedding in embeddings:

            pieces = []
            row = []

            # split one embedding into pieces
            for value in embedding:

                row.append(value)

                if len(row) == self.head_dim:
                    pieces.append(row)
                    row = []

            # put each piece into its corresponding head
            for i in range(self.num_heads):
                heads[i].append(pieces[i])

        return heads
                
    def _concatenate_heads(self, head_outputs):
        
        concatenated_head = []
        for head in head_outputs:
            for value in head:
                concatenated_head.append(value)
            
        return concatenated_head
    
    def _concatenate_heads_v2(self, head_outputs):
        concatenated = []
        
        for i in range(len(head_outputs[0])):
            token_output = []
            
            for head in head_outputs:
                token = head[i]
                for value in token:
                    token_output.append(value)        
            concatenated.append(token_output)
        
        return concatenated
    
    def forward(self, embeddings):
        
        split_heads = self._split_heads_v2(embeddings)
        
        head_outputs = []
        for head, head_embedding in zip(self.heads, split_heads):
            output = head.forward(head_embedding)
            head_outputs.append(output)
        
        concatenated = self._concatenate_heads_v2(head_outputs)
        
        final_output = []
        
        for token in concatenated:
            final_output.append(self.output.forward(token))
        
        return final_output
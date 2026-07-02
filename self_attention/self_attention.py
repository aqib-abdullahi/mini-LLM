import math
from linear_layer import Linear


class SelfAttention:
    
    def __init__(self, embed_dim):
        self.embed_dim = self.embed_dim
        
        self.query = Linear(embed_dim, embed_dim)
        self.key = Linear(embed_dim, embed_dim)
        self.value = Linear(embed_dim, embed_dim)
    
    def dot_product(self, a, b):
        product = 0
        
        for i in range(len(a)):
            product += a[i] * b[i]
        
        return product
    
    def softmax(self, scores):
        largest = max(scores)
        
        stable = []
        
        for score in scores:
            stable.append(score - largest)
        
        exp_scores = []
        total = 0
        
        for score in stable:
            value = math.exp(score)
            exp_scores.append(value)
            total += value
        
        probabilities = []
        
        for value in exp_scores:
            probabilities.append(value / total)
        
        return probabilities

    def _compute_attention_scores(self, queries, keys):
        attention_scores = []
        
        for query in queries:
            row = []
            
            for key in keys:
                row.append(self.dot_product(query, key))
            
            row = self.softmax(row)
            
            attention_scores.append(row)
        
        return attention_scores

    def _weighted_sum(Self, attention_scores, values):
        outputs = []
        
        for row in attention_scores:
            
            output_vector = [0] * len(values[0])
            
            for weight, value_vector in zip(row, values):
                
                for i in range(len(value_vector)):
                    output_vector[i] += weight * value_vector[i]
            
            outputs.append(output_vector)
        
        return outputs

    def forward(self, embeddings):
        
        queries = []
        keys = []
        values = []
        
        for embedding in embeddings:
            queries.append(self.query.forward(embedding))
            keys.append(self.key.forward(embedding))
            values.append(self.value.forward(embedding))
            
        attention_scores = self._compute_attention_scores(queries, keys)
        
        outputs = self._weighted_sum(attention_scores, values)
        
        return outputs
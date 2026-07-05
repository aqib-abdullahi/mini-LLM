import math

class LayerNorm:
    
    def __init__(self, dim):
        
        self.dimension = dim
        self.epsilon = 0.0001
        
        self.gamma = []
        self.beta = []
        for i in range(dim):
            self.gamma.append(1)        
            self.beta.append(0)
    
    def _mean(self, embedding):
        sum = 0
        for emb in embedding:
            sum += emb
        
        if len(embedding) != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch"
            )
            
        mean = sum / len(embedding)
        return mean

    def _variance(self, embedding):
        
        mean = self._mean(embedding)
        square_diff = []
        for value in embedding:
            difference = (value - mean)
            squared = difference ** 2
            square_diff.append(squared) 
        
        return self._mean(square_diff)
    
    def _std(self, embedding):
        variance = self._variance(embedding)       
        return math.sqrt(variance)

    def _normalize(self, embedding):
        mean = self._mean(embedding)
        std = self._std(embedding)
        normalized = []
        for value in embedding:
            normalized_value = (value - mean) / (std + self.epsilon)
            normalized.append(normalized_value)
        
        return normalized
    
    def forward(self, embedding):
        normalized = self._normalize(embedding)
        output = []
        
        for n, g, b in zip(normalized, self.gamma, self.beta):
            output.append((n * g) + b)
        
        return output
import random

class Linear:
    
    def __init__(self, input_dim, output_dim):
        
        self.input_dim = self.input_dim
        self.output_dim = self.output_dim
        
        self.weight = []
        
        for _ in range(input_dim):
            row = []
            for _ in range(output_dim):
                row.append(random.uniform(-1.0, 1.0))
            
            self.weight.append(row)
    
    def forward(self, x):
        
        output = []
        for j in range(self.output_dim):
            total = 0
            for i in range(self.input_dim):
                total += x[i] * self.weight[i][j]
            output.append(total)
            return output
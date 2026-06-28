

class CharacterTokenizer:
    
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0
    
    def train(self, text):
        self.vocab = sorted(set(text))
        self.stoi = {"<UNK>": 0}
        
        idx = 1
        for char in self.vocab:
            self.stoi[char] = idx
            idx += 1
        
        self.itos = {}    
        for char, idx in self.stoi.items():
            self.itos[idx] = char
        
        self.vocab_size = len(self.vocab)
    
    def encode(self, text):
        ids = []
        for char in text:
            if char in self.stoi:            
                ids.append(self.stoi[char])
            else:
                ids.append(self.stoi["<UNK>"])
        
        return ids 
    
    def decode(self, ids):
        chars = []
        
        for idx in ids:
            chars.append(self.itos[idx])
        
        return "".join(chars)


            
class BPETokenizer:
    
    def __init__(self):
        self.merges = []
        
        self.vocab =set()
        self.stoi = {}
        self.itos = {}
        
        self.vocab_size = 0
    
    
    def train(self, text, num_merges):
        
        #starts with character tokens
        tokens = list(text)
        
        #initial vocabulalry
        self.vocab = set(tokens)
        
        #learns merge rules
        for _ in range(num_merges):
            counts = self.get_pair_counts(tokens)
            if not counts:
                break            
            best_pair = self.get_best_pair(counts)
            self.merges.append(best_pair)
            
            #add newly created token to vocabulary
            merged_token = best_pair[0] + best_pair[1]
            self.vocab.add(merged_token)
            
            tokens = self.merge_pair(tokens, best_pair)
        
            print("Best Pair:", best_pair)
            print("Tokens:", tokens)
        
        #building vocabulary mappings
        sorted_vocab = sorted(self.vocab)
        
        self.stoi = {
            "<UNK>": 0
        }
        
        idx = 1
        
        for token in sorted_vocab:
            self.stoi[token] = idx
            idx += 1
        
        self.itos = {}
        
        for token, idx in self.stoi.items():
            self.itos[idx] = token
        
        self.vocab_size = len(self.stoi)
    
    def encode(self, text):
        tokens = list(text)
        
        #replay merges learned
        for pair in self.merges:
            tokens = self.merge_pair(tokens, pair)
        
        ids = []
        
        for token in tokens:
            ids.append(
                self.stoi.get(token, self.stoi["<UNK>"])
            )
        
        return ids

    def decode(self, ids):
        tokens = []
        
        for idx in ids:
            tokens.append(self.itos.get(idx, "<UNK>"))
        
        return "".join(tokens)
    
    def _get_pair_counts(self, tokens):
        counts = {}
        
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            
            counts[pair] = counts.get(pair, 0) + 1
        
        return counts

    def _merge_pair(self, tokens, pair):
        merged_tokens = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and
                (tokens[i], tokens[i + 1]) == pair
            ):
                merged_tokens.append(
                    tokens[i] + tokens[i + 1]
                )
                
                i += 2
                
            else:
                merged_tokens.append(tokens[i])
                
                i += 1
        
        return merged_tokens


    def _get_best_pair(self, counts):
        
        return max(counts, key=counts.get)

            
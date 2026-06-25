

            
class BPETokenizer:
    
    def __init__(self):
        self.merges = []
    
    
    def train(self, text, num_merges):
        tokens = list(text)
        
        for _ in range(num_merges):
            counts = self.get_pair_counts(tokens)
            if not counts:
                break            
            best_pair = self.get_best_pair(counts)
            self.merges.append(best_pair)
            
            tokens = self.merge_pair(tokens, best_pair)
        
            print("Best Pair:", best_pair)
            print("Tokens:", tokens)
        
        return tokens
    
    def get_pair_counts(self, tokens):
        counts = {}
        
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            
            counts[pair] = counts.get(pair, 0) + 1
        
        return counts

    def merge_pair(self, tokens, pair):
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


    def get_best_pair(self, counts):
        
        return max(counts, key=counts.get)

            
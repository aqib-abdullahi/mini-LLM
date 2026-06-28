

            
class BPETokenizer:
    
    def __init__(self):
        self.merges = {}
        
        self.vocab =set()
        self.stoi = {}
        self.itos = {}
        
        self.vocab_size = 0
    
    
    def train(self, corpus, vocab_size):
        # reset
        self.vocab.clear()
        self.merges.clear()
        self.stoi.clear()
        self.itos.clear()
        self.vocab_size = 0
        
        #starts with character tokens
        tokenized_words = [list(word) for word in corpus.split()]
        
        #initial vocabulalry
        
        for word in tokenized_words:
            for token in word:
                self.vocab.add(token)
        
        # check
        if vocab_size <= len(self.vocab):
            raise ValueError(
                "Vocab_size must be larger than number of unique initial tokens"
            )
        
        #learns until vocabulary size
        while len(self.vocab) < vocab_size:
            counts = self._get_pair_counts(tokenized_words)
            if not counts:
                break            
            best_pair = self._get_best_pair(counts)
            
            self._learn_merge(best_pair)
            
            tokenized_words = self._merge_pair(tokenized_words, best_pair)
    
        self._build_vocab()
    
    def _build_vocab(self):
        
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
            tokens = self._merge_word(tokens, pair)
        
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
    
    def _learn_merge(self, best_pair):
        merged_token = "".join(best_pair)
        self.merges[best_pair] = merged_token
        
        self.vocab.add(merged_token)
        
        return merged_token
    
    def _get_pair_counts(self, tokenized_words):
        counts = {}
        
        for word in tokenized_words:
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                
                counts[pair] = counts.get(pair, 0) + 1
        
        return counts

    def _merge_pair(self, tokenized_words, pair):
        merged_words = []
        
        for word in tokenized_words:
            merged_words.append(
                self._merge_word(word, pair)
            )
    
        return merged_words

    def _merge_word(self, tokens, pair):
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

            
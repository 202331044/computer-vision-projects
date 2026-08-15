from collections import Counter

class Tokenizer:
    def __init__(self, datasets, min_freq):
        self.min_freq = min_freq
        self.idx2word = {0: "<PAD>", 
                         1: "<UNK>",
                         2: "<CLS>"}
        
        self.word2idx = {"<PAD>": 0,
                         "<UNK>": 1,
                         "<CLS>": 2}
        
        self.build_vocab(datasets)
    
    def build_vocab(self, datasets):
        counter = Counter()

        for text, _ in datasets:
            tokens = text.lower().split()
            counter.update(tokens)
        
        for token, freq in counter.items():
            if freq >= self.min_freq:
                idx = len(self.idx2word)

                self.idx2word[idx] = token
                self.word2idx[token] = idx

    def tokenize(self, text):
        return text.lower().split()
    
    def encode(self, text):
        tokens = self.tokenize(text)

        token_ids = [
                    self.word2idx.get(token, self.word2idx["<UNK>"])
                    for token in tokens
                    ]

        return [self.word2idx["<CLS>"]] + token_ids    
    
    def decode(self, token_ids):
        return [self.idx2word[token_id]
                for token_id in token_ids]
    
import collections
import random
import ssl

import nltk
import torch
from torch.utils.data import Dataset
from torchtext import vocab as V
from tqdm import tqdm

try:
    from nltk import tokenize
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')

class EmbeddingDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.data = []
        for i in tqdm(range(len(X)), desc="Building dataset..."):
            self.data.append((X[i], Y[i]))
        random.shuffle(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def build_vocab(text, min_freq=1, vocab_size=5000):
    print('Building vocab...')
    tokens = tokenize.word_tokenize(text)
    cleaned_tokens = [token.lower() for token in tokens if token.isalpha()]
    counter = collections.Counter(cleaned_tokens)
    vocab = V.vocab(collections.Counter(dict(counter.most_common(
        vocab_size))), min_freq=min_freq, specials=["<unk>"])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def to_skip_gram(sent, window_size=2):
    res = []
    for i, x in enumerate(sent):
        for j in range(max(0, i-window_size), min(i+window_size+1, len(sent))):
            if i != j:
                res.append([x, sent[j]])
    return res


def encode(sent, vocab):
    encoded_token = []
    for token in tokenize.word_tokenize(sent):
        encoded_token.append(vocab[token.lower()])
    return encoded_token


def get_dataset(text, ds_len, vocab, window_size):
    X = []
    Y = []
    for _, sent in zip(range(ds_len), text):
        for w1, w2 in to_skip_gram(encode(sent, vocab), window_size):
            X.append(w1)
            Y.append(w2)

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    ds = EmbeddingDataset(X, Y)
    return ds

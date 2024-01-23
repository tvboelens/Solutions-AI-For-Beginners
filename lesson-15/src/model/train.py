import ssl
import os
import collections
import random
import torch
from torch.utils.data import random_split, DataLoader, Dataset,IterableDataset
from tqdm import tqdm
import nltk
from torchtext.data.utils import get_tokenizer
from torchtext import vocab as V
import yaml
import ssl

import time


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
from nltk import tokenize

CONFIG_PATH = 'src/config/'


class SimpleIterableDataset(IterableDataset):
    def __init__(self, X, Y):
        super().__init__()
        self.data = []
        for i in tqdm(range(len(X)), desc="Building dataset..."):
            self.data.append((X[i], Y[i]))
        random.shuffle(self.data)

    def __iter__(self):
        return iter(self.data)
    
    def __len__(self):
        return len(self.data)


class EmbeddingDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.data = []
        for i in tqdm(range(len(X)), desc="Building dataset..."):
            self.data.append((Y[i], X[i]))
        random.shuffle(self.data)

    def __getitem__(self,idx):
        return self.data[idx]


    def __len__(self):
        return len(self.data)

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def build_vocab(text, min_freq=1, vocab_size=5000):
    print('Building vocab...')
    tokens = tokenize.word_tokenize(text)
    cleaned_tokens = [token.lower() for token in tokens if token.isalpha()]
    counter = collections.Counter(cleaned_tokens)
    vocab = V.vocab(collections.Counter(dict(counter.most_common(vocab_size))), min_freq=min_freq, specials=["<unk>"])
    vocab.set_default_index(vocab['<unk>'])
    print('Vocab complete')
    return vocab


def to_skip_gram(sent, window_size=2):
    res = []
    for i, x in enumerate(sent):
        for j in range(max(0, i-window_size), min(i+window_size+1, len(sent))):
            if i != j:
                res.append([x,sent[j]])
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
    ds = SimpleIterableDataset(X,Y)
    return ds




def main(config):
    text = open('data/shakespeare.txt', 'rb').read().decode(encoding='utf-8').replace('\n',' ')
    vocab = build_vocab(text)
    text = tokenize.sent_tokenize(text)
    ds = get_dataset(text, ds_len=config["ds_len"],vocab=vocab, window_size=config["window_size"])
    starttime = time.time()
    print(f"Splitting dataset and creating dataloaders")
    train_set, test_set = random_split(
        ds, [config["ds_split"], 1-config["ds_split"]])
    train_loader = DataLoader(train_set, batch_size=config["bs"])
    test_loader = DataLoader(test_set, batch_size=config["bs"])
    print(f"Complete, time={round(time.time()-starttime,2)} seconds")

    





if __name__ == "__main__":
    if not os.path.exists('data/shakespeare.txt'):
        if not os.path.exists('data'):
            os.mkdir('data')
        os.system(
            'wget -O data/shakespeare.txt https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    config = load_config('config.yaml')
    main(config)
    


import ssl
import os
import collections
#import torch
import nltk
from torchtext.data.utils import get_tokenizer
from torchtext import vocab as V
import yaml
import ssl


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')
from nltk import tokenize

CONFIG_PATH = 'src/config/'

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def build_vocab(text, min_freq=1, vocab_size=5000):
    print('Building vocab...')
    tokens = tokenize.word_tokenize(text)
    cleaned_tokens = [token.lower() for token in tokens if token.isalpha()]
    counter = collections.Counter(cleaned_tokens)
    print(counter.most_common(vocab_size)[:100])
    vocab = V.vocab(collections.Counter(dict(counter.most_common(vocab_size))), min_freq=min_freq)
    return vocab


def to_skip_gram(sent, window_size=2):
    res = []
    for i, x in enumerate(sent):
        for j in range(max(0, i-window_size), min(i+window_size+1, len(sent))):
            if i != j:
                res.append([x,sent[j]])
    return res


def encode(x, vocabulary, tokenizer=get_tokenizer('basic_english')):
    return [vocabulary[s] for s in tokenizer(x)]



def main(config):
    text = open('data/shakespeare.txt', 'rb').read().decode(encoding='utf-8').replace('\n',' ')
    vocab = build_vocab(text)
    text = tokenize.sent_tokenize(text)





if __name__ == "__main__":
    if not os.path.exists('data/shakespeare.txt'):
        if not os.path.exists('data'):
            os.mkdir('data')
        os.system(
            'wget -O data/shakespeare.txt https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    config = load_config('config.yaml')
    main(config)
    


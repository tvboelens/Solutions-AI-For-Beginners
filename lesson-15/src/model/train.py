import os
import collections
#import torch
from torchtext.data.utils import get_tokenizer
from torchtext import vocab as V
import yaml

CONFIG_PATH = 'src/config/'


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def build_vocab(text, min_freq=1, vocab_size=5000):
    print('Building vocab...')
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(text)
    counter = collections.Counter(tokens)
    print(counter.most_common(vocab_size)[:10])
    vocab = V.vocab(collections.Counter(dict(counter.most_common(vocab_size))), min_freq=min_freq)
    return vocab



def main(config):
    text = open('data/shakespeare.txt', 'rb').read().decode(encoding='utf-8')
    vocab = build_vocab(text)




if __name__ == "__main__":
    if not os.path.exists('data/shakespeare.txt'):
        if not os.path.exists('data'):
            os.mkdir('data')
        os.system(
            'wget -O data/shakespeare.txt https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    config = load_config('config.yaml')
    main(config)
    


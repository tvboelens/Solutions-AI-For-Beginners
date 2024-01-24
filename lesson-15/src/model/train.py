from statistics import mean
import ssl
import os
import collections
import random
import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader, Dataset,IterableDataset
from tqdm import tqdm
import nltk
#from torchtext.data.utils import get_tokenizer
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
            self.data.append((X[i], Y[i]))
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
    ds = EmbeddingDataset(X,Y)
    return ds


def train_epoch(epoch, model, dataloader, optimizer, report_freq, device, loss_fn):
    epoch_loss = []
    running_loss = 0.0
    i=0
    starttime = time.time()
    with tqdm(dataloader, unit='batch') as tepoch:
        tepoch.set_description(f"Training for epoch {epoch+1}...")
        for features, targets in tepoch:
            features.to(device)
            targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(features)
            batch_loss = loss_fn(pred, targets)
            batch_loss.backward()
            optimizer.step()
            epoch_loss.append(batch_loss.item())
            running_loss += batch_loss.item()
            if i == report_freq:
                tepoch.set_postfix(loss=running_loss/report_freq)
                time.sleep(0.1)
                """ print(f"Batch {j+1} finished, "
                    f"time = {int((time.time()-starttime)/60)} minutes {round((time.time()-starttime)%60,2)} seconds, "
                    f"loss: {running_loss/report_freq}")
                starttime = time.time() """
                running_loss = 0.0
                i=0
            else:
                i+=1
    return epoch_loss


def train_model(model, train_loader, test_loader, optimizer, no_of_epochs, report_freq,
                device='cpu', loss_fn=nn.CrossEntropyLoss()):
    train_loss = []
    test_loss = []
    for epoch in range(no_of_epochs):
        #epoch_starttime = time.time()
        #print(f"START TRAINING FOR EPOCH {epoch + 1}:")
        model.train(True)
        epoch_loss = train_epoch(
            epoch, model, train_loader, optimizer, report_freq, device, loss_fn)
        train_loss += epoch_loss

        running_tloss = 0.0
        i=0
        model.eval()
        """ print(f"Training for epoch {epoch+1} done, time = "
              f"{int((time.time()-epoch_starttime)/60)} minutes {round((time.time()-epoch_starttime)%60,2)} seconds") """
        with torch.no_grad():
            tbatch_starttime = time.time()
            with tqdm(test_loader, unit='batch') as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}, validating model...")
                for tfeatures, ttargets in tepoch:
                    tfeatures.to(device)
                    ttargets.to(device)
                    pred = model(tfeatures)
                    tloss = loss_fn(pred, ttargets)
                    test_loss.append(tloss.item())
                    running_tloss += tloss.item()
                    tepoch.set_postfix(loss=running_tloss/(i+1))
                    time.sleep(0.1)
                    i+=1

                # print(f"Completed validation for batch {i+1}, time = "
                #      f"{int((time.time()-vbatch_starttime)/60)} minutes {round((time.time()-vbatch_starttime)%60,2)}"
                #      f"seconds")

        test_loss.append(running_tloss/(i+1))
        train_loss += epoch_loss
        """ print(f"Validation for epoch {epoch+1} done, time = "
              f"{int((time.time()-epoch_starttime)/60)} minutes {round((time.time()-epoch_starttime)%60,2)} seconds, "
              f"LOSS train {epoch_loss[-1]}, val: {test_loss[-1]}") """

    return train_loss, test_loss


def save_model(model, savetime):
    model_fn = 'word2vec_skip_gram_model_'+savetime+'.pth'
    if not os.path.exists('output/models/'):
        if not os.path.exists('output/'):
            os.makedir('output')
        os.makedir('output/models')

    model_fp = 'output/models/'+model_fn
    torch.save(model, model_fp)







def main(config):
    #Build dataset and dataloaders
    text = open('data/shakespeare.txt', 'rb').read().decode(encoding='utf-8').replace('\n',' ')
    vocab = build_vocab(text)
    text = tokenize.sent_tokenize(text)
    ds = get_dataset(text, ds_len=config["ds_len"],vocab=vocab, window_size=config["window_size"])
    starttime = time.time()
    print(f"Splitting dataset and creating dataloaders...")
    train_set, test_set = random_split(
        ds, [config["ds_split"], 1-config["ds_split"]])
    train_loader = DataLoader(train_set, batch_size=config["bs"])
    test_loader = DataLoader(test_set, batch_size=config["bs"])
    
    #Define model, optimizers and loss function
    vocab_size = len(vocab)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    embedder = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=config["embedding_dim"])
    model = torch.nn.Sequential(embedder,
    torch.nn.Linear(in_features=config["embedding_dim"], out_features=vocab_size))
    model.to(device)
    optimizer = optim.SGD(params=model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    #Start training
    train_loss, test_loss = train_model(
        model, train_loader, test_loader, optimizer, config["no_of_epochs"],
        config["report_freq"], device, loss_fn)
    
    #Save model
    savetime = time.strftime(
        '%Y-%m-%d-%H%%%M', time.localtime()).replace('-', '_')
    print(f"Training completed, saving model...")
    save_model(model, savetime)

    





if __name__ == "__main__":
    if not os.path.exists('data/shakespeare.txt'):
        if not os.path.exists('data'):
            os.mkdir('data')
        os.system(
            'wget -O data/shakespeare.txt https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    config = load_config('config.yaml')
    main(config)
    


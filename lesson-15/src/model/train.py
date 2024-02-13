import argparse
import os
from pathlib import Path
from matplotlib import pyplot as plt
import time
import yaml

import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm

from google.cloud import storage
import utils as U



CONFIG_PATH = 'src/config/'

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def train_epoch(
        epoch, model, dataloader, optimizer, 
        report_freq, device, loss_fn):
    epoch_loss = []
    running_loss = 0.0
    i=0
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


def train_model(
        model, train_loader, test_loader, 
        optimizer, no_of_epochs, report_freq,
        device='cpu', loss_fn=nn.CrossEntropyLoss()):
    train_loss = []
    test_loss = []
    for epoch in range(no_of_epochs):
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


def save_model(model, savetime, bucket_name: str=None)-> None:
    model_fn = 'word2vec_skip_gram_model_'+savetime+'.pth'
    Path('output/models/').mkdir(parents=True, exist_ok=True)
    model_fp = 'output/models/'+model_fn
    torch.save(model, model_fp)
    if bucket_name is not None:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        destination_fn = 'output/models/'+model_fn
        blob = bucket.blob(destination_fn)
        blob.upload_from_filename(model_fp)


def plot_loss(train_loss: list, val_loss: list,
              savetime: str, bucket_name: str = None):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(train_loss)
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss")

    ax2.plot(val_loss)
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss")
    fname = 'output/plots/train_val_loss_'+savetime+'.png'
    try:
        plt.savefig(fname)
    except OSError:
        Path('output/plots').mkdir(exist_ok=True)
        plt.savefig(fname)
    if bucket_name is not None:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        destination_fn = 'output/plots/train_val_loss_'+savetime+'.png'
        blob = bucket.blob(destination_fn)
        blob.upload_from_filename(fname)







def main(config, args):
    #Build dataset and dataloaders
    text = open('data/shakespeare.txt', 'rb').read().decode(encoding='utf-8').replace('\n',' ')
    vocab = U.build_vocab(text)
    text = U.tokenize.sent_tokenize(text)
    ds = U.get_dataset(text, ds_len=config["ds_len"],vocab=vocab, window_size=config["window_size"])
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
    plot_loss(
        train_loss, test_loss, savetime, args.bucket_name)
    model_scripted = torch.jit.script(embedder)
    save_model(model_scripted, savetime, args.bucket_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train a skip-gram embedding model")
    parser.add_argument("-b", "--bucket_name",
                        help="Store output in Google Cloud Storage bucket",
                        action='store')
    args = parser.parse_args()
    if args.bucket_name is not None:
        print("Fetching data from Google Cloud Storage...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(
            args.bucket_name)
        blob = bucket.blob('shakespeare.txt')
        try:
            blob.download_to_filename('data/shakespeare.txt')
        except FileNotFoundError:
            Path('data').mkdir()
            blob.download_to_filename('data/shakespeare.txt')
    elif not os.path.exists('data/shakespeare.txt'):
        Path('data').mkdir(exist_ok=True)
        os.system(
            'wget -O data/shakespeare.txt https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    config = load_config('config.yaml')
    main(config, args)
    


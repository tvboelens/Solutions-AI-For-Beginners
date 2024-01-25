import os
from statistics import mean
import time
import yaml

from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from tqdm import tqdm

import utils as U

CONFIG_PATH = 'src/config/'


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def train_epoch(epoch, model, dataloader, optimizer, loss_fn, report_freq, device):
    epoch_loss = []
    running_loss = 0.0
    i=0
    with tqdm(dataloader, unit='batch') as dl_bar:
        dl_bar.set_description(f"Training for epoch {epoch+1}...")
        for images, targets, _ in dl_bar:
            images.to(device)
            targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(images)
            batch_loss = loss_fn(pred, targets)
            batch_loss.backward()
            optimizer.step()
            epoch_loss.append(batch_loss.item())
            running_loss += batch_loss.item()
            if (i+1) == report_freq:
                dl_bar.set_postfix(loss=running_loss/report_freq)
                time.sleep(0.1)
                running_loss = 0.0
                i=0
            else:
                i+=1
    return epoch_loss


def train_model(model, train_loader, val_loader, optimizer, no_of_epochs, report_freq,
                device='cpu', loss_fn=nn.BCEWithLogitsLoss):
    train_loss = []
    val_loss = []
    for epoch in range(no_of_epochs):
        model.train(True)
        epoch_loss = train_epoch(
            epoch, model, train_loader, optimizer, loss_fn, report_freq, device)

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader) as dl_bar:
                dl_bar.set_description(f"Epoch {epoch+1}: validating model...")
                running_vloss=[]
                for vimages, vtargets, _ in dl_bar:
                    vimages.to(device)
                    vtargets.to(device)
                    pred = model(vimages)
                    vloss = loss_fn(pred, vtargets)
                    running_vloss.append(vloss.item())
                    dl_bar.set_postfix(loss=mean(running_vloss))

        val_loss.append(mean(running_vloss))
        train_loss.extend(epoch_loss)

    return train_loss, val_loss

def main(config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = U.UNet()
    model.to(device)

    transforms = T.Compose([T.Resize((256, 256), antialias=None),  
                            T.ToImage(),
                            T.ToDtype(dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.float32, "others": None},
                                      scale=True)])
    root = config["data_dir"]
    train_set, val_set, _ = U.get_dataset(
        dataset_path=root, transforms=transforms, config=config)
    train_loader = DataLoader(train_set, batch_size=config["bs"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["bs"], shuffle=True)

    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    train_loss, val_loss = train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, no_of_epochs=config["no_of_epochs"], 
        report_freq=config["report_freq"], device=device, loss_fn=loss_fn)
    print("TRAINING FINISHED")

    savetime = time.strftime('%Y-%m-%d-%H%%%M', time.localtime()).replace('-', '_')
    U.save_model(model,config["model_output_dir"], savetime)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(train_loss)
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss")

    ax2.plot(val_loss)
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss")

    if not os.path.exists(config["plot_output_dir"]):
        os.mkdir(config["plot_output_dir"])
    fname = config["plot_output_dir"]+'train_val_loss_'+savetime+'.png'
    plt.savefig(fname)

if __name__=='__main__':
    config = load_config('config.yaml')
    main(config)

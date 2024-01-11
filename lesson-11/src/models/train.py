import os
import yaml
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

#import utils as U
from utils import save_model, HollywoodHeadDataset, get_model, collate_fn as cf

import time
from statistics import mean

CONFIG_PATH = 'src/config/'

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config




def train_epoch(model, dataloader, optimizer, report_freq, device):
    epoch_loss = []
    running_loss = 0.0
    starttime = time.time()
    for j, (images, targets, _) in enumerate(dataloader):
            images = images.to(device)
            optimizer.zero_grad(set_to_none=True)
            for target in targets:
                for key in target.keys():
                    target[key] = target[key].to(device)
            batch_loss_dict = model(images, targets)
            batch_loss = sum(loss for loss in batch_loss_dict.values())
            batch_loss.backward()
            optimizer.step()
            epoch_loss.append(batch_loss.item())
            running_loss += batch_loss.item()
            if j % report_freq == report_freq-1:
                print(f"Batch {j+1} finished, "
                      f"time = {int((time.time()-starttime)/60)} minutes {round((time.time()-starttime)%60,2)} seconds, "
                      f"loss: {running_loss/report_freq}")
                starttime = time.time()
                running_loss = 0.0
    return epoch_loss


def train_model(model, train_loader, test_loader, optimizer, no_of_epochs, report_freq, device='cpu'):
    train_loss = []
    val_loss = []
    # accuracy = []
    for epoch in range(no_of_epochs):
        epoch_starttime = time.time()
        print(f"START TRAINING FOR EPOCH {epoch + 1}:")
        model.train(True)
        epoch_loss = train_epoch(
            model, train_loader, optimizer, report_freq, device)
        train_loss += epoch_loss

        running_vloss = 0.0
        model.eval()
        print(f"Training for epoch {epoch+1} done, time = "
              f"{int((time.time()-epoch_starttime)/60)} minutes {round((time.time()-epoch_starttime)%60,2)} seconds")
        with torch.no_grad():
            for i, (vimages, vtargets, _) in enumerate(test_loader):
                vimages = vimages.to(device)
                for target in vtargets:
                    for key in target.keys():
                      target[key] = target[key].to(device)
                vbatch_starttime = time.time()
                vloss_dict = model(vimages, vtargets)
                vloss = sum(loss for loss in vloss_dict.values())
                  # correct = (torch.argmax(vpred, dim=1) == vlabels).type(torch.FloatTensor)
                val_loss.append(vloss.item())
                running_vloss += vloss.item()
                  # accuracy.append(correct.mean().item())
                print(f"Completed validation for batch {i+1}, time = "
                        f"{int((time.time()-vbatch_starttime)/60)} minutes {round((time.time()-vbatch_starttime)%60,2)}"
                        f"seconds")

        val_loss.append(running_vloss/(i+1))
        train_loss += epoch_loss
        print(f"Validation for epoch {epoch+1} done, time = "
              f"{int((time.time()-epoch_starttime)/60)} minutes {round((time.time()-epoch_starttime)%60,2)} seconds, "
              f"LOSS train {epoch_loss[-1]}, val: {val_loss[-1]}")

    return train_loss, val_loss

def main(config):
    transforms = T.Compose([T.Resize((255, 255), antialias=None),
                             T.ToImage(),
                             T.ToDtype(dtype={
                                 tv_tensors.Image: torch.float32,
                                 tv_tensors.BoundingBoxes: torch.float32
                             })])
    #transforms = None#From config?
    #collate_fn = cf
    train_set = HollywoodHeadDataset(
        root='data', transforms=transforms, mode='train')
    val_set = HollywoodHeadDataset(
        root='data', transforms=transforms, mode='val')
    train_loader = DataLoader(
        train_set, batch_size=config["bs_train"], shuffle=True, collate_fn=cf)#collate_fn)
    val_loader = DataLoader(
        val_set, batch_size=config["bs_val"], shuffle=True, collate_fn=cf)#collate_fn)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    trainable_backbone_layers = config["trainable_backbone_layers"]

    model = get_model(trainable_backbone_layers)
    
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=config["lr"], momentum=config["momentum"])

    no_of_epochs = config["no_of_epochs"]
    train_loss, val_loss = train_model(model, train_loader, val_loader, optimizer=optimizer,
                                       no_of_epochs=no_of_epochs, report_freq=config["report_freq"], device=device)
    savetime = time.strftime(
        '%Y-%m-%d-%H%%%M', time.localtime()).replace('-', '_')
    print(f"Training completed, saving model")
    save_model(model, savetime)


    

if __name__=='__main__':
    config = load_config('config.yaml')
    main(config)
    

    

    



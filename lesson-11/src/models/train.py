import os
#from statistics import mean
import time
import yaml

from matplotlib import pyplot as plt
import torch
from torch import optim
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




def train_epoch(epoch,model, dataloader, optimizer, report_freq, device):
    epoch_loss = []
    running_loss = 0.0
    i=0
    with tqdm(dataloader, unit='batch') as tepoch:
        tepoch.set_description(f"Training for epoch {epoch+1}...")
        for images, targets, _ in tepoch:
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
            if (i+1)==report_freq:
                tepoch.set_postfix(loss=running_loss/report_freq)
                time.sleep(0.1)
                #print(f"Batch {j+1} finished, "
                #    f"time = {int((time.time()-starttime)/60)} minutes {round((time.time()-starttime)%60,2)} seconds, "
                #    f"loss: {running_loss/report_freq}")
                running_loss = 0.0
                i=0
            else:
                i+=1
    return epoch_loss


def train_model(
        model, train_loader, test_loader, 
        optimizer, no_of_epochs, report_freq, device='cpu'):
    train_loss = []
    val_loss = []
    # accuracy = []
    for epoch in range(no_of_epochs):
        model.train(True)
        epoch_loss = train_epoch(
            epoch, model, train_loader, optimizer, report_freq, device)
        train_loss += epoch_loss

        running_vloss = 0.0
        model.eval()
        with torch.no_grad():
            i=0
            with tqdm(test_loader, unit='batch') as tepoch:
                tepoch.set_description(f"Epoch {epoch+1}, validating model...")
                for vimages, vtargets, _ in tepoch:
                    vimages = vimages.to(device)
                    for target in vtargets:
                        for key in target.keys():
                            target[key] = target[key].to(device)
                    vloss_dict = model(vimages, vtargets)
                    vloss = sum(loss for loss in vloss_dict.values())
                    # correct = (torch.argmax(vpred, dim=1) == vlabels).type(torch.FloatTensor)
                    val_loss.append(vloss.item())
                    running_vloss += vloss.item()
                    tepoch.set_postfix(loss=running_vloss/(i+1))
                    time.sleep(0.1)
                    i+=1
                    # accuracy.append(correct.mean().item())

        val_loss.append(running_vloss/(i+1))
        train_loss += epoch_loss

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
    train_set = U.HollywoodHeadDataset(
        root='data', transforms=transforms, mode='train')
    val_set = U.HollywoodHeadDataset(
        root='data', transforms=transforms, mode='val')
    train_loader = DataLoader(
        train_set, batch_size=config["bs_train"], shuffle=True, collate_fn=U.collate_fn)
    val_loader = DataLoader(
        val_set, batch_size=config["bs_val"], shuffle=True, collate_fn=U.collate_fn)

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    trainable_backbone_layers = config["trainable_backbone_layers"]

    model = U.get_model(trainable_backbone_layers)
    
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=config["lr"], momentum=config["momentum"])

    no_of_epochs = config["no_of_epochs"]
    train_loss, val_loss = train_model(model, train_loader, val_loader, optimizer=optimizer,
                                       no_of_epochs=no_of_epochs, report_freq=config["report_freq"], device=device)
    savetime = time.strftime(
        '%Y-%m-%d-%H%%%M', time.localtime()).replace('-', '_')
    model_scripted = torch.jit.script(model)
    print(f"Training completed, saving model")
    U.save_model(config["model_output_dir"], model_scripted, savetime)



    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(train_loss)
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss")

    ax2.plot(val_loss)
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss")
    if not os.path.exists('output/plots/'):
        os.mkdir('output/plots')
    fname = 'output/plots/train_val_loss_'+savetime+'.png'
    plt.savefig(fname)


    

if __name__=='__main__':
    config = load_config('config.yaml')
    main(config)
    

    

    



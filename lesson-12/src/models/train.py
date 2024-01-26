import argparse
import os
import time
import yaml

from matplotlib import pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

import utils as U

CONFIG_PATH = 'src/config/'




def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def main(config, args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = U.UNet()
    model.to(device)

    transforms = T.Compose([T.Resize((256, 256), antialias=None),  
                            T.ToImage(),
                            T.ToDtype(dtype={tv_tensors.Image: torch.float32, tv_tensors.Mask: torch.float32, "others": None},
                                      scale=True)])
    root = config["data_dir"]
    train_set, val_set, test_set = U.get_dataset(
        dataset_path=root, transforms=transforms, config=config)
    train_loader = DataLoader(train_set, batch_size=config["bs"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["bs"], shuffle=True)
    if args.test:
        test_loader = DataLoader(test_set, batch_size=config["bs_test"], shuffle=True)


    optimizer = optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()

    train_loss, val_loss = U.train_model(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, no_of_epochs=config["no_of_epochs"], 
        report_freq=config["report_freq"], device=device, loss_fn=loss_fn)
    print("TRAINING FINISHED")

    savetime = time.strftime('%Y-%m-%d-%H%%%M', time.localtime()).replace('-', '_')
    U.save_model(model,config["model_output_dir"], savetime)

    if args.test:
        test_loss = U.test_model(
            model, test_loader, config["plot_output_dir"], config["save_freq"], device)
        print(f"Average loss on test set is {test_loss}")

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
    parser = argparse.ArgumentParser(description='Script to train and optionally test a model')
    parser.add_argument("-t","--test", 
                        help="test model after training",
                        action='store_true')
    args = parser.parse_args()
    config = load_config('config.yaml')
    main(config, args)

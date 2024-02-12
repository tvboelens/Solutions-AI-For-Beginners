import argparse
import os
from statistics import mean
from time import sleep
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from torchvision import tv_tensors

import utils as U

CONFIG_PATH = 'src/config/'


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config



def main(config,args):
    #Load model and move to gpu if available
    model = U.load_model(config, args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    #Build dataset and dataloader
    transforms = T.Compose([T.Resize((255, 255), antialias=None),
                            T.ToImage(),
                            T.ToDtype(dtype={
                                tv_tensors.Image: torch.float32,
                                tv_tensors.BoundingBoxes: torch.float32
                            })])#TODO: Set transforms in utils.py?
    test_set = U.HollywoodHeadDataset(
        root='data', transforms=transforms, mode='test')
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=True, collate_fn=U.collate_fn)

    #Test model
    test_loss = U.test_model(
        model, test_loader, config, device, args.bucket_name)
    print(f"Average loss on test set is {test_loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test a model')
    parser.add_argument("modelname", 
                        help="Filename of model you want to test")
    parser.add_argument("-s",
                        "--save_freq",
                        type=int,
                        help="Frequency of saving output images")
    parser.add_argument("-b", "--bucket_name",
                        type=str,
                        action='store',
                        help="Name of Google Cloud Storage bucket \
                            to store output in")
    args = parser.parse_args()
    config = load_config('config.yaml')
    if args.save_freq is not None:
        config["save_freq"]=args.save_freq
    main(config, args)
    


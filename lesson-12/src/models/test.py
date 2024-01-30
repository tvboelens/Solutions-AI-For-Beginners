import argparse
import os
import yaml

import torch
from torch import nn
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
    # Load model and move to gpu if available
    model_fp = config["model_output_dir"] + args.modelname
    model = torch.jit.load(model_fp)
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    transforms = T.Compose(
        [T.Resize((256, 256), antialias=None),
         T.ToImage(),
         T.ToDtype(
            dtype={
                tv_tensors.Image: torch.float32, 
                tv_tensors.Mask: torch.float32, 
                "others": None},
            scale=True)])
    root = config["data_dir"]
    _, _, test_set = U.get_dataset(
        dataset_path=root, transforms=transforms, config=config)
    test_loader = DataLoader(
        test_set, batch_size=config["bs_test"], shuffle=True)
    loss_fn = nn.BCEWithLogitsLoss()

    test_loss = U.test_model(model, test_loader, config["image_output_dir"],
                             loss_fn, config["save_freq"],device)
    print(f"Average loss on test set is {test_loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test a model')
    parser.add_argument("modelname",
                        help="Filename of model you want to test")
    parser.add_argument("-s",
                        "--save_freq",
                        type=int,
                        help="Frequency of saving output images")
    args = parser.parse_args()
    config = load_config('config.yaml')
    if args.save_freq is not None:
        config["save_freq"] = args.save_freq
    main(config, args)

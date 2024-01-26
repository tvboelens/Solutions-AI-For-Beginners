import argparse
import os
from statistics import mean
from time import sleep
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils as U

CONFIG_PATH = 'src/config/'


def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def test_model(model, test_loader, output_folder, save_freq=100, device='cpu'):
    test_loss = []
    model.eval()
    with torch.no_grad():
        i=0
        with tqdm(test_loader, unit='batch') as tl_bar:
            tl_bar.set_description('Testing model...')
            for img, target, img_filename in tl_bar:
                img = img.to(device)
                for t in target:
                    for key in t:
                        t[key] = t[key].to(device)
                tloss_dict = model(img, target)
                tloss = sum(loss for loss in tloss_dict.values())
                test_loss.append(tloss.item())
                tl_bar.set_postfix(loss=mean(test_loss))
                sleep(0.1)
                if (i+1)%save_freq == 0:
                    U.draw_pred_bounding_boxes(img, model, output_folder, img_filename)
                i+=1
    return mean(test_loss)

def main(args, config):
    #Load model and move to gpu if available
    model_fp = config["model_output_dir"] + args.modelname
    model = torch.load(model_fp)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    #Build dataset and dataloader
    test_set = U.HollywoodHeadDataset(root='data', mode='test')
    test_loader = DataLoader(test_set, batch_size=1,
                             shuffle=True, collate_fn=U.collate_fn)

    #Test model
    test_loss = test_model(
        model, test_loader, config["plot_output_dir"], config["save_freq"], device)
    print(f"Average loss on test set is {test_loss}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to test a model')
    parser.add_argument("modelname", 
                        help="Filename of model you want to test")
    args = parser.parse_args()
    config = load_config('config.yaml')
    main(config, args)
    


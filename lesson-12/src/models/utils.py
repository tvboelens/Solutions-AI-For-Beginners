from matplotlib import pyplot as plt
from pathlib import Path
import os
import shutil
from statistics import mean
from time import sleep
from typing import Callable, Iterable, Optional
import zipfile

from google.cloud import storage
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from tqdm import tqdm




class MADSDataset(Dataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.imgs_dir = os.path.join(root, 'images')
        self.masks_dir = os.path.join(root, 'masks')
        self.img_names = [img_name[:-4] for img_name in os.listdir(self.imgs_dir)
                          if os.path.isfile(os.path.join(self.imgs_dir, img_name))]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_fn = img_name+'.png'
        img_path = os.path.join(self.imgs_dir, img_fn)
        img = Image.open(img_path).convert('RGB')
        img = T.functional.pil_to_tensor(img)

        mask_path = os.path.join(self.masks_dir, self.img_names[idx]+'.png')
        mask = Image.open(mask_path).convert('L')
        mask = T.functional.pil_to_tensor(mask)
        # Since we use binary cross entropy loss, values have to be in [0,1] and
        mask = mask/255.0
        # v2 transforms don't scale masks, therefore we have to do it by hand
        target = tv_tensors.Mask(mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, img_name

    def get_class_index(self, mask):
        mask = torch.where(mask > 128, torch.ones(
            mask.size(), dtype=torch.float32), torch.zeros(mask.size(), dtype=torch.float32))
        mask = tv_tensors.Mask(mask)
        return mask
    

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
        self.act0 = nn.ReLU()
        self.bn0 = nn.BatchNorm2d(16)
        self.pool0 = nn.MaxPool2d(kernel_size=(2, 2))

        self.enc_conv1 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.act1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.enc_conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.act2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.enc_conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.act3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2))

        self.bottleneck_conv = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)

        self.upsample0 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv0 = nn.Conv2d(
            in_channels=384, out_channels=128, kernel_size=(3, 3), padding=1)
        self.dec_act0 = nn.ReLU()
        self.dec_bn0 = nn.BatchNorm2d(128)

        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv1 = nn.Conv2d(
            in_channels=192, out_channels=64, kernel_size=(3, 3), padding=1)
        self.dec_act1 = nn.ReLU()
        self.dec_bn1 = nn.BatchNorm2d(64)

        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv2 = nn.Conv2d(
            in_channels=96, out_channels=32, kernel_size=(3, 3), padding=1)
        self.dec_act2 = nn.ReLU()
        self.dec_bn2 = nn.BatchNorm2d(32)

        self.upsample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_conv3 = nn.Conv2d(
            in_channels=48, out_channels=1, kernel_size=(1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e0 = self.pool0(self.bn0(self.act0(self.enc_conv0(x))))
        e1 = self.pool1(self.bn1(self.act1(self.enc_conv1(e0))))
        e2 = self.pool2(self.bn2(self.act2(self.enc_conv2(e1))))
        e3 = self.pool3(self.bn3(self.act3(self.enc_conv3(e2))))

        cat0 = self.bn0(self.act0(self.enc_conv0(x)))
        cat1 = self.bn1(self.act1(self.enc_conv1(e0)))
        cat2 = self.bn2(self.act2(self.enc_conv2(e1)))
        cat3 = self.bn3(self.act3(self.enc_conv3(e2)))

        b = self.bottleneck_conv(e3)

        d0 = self.dec_bn0(self.dec_act0(self.dec_conv0(
            torch.cat((self.upsample0(b), cat3), dim=1))))
        d1 = self.dec_bn1(self.dec_act1(self.dec_conv1(
            torch.cat((self.upsample1(d0), cat2), dim=1))))
        d2 = self.dec_bn2(self.dec_act2(self.dec_conv2(
            torch.cat((self.upsample2(d1), cat1), dim=1))))
        d3 = self.sigmoid(self.dec_conv3(
            torch.cat((self.upsample3(d2), cat0), dim=1)))
        return d3
    

def train_epoch(epoch: str, model: Callable, dataloader: Iterable, 
                optimizer: torch.optim, loss_fn: Callable, report_freq: int, 
                device: str = 'cpu') -> list:
    epoch_loss = []
    running_loss = 0.0
    i = 0
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
                sleep(0.1)
                running_loss = 0.0
                i = 0
            else:
                i += 1
    return epoch_loss


def train_model(model: Callable, train_loader: Iterable, 
                val_loader: Iterable, optimizer: torch.optim, no_of_epochs: int, 
                report_freq: int,
                device='cpu', loss_fn=nn.BCEWithLogitsLoss) -> tuple[list]:
    train_loss = []
    val_loss = []
    for epoch in range(no_of_epochs):
        model.train(True)
        epoch_loss = train_epoch(
            epoch, model, train_loader, optimizer, loss_fn, report_freq, device)

        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, unit='batch') as dl_bar:
                dl_bar.set_description(f"Epoch {epoch+1}: validating model...")
                running_vloss = []
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


def test_model(model: Callable, test_loader: Iterable, config: dict, 
               loss_fn: Callable, device: str = 'cpu', 
               bucket_name: Optional[str]=None) -> float:
    test_loss = []
    model.eval()
    with torch.no_grad():
        i = 0
        with tqdm(test_loader, unit='batch') as tl_bar:
            tl_bar.set_description('Testing model...')
            for img, target, img_filenames in tl_bar:
                img.to(device)
                target.to(device)
                pred = model(img)
                tloss = loss_fn(pred, target)
                test_loss.append(tloss.item())
                tl_bar.set_postfix(loss=mean(test_loss))
                sleep(0.1)
                if (i+1) == config["save_freq"]:
                    draw_pred_segmentation_masks(
                        model, img, img_filenames, config)
                    i=0
                else:
                    i += 1
        if bucket_name is not None:
            pred_fnames = [fn for fn in os.listdir(config["image_output_dir"])
                           if os.path.isfile(os.path.join(config["image_output_dir"],fn))]
            storage_client = storage.Client()
            bucket = storage_client.get_bucket(bucket_name)
            with tqdm(pred_fnames, unit = 'img') as ul_bar:
                ul_bar.set_description('Uploading images to GCS...')
                for fn in ul_bar:
                    fp = os.path.join(config["image_output_dir"],fn)
                    destination_fp = 'output/images/'+fn
                    blob = bucket.blob(destination_fp)
                    blob.upload_from_filename(fp)
    return mean(test_loss)

def save_model(model: nn.Module, output_dir: str, savetime: str, 
               bucket_name: Optional[str] = None) -> None:
    model_fn = 'BodySegmentation_model_'+savetime+'.pth'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_fp = os.path.join(output_dir, model_fn)
    torch.save(model.state_dict(),model_fp)

    if bucket_name is not None:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        destination_fn = 'output/models/'+model_fn
        blob = bucket.blob(destination_fn)
        blob.upload_from_filename(model_fp)


def draw_pred_segmentation_masks(model: Callable, imgs: torch.Tensor, 
                                 img_names: tuple, config: dict) -> None:
    imgs_path = os.path.join(config["data_dir"],'images')
    masks_path = os.path.join(config["data_dir"],'masks')
    model.eval()
    pred = model(imgs)
    pred[pred > 0.5] = 255.0
    pred[pred <= 0.5] = 0.0

    Path(config["image_output_dir"]).mkdir(parents=True, exist_ok=True)
    pred_fnames = [fn+'_pred.png' for fn in img_names]
    for i in range(len(img_names)):
        fn = pred_fnames[i]
        fp = os.path.join(config["image_output_dir"], fn)
        to_pil_transform = T.ToPILImage()
        output_img = to_pil_transform(pred[i,:,:,:]).resize((512,384))
        img = Image.open(os.path.join(imgs_path,img_names[i] +'.png')).convert('RGB')
        mask = Image.open(os.path.join(masks_path, img_names[i] +'.png')).convert('L')
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.set_title('Image')
        ax1.axis('off')
        ax1.imshow(img)
        ax2.set_title('Original mask')
        ax2.axis('off')
        ax2.imshow(mask)
        ax3.set_title('Predicted mask')
        ax3.axis('off')
        ax3.imshow(output_img)
        plt.savefig(fp)
        plt.close()

def get_dataset(dataset_path: str, transforms: Callable, 
                config: dict) -> tuple[Dataset]:
    if not os.path.exists(dataset_path):
        with zipfile.ZipFile('mads_ds_1192.zip') as file:
            file.extractall()
        for dir in os.listdir(os.path.join(dataset_path,dataset_path)):
            shutil.move(os.path.join(dataset_path, dataset_path, dir),
                        os.path.join(dataset_path, dir))
        shutil.rmtree(os.path.join(dataset_path, dataset_path))
    full_dataset = MADSDataset(root=dataset_path, transforms=transforms)
    #set random seed for random_split, since we potentially call it twice 
    #(once for training and once for testing)
    generator = torch.Generator().manual_seed(29)
    train_set, val_set, test_set = random_split(
        full_dataset, [config["train_size"], 
                       config["val_size"], 
                       config["test_size"]],
                       generator=generator)
    return train_set, val_set, test_set

def plot_loss(train_loss: list, val_loss: list, 
              config: dict, savetime:str, bucket_name=None) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(train_loss)
    ax1.set_ylabel("Loss")
    ax1.set_title("Train Loss")

    ax2.plot(val_loss)
    ax2.set_ylabel("Loss")
    ax2.set_title("Validation Loss")

    Path(config["plot_output_dir"]).mkdir(parents=True, exist_ok=True)
    fname = config["plot_output_dir"]+'train_val_loss_'+savetime+'.png'
    plt.savefig(fname)
    plt.close()

    if bucket_name is not None:
        destination_fp = 'output/images/train_val_loss_'+savetime+'.png'
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(destination_fp)
        blob.upload_from_filename(fname)        



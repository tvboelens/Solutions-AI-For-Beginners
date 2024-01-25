import os
import shutil
from typing import Callable, Optional
import zipfile

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, random_split
from torchvision import tv_tensors
from torchvision.utils import save_image
from torchvision.transforms import v2 as T




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
        # img = torchvision.io.read_image(img_path, ImageReadMode.RGB)

        mask_path = os.path.join(self.masks_dir, self.img_names[idx]+'.png')
        mask = Image.open(mask_path).convert('L')
        mask = T.functional.pil_to_tensor(mask)
        # Since we use binary cross entropy loss, values have to be in [0,1] and
        mask = mask/255.0
        # v2 transforms don't scale masks, therefore we have to do it by hand
        # mask = torchvision.io.read_image(mask_path, ImageReadMode.GRAY)
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

def save_model(model, output_dir, savetime):
    model_fn = 'BodySegmentation_model_'+savetime+'.pth'
    if not os.path.exists(output_dir):
        os.makedir(output_dir)

    model_fp = os.path.join(output_dir, model_fn)
    torch.save(model, model_fp)


def draw_pred_segmentation_masks(model: Callable, img: torch.Tensor, img_name: str) -> None:
    # TODO: output both the predicted mask and the mask over the image
    model.eval()
    pred = model(img)
    img = img.reshape(img.size()[1:]).to(torch.uint8)

    # pred = pred.reshape(pred.size()[2:]).bool()
    pred[pred > 0.5] = 255.0
    pred[pred <= 0.5] = 0.0
    # img = torchvision.utils.draw_segmentation_masks(img, pred)
    # img = torchvision.transforms.ToPILImage()(img)
    fn = img_name+'_pred.png'
    fp = os.path.join('output_imgs', fn)
    if not os.path.exists('output_imgs'):
        os.mkdir('output_imgs')
    # img.save(fp)
    save_image(pred, fp)
    return None

def get_dataset(dataset_path, transforms, config):
    if not os.path.exists(dataset_path):
        with zipfile.ZipFile('mads_ds_1192.zip') as file:
            file.extractall()
        for dir in os.listdir(os.path.join(dataset_path,dataset_path)):
            shutil.move(os.path.join(dataset_path, dataset_path, dir),
                        os.path.join(dataset_path, dir))
        shutil.rmtree(os.path.join(dataset_path, dataset_path))
    full_dataset = MADSDataset(root=dataset_path, transforms=transforms)
    train_set, val_set, test_set = random_split(#TODO: set a random seed
        full_dataset, [config["train_size"], 
                       config["val_size"], 
                       config["test_size"]])
    return train_set, val_set, test_set





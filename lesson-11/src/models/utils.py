from xml.etree import ElementTree as et
from statistics import mean
import os
from pathlib import Path
from functools import partial
from time import sleep
from tqdm import tqdm

from matplotlib import pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import draw_bounding_boxes, save_image
from torchvision.models import resnet50
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import backbone_utils
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from google import storage
import utils as U

class HollywoodHeadDataset(Dataset):
    def __init__(self, root, transforms=None, mode='train') -> None:
        super().__init__()
        assert mode.lower() in ['train', 'test', 'val']
        self.transforms = transforms
        self.root = root

        filename = mode.lower() + '.txt'
        filepath = os.path.join(root, 'Splits', filename)

        with open(filepath, 'r') as f:
            img_names = f.readlines()
        self.imgs = [img.strip('\n') for img in img_names]

        self.imgs_dir = os.path.join(root, 'JPEGImages')
        self.annot_dir = os.path.join(root, 'Annotations')
        # self.classes = ['background','head']

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_filename = self.imgs[idx]+'.jpeg'
        image_path = os.path.join(self.imgs_dir, img_filename)

        annot_filename = self.imgs[idx]+'.xml'
        annot_file_path = os.path.join(self.annot_dir, annot_filename)

        img = read_image(image_path, ImageReadMode.RGB)
        boxes = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        for object in root.findall('object'):
            if object.find('bndbox') is not None:
                xmin = float(object.find('bndbox').find('xmin').text)
                xmax = float(object.find('bndbox').find('xmax').text)

                ymin = float(object.find('bndbox').find('ymin').text)
                ymax = float(object.find('bndbox').find('ymax').text)

                boxes.append([xmin, ymin, xmax, ymax])

        img = tv_tensors.Image(img)
        if len(boxes) != 0:
            boxes = tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=img.shape[-2:])
            # We only have one class
            labels = torch.ones(boxes.shape[0], dtype=torch.int64)
        # If there are no bounding boxes in the image put in a degenerate box and label it as background
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            boxes = tv_tensors.BoundingBoxes(
                boxes, format="XYXY", canvas_size=img.shape[-2:])
            labels = torch.zeros(0, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, img_filename

def save_model(output_dir, model, savetime, bucket_name:str=None):
    model_fn = 'HollywoodHeadDetection_model_'+savetime+'.pth'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_fp = os.path.join(output_dir, model_fn)
    model.save(model_fp)
    if bucket_name is not None:
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        destination_fn = 'output/models/'+model_fn
        blob = bucket.blob(destination_fn)
        blob.upload_from_filename(model_fp)

def draw_pred_bounding_boxes(img, model, output_folder, img_name, bucket_name=None):
    model.eval()

    # img = tv_tensors.Image(img)
    pred = model(img)
    boxes = pred["boxes"]

    img = draw_bounding_boxes(img, boxes)
    # img = torchvision.transforms.ToPILImage()(img)
    fn = img_name+'_pred.jpg'
    fp = os.path.join(output_folder, fn)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    save_image(img, fp)


def draw_annot_bounding_boxes(img_name):
    img_dir = 'data/JPEGImages/'
    annot_dir = 'data/Annotations/'

    annot_filepath = os.path.join(annot_dir, img_name+'.xml')
    img_filepath = os.path.join(img_dir, img_name+'.jpeg')
    img = read_image(img_filepath, ImageReadMode.RGB)

    img = tv_tensors.Image(img)
    boxes = []
    tree = et.parse(annot_filepath)
    root = tree.getroot()
    for object in root.findall('object'):
        if object.find('bndbox') is not None:
            xmin = float(object.find('bndbox').find('xmin').text)
            xmax = float(object.find('bndbox').find('xmax').text)

            ymin = float(object.find('bndbox').find('ymin').text)
            ymax = float(object.find('bndbox').find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
    if len(boxes) != 0:
        img = draw_bounding_boxes(img, boxes)
    # img = torchvision.transforms.ToPILImage()(img)
    output_fp = os.path.join('data/output/', img_name+'_annot.jpeg')
    save_image(img, output_fp)
    return None


def collate_fn(batch): 
    images = []
    targets = []
    img_names = []
    for b in batch:
        images.append(b[0])
        targets.append(b[1])
        img_names.append(b[2])
    images = torch.stack(images, dim=0)
    return images, targets, img_names

def get_model(trainable_backbone_layers):
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3)))
                         for x in [8, 16, 32, 64, 128])
    aspect_ratios = ((0.25, 0.5, 1.0, 1.5),)*len(anchor_sizes)
    anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

    backbone = resnet50(weights='DEFAULT')
    backbone = backbone_utils._resnet_fpn_extractor(
        backbone,
        trainable_layers=trainable_backbone_layers,
        returned_layers=[2, 3, 4],
        extra_blocks=LastLevelP6P7(2048, 256))

    backbone.out_channels = 256
    head = RetinaNetHead(backbone.out_channels,
                         anchor_generator.num_anchors_per_location()[0],
                         num_classes=2,
                         norm_layer=partial(torch.nn.GroupNorm, 32))

    head.regression_head._loss_type = "giou"
    model = RetinaNet(backbone=backbone,
              num_classes=2,
              anchor_generator=anchor_generator,
              head=head)
    return model

def plot_loss(train_loss: list, val_loss: list,
              savetime: str, bucket_name: str=None):
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


def test_model(model, test_loader, config, device='cpu', bucket_name=None):
    test_loss = []
    model.eval()
    with torch.no_grad():
        i = 0
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
                if (i+1) % config["save_freq"] == 0:
                    U.draw_pred_bounding_boxes(
                        img, model, config["image_output_dir"], img_filename)
                i += 1
    if bucket_name is not None:
        pred_fnames = [fn for fn in os.listdir(config["image_output_dir"])
                       if os.path.isfile(os.path.join(config["image_output_dir"], fn))]
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        with tqdm(pred_fnames, unit='img') as ul_bar:
            ul_bar.set_description('Uploading images to GCS...')
            for fn in ul_bar:
                fp = os.path.join(config["image_output_dir"], fn)
                destination_fp = 'output/images/'+fn
                blob = bucket.blob(destination_fp)
                blob.upload_from_filename(fp)
    return mean(test_loss)

def load_model(config, args):
    model_fp = config["model_output_dir"] + args.modelname
    if args.bucket_name is not None:
        print("Fetching model from Google Cloud Storage...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(
            args.bucket_name)
        source_fn = 'output/models/'+args.modelname
        blob = bucket.blob(source_fn)
        # If model is stored in GCS, then config["model_output_dir"] might not exist
        try:
            blob.download_to_filename(model_fp)
        except FileNotFoundError:
            Path(config["model_output_dir"]).mkdir(parents=True)
            blob.download_to_filename(model_fp)
        print("Model download complete")
    model = torch.jit.load(model_fp)

    return model

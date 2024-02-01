import os
import shutil
import zipfile

dataset_path = 'segmentation_full_body_mads_dataset_1192_img'


with zipfile.ZipFile('mads_ds_1192.zip') as file:
    file.extractall()
for dir in os.listdir(os.path.join(dataset_path, dataset_path)):
    shutil.move(os.path.join(dataset_path, dataset_path, dir),
                os.path.join(dataset_path, dir))
shutil.rmtree(os.path.join(dataset_path, dataset_path))

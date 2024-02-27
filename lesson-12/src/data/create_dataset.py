import argparse
from pathlib import Path
import os
import shutil
import zipfile

from google.cloud import storage

dataset_path = 'segmentation_full_body_mads_dataset_1192_img'




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bucket_name',
                    action='store',
                    help='Provide bucket name\
                          to fetch dataset from \
                          Google Cloud Storage bucket',
                    type=str)
    args = parser.parse_args()
    if not os.path.exists('data'):
        if args.bucket_name is not None:
            print("Fetching data from Google Cloud Storage...")
            storage_client = storage.Client()
            bucket = storage_client.bucket(
                args.bucket_name)
            blob = bucket.blob('mads_ds_1192.zip')
            blob.download_to_filename('mads_ds_1192.zip')
        
        print("Unzipping dataset...")    
        with zipfile.ZipFile('mads_ds_1192.zip') as file:
            file.extractall()
        if args.bucket_name is not None:
            Path('mads_ds_1192.zip').unlink(missing_ok=True)
        for dir in os.listdir(os.path.join(dataset_path, dataset_path)):
            shutil.move(os.path.join(dataset_path, dataset_path, dir),
                    os.path.join(dataset_path, dir))
        shutil.rmtree(os.path.join(dataset_path, dataset_path))
        os.rename(dataset_path, 'data')
        print("Dataset created.")
            


import argparse
from pathlib import Path
import os
import zipfile

from google.cloud import storage

parser = argparse.ArgumentParser(description='Script to download Hollywood Heads Dataset and \
                                 unpack the zip file.')
parser.add_argument('-r',
                    '--remove_zip',
                    action='store_true',
                    help='Remove zip file after unpacking')
parser.add_argument('-b', '--bucket_name',
                    action='store',
                    help='Provide bucket name\
                          to fetch dataset from \
                          Google Cloud Storage bucket',
                    type=str)
args = parser.parse_args()

if not os.path.exists('data'):
    if not os.path.exists('HollywoodHeads.zip'):
        print('Downloading dataset, this might take a while...')
        if args.bucket_name is not None:
            print("Fetching data from Google Cloud Storage...")
            storage_client = storage.Client()
            bucket = storage_client.bucket(
                args.bucket_name)
            blob = bucket.blob('HollywoodHeads.zip')
            blob.download_to_filename('HollywoodHeads.zip')
        else:
            os.system('wget https://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip')

    with zipfile.ZipFile('HollywoodHeads.zip') as file:
        print('Unzipping dataset...')
        file.extractall()
    os.rename('HollywoodHeads','data')
    if args.remove_zip:
        Path('HollywoodHeads.zip').unlink()
    print('Dataset created')
else:
    print('Dataset already created. If you want to create it anew, first remove the folder \'data/\'.')
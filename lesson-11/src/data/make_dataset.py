import argparse
import os
import zipfile

parser = argparse.ArgumentParser(description='Script to download Hollywood Heads Dataset and \
                                 unpack the zip file.')
parser.add_argument('-r',
                    '--remove_zip',
                    action='store_true',
                    help='Remove zip file after unpacking')
args = parser.parse_args()

if not os.path.exists('data'):
    if not os.path.exists('HollywoodHeads.zip'):
        print('Downloading dataset, this might take a while...')
        os.system('wget https://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip')

    with zipfile.ZipFile('HollywoodHeads.zip') as file:
        print('Unzipping dataset...')
        file.extractall()
    os.rename('HollywoodHeads','data')
    if args.remove_zip:
        os.system('rm HollywoodHeads.zip')
    print('Dataset created')
else:
    print('Dataset already created. If you want to create it anew, first remove the folder \'data/\'.')
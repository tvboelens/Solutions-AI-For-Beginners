import os
import zipfile

if not os.path.exists('data'):
    if not os.path.exists('HollywoodHeads.zip'):
        print('Downloading dataset, this might take a while')
        os.system('wget https://www.di.ens.fr/willow/research/headdetection/release/HollywoodHeads.zip')

    with zipfile.ZipFile('HollywoodHeads.zip') as file:
        print('Unzipping dataset')
        file.extractall()
    os.rename('HollywoodHeads','data')
    os.system('rm HollywoodHeads.zip')
    print('Dataset created')
else:
    print('Dataset already created. If you want to create it anew, first remove the folder \'data/\'.')
# Lesson 12: Segmentation
Solution for [lab assignment of lesson 12](https://github.com/microsoft/AI-For-Beginners/tree/main/lessons/4-ComputerVision/12-Segmentation/lab), which involves segmentation of body poses from images. This solution contains both a notebook and scripts, which train a U-Net model. The scripts can be run either locally or on Google Cloud Platform in a Docker container. Apart from downloading the data the notebook is entirely self-contained. Code was developed in Python 3.10.
## Data
The assigment uses a subset of [the MADS full body dataset](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset). Download the data manually from Kaggle and then either:
1. Run the notebook
2. Run `src/data/create_dataset.py`
3. If you want to train on GCP in a Docker container: Store the zip file in a Google Cloud Storage Bucket as `mads_ds_1192.zip`. 


# Training and testing the model locally
Run `src/models/train.py` to train the model. 
Use the optional flag `-t` or  `--test` to immediately test the model after training. 
Hyperparameters and other settings can be adjusted in `src/config/config.yaml`.
If you want to test a previously trained model, run `src/models/test.py modelname`, where `modelname` is the filename of the saved model (just the filename, not the path, which should be set in the config file).

# Training and testing the model on GCP in a Docker container
1. To train the model run the Docker container with the argument
    ```
    shell_scripts/train.sh YOUR_BUCKET
    ```
    
2.  To train the model and subsequently test it run the Docker container with the argument
    ```
    shell_scripts/train_test.sh YOUR_BUCKET
    ```
3.  If you already have a trained model stored in Google Cloud Storage and want to test it, run the Docker container with the argument
    ```
    shell_scripts/test.sh YOUR_BUCKET MODELNAME
    ```
Here `YOUR_BUCKET` is the name of the Google Cloud Storage bucket  and `MODELNAME` is the filename of the model, which is assumed to be stored in the folder `output/models`. The bucket needs to be in the same project as the Google Cloud VM on which the container is running and the Container needs both read and write permissions for Google Cloud Storage.

Testing will generate output images for visual inspection in the folder `output/images`.



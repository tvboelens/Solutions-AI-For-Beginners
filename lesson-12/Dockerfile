FROM ubuntu:22.04

RUN apt update
RUN apt-get update -y
RUN apt-get install -y python3-pip python3.10
RUN pip3 install --upgrade pip

COPY ./src/ /src/
#COPY mads_ds_1192.zip mads_ds_1192.zip
COPY requirements.txt requirements.txt

RUN mkdir output
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python3"]
CMD ["src/data/get_dataset.py -g body_segmentation_bucket" ]
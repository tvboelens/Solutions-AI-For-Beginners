FROM ubuntu:latest

RUN apt update
RUN apt-get update -y
RUN apt-get install -y python3-pip python3.10
RUN pip install --upgrade pip

COPY ./src /src
COPY mads_ds_1192.zip mads_ds_1192.zip
COPY requirements.txt requirements.txt

RUN mkdir output
RUN pip3 install -r requirements.txt

CMD [ "python3", "src/data/get_dataset.py" ]
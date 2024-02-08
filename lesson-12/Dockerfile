FROM ubuntu:22.04

RUN apt update
RUN apt install sudo
RUN apt-get update -y
RUN apt-get install -y python3-pip python3.10
RUN pip3 install --upgrade pip

COPY ./src/ src/
COPY ./shell_scripts shell_scripts/
COPY requirements.txt requirements.txt

RUN mkdir output
RUN pip3 install -r requirements.txt

CMD [ "shell_scripts/train.sh", "body-segmentation-bucket" ]
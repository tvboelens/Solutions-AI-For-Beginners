FROM pytorch/pytorch:latest

COPY ./src src/
COPY ./requirements.txt requirements.txt
#COPY ./shell_scripts ./shell_scripts
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
CMD ["python3", "src/model/train.py", "-b", "skip-gram-bucket" ]

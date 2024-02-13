#!/bin/sh
python3 src/data/make_dataset.py -b $1
python3 src/models/train.py -b $1
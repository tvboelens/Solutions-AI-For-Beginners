#!/bin/sh
python3 src/data/create_dataset.py -b $1
python3 src/models/test.py $2 -b $1  
#!/bin/zsh

python3 ./scripts/preprocessing.py train
python3 ./scripts/preprocessing.py test
python3 ./scripts/wordEmbedding.py train
python3 ./scripts/wordEmbedding.py test
python3 ./scripts/modelFitting.py
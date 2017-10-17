#!/bin/zsh

# run this being on the base directory , not the script directory
# it was used python 3.5

python ./scripts/preprocessing.py train
python ./scripts/preprocessing.py test
python ./scripts/wordEmbedding.py "train" "nDocsToUse=50" "minFreq_ofWords=5" "nTokens_inDoc=20"
python ./scripts/wordEmbedding.py "test" "nDocsToUse=50" "minFreq_ofWords=5" "nTokens_inDoc=20"
python ./scripts/modelFitting.py
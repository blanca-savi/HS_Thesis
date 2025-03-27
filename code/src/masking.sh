#!/bin/bash

if [ ! -d "ConfMtxes" ]; then
  mkdir ConfMtxes
  cd ConfMtxes
  
  mkdir bbase
  mkdir distilroberta
  
  cd bbase
  mkdir plots
  cd ..
  
  cd distilroberta
  mkdir plots
  cd ../..

fi
cd utils
python3 dataset_preprocess.py expanded
cd ../

./mask_words.sh
./mask_lists.sh

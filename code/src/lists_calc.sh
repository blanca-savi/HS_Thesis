#!/bin/bash

if [ ! -d "shapval" ]; then
    mkdir shapval
fi

python3 utils/common_words.py
python3 shap_values.py bbase
python3 shap_values.py distilroberta

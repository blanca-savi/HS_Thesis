#!/bin/bash


python3 masking_dataset.py  bbase common
python3 masking_dataset.py  bbase top_nor_bbase
python3 masking_dataset.py  bbase top_nor_pos_bbase
python3 masking_dataset.py  bbase top_hs_bbase
python3 masking_dataset.py  bbase top_off_bbase
python3 masking_dataset.py  bbase all_influential_bbase
python3 masking_dataset.py  bbase intersection_bbase

#IDENTITY TARGETS:
python3 masking_dataset.py  bbase black_target_bbase
python3 masking_dataset.py  bbase black_target_no_nigger_bbase
python3 masking_dataset.py  bbase jewish_target_bbase
python3 masking_dataset.py  bbase disabilities_target_bbase
python3 masking_dataset.py  bbase woman_target_bbase
python3 masking_dataset.py  bbase hispanic_target_bbase
python3 masking_dataset.py  bbase muslim_target_bbase



python3 masking_dataset.py  distilroberta common
python3 masking_dataset.py  distilroberta top_nor_roberta
python3 masking_dataset.py  distilroberta top_nor_pos_roberta
python3 masking_dataset.py  distilroberta top_hs_roberta
python3 masking_dataset.py  distilroberta top_off_roberta
python3 masking_dataset.py  distilroberta all_influential_roberta
python3 masking_dataset.py  distilroberta intersection_roberta

#IDENTITY TARGETS:
python3 masking_dataset.py  distilroberta black_target_roberta
python3 masking_dataset.py  distilroberta black_target_no_nigger_roberta
python3 masking_dataset.py  distilroberta hispanic_target_roberta
python3 masking_dataset.py  distilroberta jewish_target_roberta
python3 masking_dataset.py  distilroberta muslim_target_roberta
python3 masking_dataset.py  distilroberta disabilities_target_roberta
python3 masking_dataset.py  distilroberta lgtb_target_roberta





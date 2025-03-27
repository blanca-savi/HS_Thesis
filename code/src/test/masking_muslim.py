import datasets
import pandas as pd
import numpy as np
import transformers
import joblib
import shap
import re
import argparse

from datasets import load_dataset, load_from_disk, Dataset, ClassLabel, Features, Value, Sequence
from sklearn.preprocessing import LabelEncoder
from collections import Counter    

def masking_function(examples):  
  examples["input_ids"] = [
    [mask_token_id[0] if token_id == word_encoded[0] else token_id for token_id in sublist]
    for sublist in examples["input_ids"]
  ]

  return examples

def mask_words(words):
    
    pattern = '|'.join(words)
    masked_dataset = data.copy()
    masked_dataset['text'] = masked_dataset['text'].str.replace(pattern, '[MASK]', regex=True)
    return masked_dataset



def fixing_function(examples):
     
    final_labels = [Counter(notations['label']).most_common(1)[0][0] for notations in examples['annotators']]   
    examples['final_label'] = final_labels
    
    fixed_ptokens = [
        " ".join([token for token in sentence if not token.startswith('<')])
        for sentence in examples["post_tokens"]         
    ]
    
    examples["post_tokens"] = fixed_ptokens
    encoded_input=tokenizer(examples['post_tokens'], truncation=True)
    return encoded_input


def dataset_preprocess():
    dataset = load_dataset("hatexplain")
    splits = ['train', 'test', 'validation']
    for split in splits:
        values = ["foo"] * len(dataset[split])
        dataset[split] = dataset[split].add_column("final_label", values)

    return dataset.map(fixing_function, batched=True)    
    
    
def shap_calculator(model, dataset):
    pred = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        return_all_scores=True,
    )

    explainer = shap.Explainer(pred)
    
    shap_values = explainer(dataset) 
    
    return shap_values
    

def main():
    parser = argparse.ArgumentParser(description='Calculate shap values')
    parser.add_argument('model', type=str, nargs='?', default='bbase', help='Model to use, either bbase or distiroberta')
    args = parser.parse_args()
    chosen_model = args.model

    encoder = LabelEncoder()
    encoder.classes_ = np.load('/home/blanca/Documents/Thesis_Code/Dataset/classes.npy',allow_pickle=True)
    id2label = {idx: label for idx, label in enumerate(encoder.classes_)}
    label2id = {label: idx for idx, label in enumerate(encoder.classes_)}
    global tokenizer
    
    if(chosen_model == 'distilroberta'):
        tokenizer = transformers.AutoTokenizer.from_pretrained('blancasavi/pretrained_light')
        model = transformers.AutoModelForSequenceClassification.from_pretrained("blancasavi/pretrained_light",num_labels=len(encoder.classes_), id2label=id2label, label2id=label2id).cuda()
        dataset_name = 'dataset_light'
        
        
    else:
        tokenizer = transformers.BertTokenizer.from_pretrained('blancasavi/bbase_sandy')
        model = transformers.AutoModelForSequenceClassification.from_pretrained("blancasavi/bbase_sandy",num_labels=len(encoder.classes_), id2label=id2label, label2id=label2id).cuda()
        dataset_name = 'dataset_sandy'


    dataset = dataset_preprocess()
    dataset.save_to_disk(f'/home/blanca/Documents/Thesis_Code/code/saved_datasets/{dataset_name}')
    
    
    # TEST ________________________________________________________________________________________________

    global word_encoded
    word_encoded = tokenizer.encode("muslim", add_special_tokens=False)
    filtered_dataset = dataset.filter(
        lambda x: (
            any(token == word_encoded[0] for token in x['input_ids'])
        )
    )

    
    global  mask_token_id
    mask_token_id = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
    masked_dataset = filtered_dataset.map(masking_function, batched=True)
    
    decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in masked_dataset['test']['input_ids']]
    print(decoded_texts)

    
    shap_values = shap_calculator(model, decoded_texts)
    

    # _____________________________________________________________________________________________________
    
    filename = 'masked_muslim_shapvals'
    path=f'./shapval/{filename}'
    joblib.dump(shap_values, filename=path, compress=('bz2', 9))


if __name__ == "__main__":
    main()
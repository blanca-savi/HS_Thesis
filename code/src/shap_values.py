import os
import pandas as pd
import numpy as np
import transformers
import joblib
import shap
import argparse

from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from collections import Counter    

def fixing_function(examples):
    """
    This function processes the input examples by assigning a final label based on majority voting,
    reconstructing the sentences by joining the words, and removing any special characters.
    It also tokenizes the input with the original BERT-base tokenizer.

    Parameters:
    - examples (dict): A dictionary containing the following keys:
        - 'annotators': A list of annotations, where each element is a dictionary containing a 'label' key.
        - 'post_tokens': A list of tokenized sentences (where each sentence is a list of strings).

    Returns:
    - The tokenized input.
    """

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
    """
    Adds a new column called 'final_label' to each dataset split.
    Applies the `fixing_function` to preprocess the dataset.

    Returns:
    The preprocessed dataset with the added 'final_label' column and the applied changes from `fixing_function`.
    """

    dataset = load_dataset("hatexplain")
    splits = ['train', 'test', 'validation']
    for split in splits:
        values = ["foo"] * len(dataset[split])
        dataset[split] = dataset[split].add_column("final_label", values)

    return dataset.map(fixing_function, batched=True)    

def shap_calculator(model, dataset):
    """
    Calculates SHAP (SHapley Additive exPlanations) values.

    Parameters:
    - model (transformers.PreTrainedModel): The trained model to explain.
    - dataset (dict): The dataset to use for generating SHAP values.

    Returns:
    - shap_values (shap.Explanation): SHAP values that explain the model's predictions.
    """
    pred = transformers.pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0,
        return_all_scores=True,
    )   

    explainer = shap.Explainer(pred)
    
    
    dataset = dataset["test"]
    data = pd.DataFrame({"text": dataset["post_tokens"], "label": dataset["final_label"]})

    shap_values = explainer(data["text"])

    return shap_values

def main():
    parser = argparse.ArgumentParser(description='Calculate shap values')
    parser.add_argument('model', type=str, nargs='?', default='bbase', help='Model to use, either bbase or distiroberta')
    args = parser.parse_args()
    chosen_model = args.model

    encoder = LabelEncoder()
    encoder_path = os.path.join('..' , '..', 'Dataset', 'classes.npy')
    encoder.classes_ = np.load(encoder_path,allow_pickle=True)
    id2label = {idx: label for idx, label in enumerate(encoder.classes_)}
    label2id = {label: idx for idx, label in enumerate(encoder.classes_)}
    global tokenizer

    if(chosen_model == 'distilroberta'):
        tokenizer = transformers.AutoTokenizer.from_pretrained('blancasavi/pretrained_light')
        model = transformers.AutoModelForSequenceClassification.from_pretrained("blancasavi/pretrained_light",num_labels=len(encoder.classes_), id2label=id2label, label2id=label2id).cuda()
        

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained('blancasavi/bbase_sandy')
        model = transformers.AutoModelForSequenceClassification.from_pretrained("blancasavi/bbase_sandy",num_labels=len(encoder.classes_), id2label=id2label, label2id=label2id).cuda()
        

    dataset = dataset_preprocess()
    shap_values = shap_calculator(model, dataset)
    filename = f'{chosen_model}_shapvals'
    path=f'./shapval/{filename}'
    joblib.dump(shap_values, filename=path, compress=('bz2', 9))
    

if __name__ == "__main__":
    main()

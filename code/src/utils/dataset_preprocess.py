import transformers
import argparse
from datasets import load_dataset
from collections import Counter
from transformers import AutoTokenizer

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


def preprocess():
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


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset')
    parser.add_argument('tokenizer_mode', type=str, nargs='?', default='basic', help='Tokenizer, either "basic or expanded"')
    args = parser.parse_args()
    tokenizer_mode = args.tokenizer_mode
    global tokenizer
    
    if(tokenizer_mode == "basic"):
        tokenizer = transformers.AutoTokenizer.from_pretrained("badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification")
        fixed_dataset = preprocess()
        fixed_dataset.save_to_disk('../../saved_datasets/dataset_berta_basictokenizer')

        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        fixed_dataset = preprocess()
        fixed_dataset.save_to_disk('../../saved_datasets/fixed_dataset_bb_basetokenizer')

    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained('blancasavi/pretrained_light')
        fixed_dataset = preprocess()
        fixed_dataset.save_to_disk('../../saved_datasets/dataset_light')
        
        tokenizer = transformers.BertTokenizer.from_pretrained('blancasavi/bbase_sandy')
        fixed_dataset = preprocess()
        fixed_dataset.save_to_disk('../../saved_datasets/dataset_sandy')
        



        
    
    

if __name__ == "__main__":
    main()  
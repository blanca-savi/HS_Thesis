
import transformers
import os
from dataset_preprocess import preprocess
from datasets import load_dataset, load_from_disk
from tqdm import tqdm 


def word_compare():
    """
    Compares tokenization results across three different datasets: `dataset2`, `dataset1`, and `dataset_berta`.
    "Dataset1" is the original HateXplain dataset, "dataset2" has been tokenized by the BERT-base tokenizer
    and dataset_berta by the tokenizer "badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification".
    The function identifies and collects new words by comparing tokenized outputs between the datasets. 

    Returns:
        set: A set of newly identified words formed from tokenization discrepancies between datasets.
    """

    splits = ["train", "validation", "test"]
    for split in splits:
        for i in tqdm(range(len(dataset2[split]['post_tokens'])), desc=f'Processing {split}'):
            sentence = dataset2[split]['post_tokens'][i] 
            tokens = tokenizer.tokenize(sentence)
            
            manual_tokens = dataset1[split]["post_tokens"][i]
            construct_token = ''
            
            for token in tokens:
                token = token.replace('Ġ', '')
                if token not in manual_tokens:
                    construct_token += token
                    if(construct_token in manual_tokens):
                        new_words.add(construct_token) 
                        construct_token = ''
                        
    return new_words
                
def save_list(new_words):
    """
    This function appends a list of new words to a file called 'vocabulary_expansion.txt'.
    
    Parameters:
    new_words: A set containing the words to be added to the file.

    """
    with open('../vocabulary_expansion.txt', 'w') as f_out:
        final_list = list(new_words)
        f_out.write("\n".join(final_list) + "\n")
            
            
            
def main():
    dataset1 = load_dataset("hatexplain")
    
    
    dataset2_path= os.path.join('..', 'saved_datasets', 'dataset_berta_basictokenizer')
    dataset2= load_from_disk(dataset2_path)

    tokenizer = transformers.AutoTokenizer.from_pretrained("badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification")
    new_words = set()
    new_list = word_compare()
    save_list(new_list)
    
            
if __name__ == "__main__":
    main()
    
import nltk
import pandas as pd

from datasets import load_dataset
from collections import Counter 
from nltk.corpus import stopwords

def common_retreiver(number_words):
    """
    Retrieves the most frequent words from the test split within the 'hatexplain' dataset. Excludes stopwords. 
    "Would"  and "get" are manually included into the list of stopwords.
    
    Parameters:
        number_words (int): The number of most frequent words to be retreived.
    
    Returns:
        list: A list of tuples where each tuple contains a word and its count, sorted by frequency in descending order.
    """
    nltk.download('stopwords')
    stop = stopwords.words('english')
    stop.extend(["would", "get"]) 


    dataset = load_dataset("hatexplain")["test"]
    data = pd.DataFrame({"text": dataset["post_tokens"]})

    data["text"]=data["text"].apply(lambda x:  ' '.join([word for word in x if word not in (stop) and not word.startswith('<')]))

    most_common=Counter(" ".join(data["text"]).split()).most_common(number_words) 
    return most_common

def main():
    number_words = 15
    most_common = common_retreiver(number_words)
    print(most_common)

if __name__ == "__main__":
    main()




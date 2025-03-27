import numpy as np
import transformers
import torch
import argparse

from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay 
    

def masking_function(examples):  
  """
  Replaces a specific token in the input dataset with a mask token.

  Parameters:
  examples (dict): A dictionary containing a key "input_ids", where the value is a list of lists
                    representing tokenized input sequences.

  Returns:
  dict: The updated dictionary with masked tokens in "input_ids".
  """
  examples["input_ids"] = [
     # Replace token_id with mask_token_id if it matches word_encoded
    [mask_token_id[0] if token_id == word_encoded[0] else token_id for token_id in sublist]
    for sublist in examples["input_ids"]
  ]

  return examples


def evaluation(row): 
    """
    Evaluates a single input sample using the model in evaluation mode.
    
    Parameters:
        row (dict): A dictionary containing:
            - "input_ids": Tokenized input sequence (list).
            - "attention_mask": Attention mask for the input sequence (list).
            - "final_label": The ground-truth label for the input (list).
    
    Returns:
        float: The softmax probability of the specific label (e.g., hate speech label).
    """
    model.eval()
    
    with torch.no_grad():
        input_sample = row["input_ids"]
        attention_mask = row['attention_mask']
        label = row['final_label']
        
        input_sample_tensor = torch.tensor(input_sample).squeeze() 
        attention_mask_tensor = torch.tensor(attention_mask).squeeze()  
        label_tensor = torch.tensor(label).squeeze()        
    
        input_sample_tensor = input_sample_tensor.to(device)
        attention_mask_tensor = attention_mask_tensor.to(device)
        label_tensor = label_tensor.to(device)
        
        outputs = model(input_ids=input_sample_tensor.unsqueeze(0), labels=label_tensor.unsqueeze(0), attention_mask=attention_mask_tensor.unsqueeze(0))
        
        logits = outputs.logits 
    
        softmax_logits = torch.nn.functional.softmax(logits, dim=1)
        value = float(softmax_logits[0][class_index])
        
    return value
  

def bias_calc():
  """
  Calculates the average bias for a specific word in the dataset by masking the word
  and comparing model predictions before and after masking. Based on the paper
  "Bias Mitigation in Misogynous Meme Recognition: A Preliminary Study”
  (https://ceur-ws.org/Vol-3596/paper7.pdf) .
  
  Parameters:
    class_index (int): the index of the analysed class
  
  Steps:
      1. Encodes the target word to its token ID using the tokenizer.
      2. Filters the dataset to include only rows containing the target word.
      3. Masks occurrences of the target word in the filtered dataset.
      4. Computes the difference in model predictions between the original and masked datasets.
      5. Averages the differences across all filtered rows.

  Returns:
      float: The average bias value for the target word. 
  """


  global word_encoded
  word_encoded = tokenizer.encode(word_to_mask, add_special_tokens=False)
  
  filtered_dataset = dataset.filter(
      lambda x: (
          any(token == word_encoded[0] for token in x['input_ids'])
      )
  )

  M = len(filtered_dataset)
  print(f'Number of rows in purged dataset: {M}')

  global  mask_token_id
  mask_token_id = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
  dataset_masked = filtered_dataset.map(masking_function, batched=True)

  sumatorio = 0
  for i in range(M):
      sumatorio += (evaluation(filtered_dataset[i])-evaluation(dataset_masked[i]))

  if(M!=0):
    result = sumatorio / M

  return result 
  


def main():
  parser = argparse.ArgumentParser(description='Mask a word in the dataset.')
  parser.add_argument('word', type=str, nargs='?', default='kike', help='Word to mask within the dataset')
  parser.add_argument('model', type=str, nargs='?', default='bbase', help='Model to use, either bbase or distiroberta')
  parser.add_argument('class_label', type=str, nargs='?', default='hs', help='Either "hs", "off" or "nor"')
  args = parser.parse_args()
  global word_to_mask
  word_to_mask = args.word
  chosen_model = args.model
  class_label = args.class_label
  
  global class_index
  classes = ["hs", "nor", "off"]
  class_index = classes.index(class_label)
  print(f' class_index : {class_index}')
  

  encoder = LabelEncoder()
  encoder.classes_ = np.load('/home/blanca/Documents/Thesis_Code/Dataset/classes.npy',allow_pickle=True)
  id2label = {idx: label for idx, label in enumerate(encoder.classes_)}
  label2id = {label: idx for idx, label in enumerate(encoder.classes_)}

  global device, tokenizer, model, loss_fn, dataset
  
  if(chosen_model == 'distilroberta'):
    tokenizer = transformers.AutoTokenizer.from_pretrained('blancasavi/pretrained_light')
    model = transformers.AutoModelForSequenceClassification.from_pretrained("blancasavi/pretrained_light",num_labels=len(encoder.classes_), id2label=id2label, label2id=label2id).cuda()
    dataset = load_from_disk('/home/blanca/Documents/Thesis_Code/code/saved_datasets/dataset_light')

  else:
    tokenizer = transformers.BertTokenizer.from_pretrained('blancasavi/bbase_sandy')
    model = transformers.AutoModelForSequenceClassification.from_pretrained("blancasavi/bbase_sandy",num_labels=len(encoder.classes_), id2label=id2label, label2id=label2id).cuda()
    dataset = load_from_disk('/home/blanca/Documents/Thesis_Code/code/saved_datasets/dataset_sandy')
     

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  loss_fn = torch.nn.CrossEntropyLoss() 
  dataset = dataset["test"]

  bias_value = bias_calc()
  print(f'RESULT = {bias_value}')

if __name__ == "__main__":
    main()
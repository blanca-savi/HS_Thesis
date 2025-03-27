import os
import numpy as np
import transformers
import torch
import argparse


from torch.utils.data import DataLoader
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

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
    [mask_token_id[0] if token_id == word_encoded[0] else token_id for token_id in sublist]
    for sublist in examples["input_ids"]
  ]

  return examples    


def collate_pad(data_collate):
  """
    Collates and pads a batch of data. 

    Parameters:
    - data_collate (list of dict): A batch of examples where each example is 
      a dictionary containing 'input_ids', 'attention_mask', 'token_type_ids', 
      and 'final_label'.

    Returns:
    - dict: A dictionary containing padded tensors:
      - 'input_ids': Padded tensor of input token IDs.
      - 'attention_mask': Padded tensor for attention masking.
      - 'token_type_ids': Padded tensor for token type IDs.
      - 'labels': Padded tensor of labels.
  """
  input_ids_batch = []
  attention_mask_batch = []
  labels_batch = []
  for i in range(len(data_collate)):
    input_ids_batch.append(torch.tensor(data_collate[i]['input_ids']).squeeze())
    attention_mask_batch.append(torch.tensor(data_collate[i]['attention_mask']).squeeze())
    labels_batch.append(torch.tensor(data_collate[i]['final_label']).unsqueeze(0)) 

  input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=0)
  attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)
  labels_batch = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True)

  return{
    'input_ids': input_ids_batch,
    'attention_mask': attention_mask_batch,
    'labels': labels_batch       
} 


def evaluation(loader, word_to_mask, chosen_model): 
  """
  Evaluates the performance of the model on the dataset with the "word_to_mask" masked.
  Calculates and saves the confusion matrix as a .npy.

  Parameters:
  - loader (DataLoader): The DataLoader containing the data.
  - word_to_mask (str): The word masked in the dataset, used for naming the confusion matrix file.
  - chosen_model (str): The name of the model being evaluated (either "bbase" or "distilroberta").

  Returns:
  - None
  """

  model.eval()
  epoch_test_loss=0
  test_all_labels = []
  test_all_predictions = []
  with torch.no_grad():
      for i, batch in enumerate(loader):
          inputs = batch['input_ids']
          attention_mask = batch['attention_mask']
          labels = batch['labels']
          inputs, attention_mask, labels = inputs.to(device), attention_mask.to(device), labels.to(device)

          outputs = model(input_ids = inputs, labels=labels, attention_mask=attention_mask)
          
          logits = outputs.logits

          loss = loss_fn(logits, labels.squeeze())
          epoch_test_loss += loss.item()

          test_all_labels.extend(labels.cpu().numpy())
          test_all_predictions.extend(logits.argmax(1).cpu().numpy())



  conf_matrix = confusion_matrix(test_all_labels, test_all_predictions)
  filename = f'{word_to_mask}_confmtx.npy'
  path = f'./ConfMtxes/{chosen_model}/{filename}'
  np.save(path, conf_matrix)

  




def main():
  parser = argparse.ArgumentParser(description='Mask a word in the dataset.')
  parser.add_argument('word', type=str, nargs='?', default='kike', help='Word to mask within the dataset')
  parser.add_argument('model', type=str, nargs='?', default='bbase', help='Model to use, either bbase or distiroberta')
  args = parser.parse_args()
  word_to_mask = args.word
  chosen_model = args.model

  encoder = LabelEncoder()
  encoder_path = os.path.join('..' , '..', 'Dataset', 'classes.npy')
  encoder.classes_ = np.load(encoder_path,allow_pickle=True)
  id2label = {idx: label for idx, label in enumerate(encoder.classes_)}
  label2id = {label: idx for idx, label in enumerate(encoder.classes_)}

  global device, tokenizer, model, loss_fn, dataset
  if(chosen_model == 'distilroberta'):
    tokenizer = transformers.AutoTokenizer.from_pretrained('blancasavi/pretrained_light')
    model = transformers.AutoModelForSequenceClassification.from_pretrained("blancasavi/pretrained_light",num_labels=len(encoder.classes_), id2label=id2label, label2id=label2id).cuda()
    dataset_path = os.path.join('..', 'saved_datasets', 'dataset_light')
    dataset = load_from_disk(dataset_path)

  else:
    tokenizer = transformers.BertTokenizer.from_pretrained('blancasavi/bbase_sandy')
    model = transformers.AutoModelForSequenceClassification.from_pretrained("blancasavi/bbase_sandy",num_labels=len(encoder.classes_), id2label=id2label, label2id=label2id).cuda()
    dataset_path = os.path.join('..', 'saved_datasets', 'dataset_sandy')
    dataset = load_from_disk(dataset_path)
    

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  loss_fn = torch.nn.CrossEntropyLoss() 
  dataset = dataset["test"]

  
  global word_encoded, mask_token_id
  word_encoded = tokenizer.encode(word_to_mask, add_special_tokens=False)
  mask_token_id = tokenizer.encode(tokenizer.mask_token, add_special_tokens=False)
  dataset = dataset.map(masking_function, batched=True)
  batch_size=8
  loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_pad)
  evaluation(loader, word_to_mask, chosen_model)



if __name__ == "__main__":
    main()
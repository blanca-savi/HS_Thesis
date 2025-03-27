# ctrl+shift+p - python: select interpreter
#wandb agent eule/FT2_SWEEP/old1fw3f
#wandb sweep sweep.yaml --project 'FT2_SWEEP'

import os
import numpy as np
import torch
import transformers
import wandb
import argparse


from datasets import load_dataset 
from types import SimpleNamespace
from transformers import BertTokenizer, BertForSequenceClassification 
from transformers.tokenization_utils import AddedToken
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score 

from torch.utils.data import DataLoader 
from collections import Counter


params = SimpleNamespace(
    lr=2.2865731388653697e-05,
    epochs=3,
    batch_size=22,
    wandb_project='BBASE_FINAL'
)


def parse_args():
    """
    This function parses the arguments passed to the script.

    Command-line arguments:
    - --lr: Learning rate (float), default value is set to the value from `params.lr`.
    - --epochs: Number of training epochs (int), default value is set to the value from `params.epochs`.
    - --batch_size: Size of each batch for training (int), default value is set to the value from `params.batch_size`.
    - --wandb_project: Name of the WandB project (str), default value is set to the value from `params.wandb_project`.

    Returns:
    - The parsed arguments (Namespace).
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=params.lr)
    parser.add_argument('--epochs', type=int, default=params.epochs)
    parser.add_argument('--batch_size', type=int, default=params.batch_size)
    parser.add_argument('--wandb_project', type=str, default=params.wandb_project)
    return parser.parse_args()


def wandb_init(model, config):
    """
    This function initializes the Weights and Biases (WandB) logging.
    
    Parameters:
    - model: The PyTorch model that will be tracked during training.
    - config: Contains the settings for the experiment defined by the parser (default = params).
    """

    wandb.init(config.wandb_project, config=config)
    config=wandb.config
    wandb.watch(model)


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
    
    

def load_custom_tokens(file_path):    
    """
    Loads a list of words from a text file.

    Parameters:
    - file_path (str): The path to the text file containing the words. 

    Returns:
    - tokens (list of str): A list containing all the tokens loaded from the file.
    """
    with open(file_path, 'r') as file:
        tokens = [line.strip() for line in file.readlines()]
    return tokens   
    
def expand_vocabulary():
    """
    Loads custom tokens from a file and adds them to the tokenizer.

    """
    custom_tokens_file = os.path.join('./vocabulary_expansion.txt')
    custom_tokens = load_custom_tokens(custom_tokens_file)
    
    added_tokens = [AddedToken(token, single_word=True) for token in custom_tokens]
    tokenizer.add_tokens(added_tokens)

    model.resize_token_embeddings(len(tokenizer))


def model_def():
    """
    Defines and initializes the BERT-base model for sequence classification.

    Returns:
    - model (BertForSequenceClassification): The initialized BERT model.
    """

    encoder = LabelEncoder()
    encoder_path = os.path.join('..' , '..', 'Dataset', 'classes.npy')
    encoder.classes_ = np.load(encoder_path,allow_pickle=True)

    
    id2label = {idx: label for idx, label in enumerate(encoder.classes_)}
    label2id = {label: idx for idx, label in enumerate(encoder.classes_)}

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(encoder.classes_), id2label=id2label, label2id=label2id
    ).to(device)
    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs available:", torch.cuda.device_count())
    return model


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
  token_type_ids_batch = []
  labels_batch = []
  
  for i in range(len(data_collate)):

    input_ids_batch.append(torch.tensor(data_collate[i]['input_ids']).squeeze())
    attention_mask_batch.append(torch.tensor(data_collate[i]['attention_mask']).squeeze())
    token_type_ids_batch.append(torch.tensor(data_collate[i]['token_type_ids']).squeeze())
    labels_batch.append(torch.tensor(data_collate[i]['final_label']).unsqueeze(0)) 
  
  input_ids_batch = torch.nn.utils.rnn.pad_sequence(input_ids_batch, batch_first=True, padding_value=0)
  attention_mask_batch = torch.nn.utils.rnn.pad_sequence(attention_mask_batch, batch_first=True, padding_value=0)
  token_type_ids_batch = torch.nn.utils.rnn.pad_sequence(token_type_ids_batch, batch_first=True, padding_value=0)
  labels_batch = torch.nn.utils.rnn.pad_sequence(labels_batch, batch_first=True)
 
  return{
    'input_ids': input_ids_batch,
    'attention_mask': attention_mask_batch,
    'token_type_ids': token_type_ids_batch,
    'labels': labels_batch       
} 
    

def metrics_log(all_labels, all_predictions, epoch_loss, len_loader, epoch, mode):
    """
    Calculates and logs (into Weights&Biases) evaluation metrics for a given epoch.

    """

    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    accuracy = accuracy_score(all_labels, all_predictions)

    epoch_loss /= len_loader 

    wandb.log({f"{mode}/loss": epoch_loss, 
               f"{mode}/accuracy": accuracy,
               f"{mode}/precision": precision,
               f"{mode}/recall": recall,
               f"{mode}/f1": f1,
               "epoch": epoch + 1})
    
    
def fine_tune(config=params):
    """
    Trains and evaluates the BERT-base model over multiple epochs. 
    Evaluation on the test split occurs after all epochs.
    Metrics are logged into Weights&Biases in the process.

    Parameters:
    - config (dict): Configuration parameters (lr, number of epochs, batch_size, wandb_project)
    """
    

    data_train = fixed_dataset['train']
    data_eval = fixed_dataset['validation']
    data_test = fixed_dataset['test']
    optimizer = AdamW(model.parameters(), config.lr) 
   
    train_loader = DataLoader(dataset=data_train, shuffle=True, batch_size=config.batch_size,collate_fn=collate_pad)
    eval_loader = DataLoader(dataset=data_eval, shuffle=True, batch_size=config.batch_size,collate_fn=collate_pad)
    test_loader = DataLoader(dataset=data_test, shuffle=True, batch_size=config.batch_size,collate_fn=collate_pad)


    for epoch in range(config.epochs):
        print("Epoch: ",(epoch + 1))
        
        model.train()    
        epoch_train_loss = 0
        train_all_labels = []
        train_all_predictions = []
        for i,batch in enumerate(train_loader):  
          
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            labels = batch['labels']
            inputs, attention_mask, token_type_ids, labels = inputs.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
            
        
            optimizer.zero_grad()
           
            outputs = model(input_ids = inputs, labels=labels, attention_mask=attention_mask)
            
            logits = outputs.logits
            
            loss = loss_fn(logits, labels.squeeze())

            
            loss.backward()
            
            optimizer.step()

            train_batch_loss = loss.item()
            epoch_train_loss += train_batch_loss 
            train_last_loss = train_batch_loss / config.batch_size 
            print('Training batch {} last loss: {}'.format(i + 1, train_last_loss))   
            
            train_all_labels.extend(labels.cpu().numpy())
            train_all_predictions.extend(logits.argmax(1).cpu().numpy())
        
            
        metrics_log(train_all_labels, train_all_predictions, epoch_train_loss, len(train_loader), epoch, "train")


        #VALIDATION BLOCK
        
        model.eval() 
        epoch_val_loss=0
        all_labels = []
        all_predictions = []
        
        
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                inputs = batch['input_ids']
                attention_mask = batch['attention_mask']
                token_type_ids = batch['token_type_ids']
                labels = batch['labels']
                inputs, attention_mask, token_type_ids, labels = inputs.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
                
                outputs = model(input_ids = inputs, labels=labels, attention_mask=attention_mask)
                
                logits = outputs.logits
                
                loss = loss_fn(logits, labels.squeeze())
                epoch_val_loss += loss.item() 

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(logits.argmax(1).cpu().numpy())

        
        metrics_log(all_labels, all_predictions, epoch_val_loss, len(eval_loader), epoch, "validation")
   
    
    model.eval() 
    epoch_test_loss=0
    test_all_labels = []
    test_all_predictions = []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            token_type_ids = batch['token_type_ids']
            labels = batch['labels']
            inputs, attention_mask, token_type_ids, labels = inputs.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
            
            
            
            outputs = model(input_ids = inputs, labels=labels, attention_mask=attention_mask)
            
            
            logits = outputs.logits
            
            
            loss = loss_fn(logits, labels.squeeze())
            epoch_test_loss += loss.item()
            
            test_all_labels.extend(labels.cpu().numpy())
            test_all_predictions.extend(logits.argmax(1).cpu().numpy())
            
    metrics_log(test_all_labels, test_all_predictions, epoch_test_loss, len(test_loader), epoch, "test")





def main():
    args = parse_args()  
    
    global device, model, tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_def()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    expand_vocabulary()

    global fixed_dataset, loss_fn
    fixed_dataset = dataset_preprocess()
    loss_fn = torch.nn.CrossEntropyLoss() 
    
    wandb_init(model, config=args)
    run_name = wandb.run.name
    print(run_name)
    fine_tune(config=args)

    
    base_path = os.path.abspath(os.path.join(os.getcwd(), "../models"))
    full_path = os.path.join(base_path, run_name)
    model.save_pretrained(full_path)   
    tokenizer.save_pretrained(full_path)

    wandb.finish()



if __name__ == "__main__":
    main()  
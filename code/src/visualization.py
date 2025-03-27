import matplotlib.pyplot as plt
import os
import re
import argparse
import numpy as np


lists_bbase = [
    "common",
    "top_nor_bbase",
    "top_hs_bbase",
    "top_off_bbase",
    "top_nor_bbase", 
    "all_influential_bbase",     
    "intersection_bbase",
    "black_target_bbase",
    "jewish_target_bbase",
    "disabilities_target_bbase",
    "woman_target_bbase",
    "hispanic_target_bbase",
    "muslim_target_bbase",
    "top_nor_pos_bbase",
    "black_target_no_nigger_bbase"
    ]

lists_roberta = [
    "common",
    "top_nor_roberta",
    "top_hs_roberta",
    "top_off_roberta",
    "all_influential_roberta",
    "intersection_roberta",
    "black_target_roberta",
    "hispanic_target_roberta",
    "jewish_target_roberta",
    "muslim_target_roberta",
    "disabilities_target_roberta",
    "lgtb_target_roberta",
    "middle_eastern_target_roberta",
    "top_nor_pos_roberta",
    "black_target_no_nigger_roberta"
]


def f1_extractor():
    """
    Reads `.txt` files and looks for the line containing the F1 score.
    Stores F1 score values and returns them along with the names of 
    the lists/word.

    Parameters:
    - path (str): The directory path containing the `.txt` metric files.
    - mode (str): Determines how to filter the files based on their names. If 'lists', files matching 
                  the names in `lists` are included; otherwise, files not in `lists` are included.
    - model (str): The model name.
    - lists (list): A list of strings, the names of the lists.
    
    Returns:
    - f1_values (list): A list of F1 scores.
    - lists_names (list): A list of names of the files.
    """
    f1_values = []
    lists_names = []
    
    for filename in os.listdir(path):        
        if filename.endswith('.txt'):           
            file_path = os.path.join(path, filename)
            no_extension = os.path.splitext(filename)[0]
            no_extension = no_extension.replace('_metrics', '')
            
            if (mode == 'lists' and no_extension in lists) or (mode != 'lists' and no_extension not in lists):
                no_extension = no_extension.replace(f'_{model}', '')
                no_extension = no_extension.replace('_roberta', '')
                lists_names.append(no_extension)
                
                with open(file_path, 'r') as file:
                    content = file.read()
                    
                    match = re.search(r'LAST EVAL/f1:\s*([0-9.]+)', content)
                    if match:
                        f1_value = float(match.group(1))
                        f1_values.append(f1_value)
                
    return f1_values, lists_names


def line_plotter():
    """
    Generates a line plot comparing the difference in F1 scores for different masked features.
    """

    f1_values, lists_names = f1_extractor()
    f1_differences = original_f1 - np.array(f1_values)

    sorted_indices = np.argsort(-f1_differences)  
    sorted_differences = f1_differences[sorted_indices]
    sorted_names = np.array(lists_names)[sorted_indices]    

    plt.figure(figsize=(10, 5))
    plt.plot(sorted_names, sorted_differences, marker='o', linestyle='-', color='b')

    plt.xlabel('Masked Feature')
    plt.ylabel('Difference in F1 Score')
    well_written_model = 'BERT-base' if model == 'bbase' else 'DistilRoBERTa'
    plt.title(f'Difference in F1 Scores for masked {mode} in {well_written_model}')
    plt.grid(True)
    plt.xticks(rotation=90)

    plt.show()
    
    save_dir = f'./ConfMtxes/{model}/plots/'
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f'{save_dir}/{mode}_f1_differences.png', dpi=300, bbox_inches='tight')



def main():
    
    parser = argparse.ArgumentParser(description='Choose model to create the visualization.')
    parser.add_argument('chosen_model', type=str, nargs='?', default='bbase', help='Model to use, either bbase or distiroberta')
    parser.add_argument('chosen_mode', type=str, nargs='?', default='lists', help='What to portray, lists or words')
    args = parser.parse_args()
    global model, mode
    model = args.chosen_model
    mode = args.chosen_mode
    
    global original_f1, path, lists
    
    if(model == 'distilroberta'):
        original_f1 = 0.6526487837689353
        path = "./ConfMtxes/distilroberta"
        lists = set(lists_roberta)
        
    else:
        original_f1 = 0.6510900415368353
        path = "./ConfMtxes/bbase"
        lists = set(lists_bbase)
        
        
    line_plotter()
        
if __name__ == "__main__":
    main()



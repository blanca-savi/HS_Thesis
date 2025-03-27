# **Practice and philosophy in machine learning:**
### **Evaluation and critical reflection of the model-dependency of machine-learning algorithms for hate speech classification**

### **Abstract**
This study seeks to fine-tune two hate-speech detection models using the same dataset in
order to uncover intrinsic differences that may affect how just they are. It includes both a
technical analysis, which quantifies biases through an explainability method, and an ethical
analysis. The ethical analysis explores the values embedded in the models’ design, anticipates
potential negative consequences within the theoretical context of deployment, and proposes
ideal practices to reduce such issues.

### **Folder description**
- **Dataset**: This folder contains the [HateXplain](https://github.com/hate-alert/HateXplain) dataset.
- **code**: This folder contains the code files
  - **models/**: Contains fine-tuned models 
  - **results/**: Contains two jupyter notebooks to visualize the results. 
  - **saved_datasets/**: Stores processed datasets.
  - **src/**: Contains the source code.
     - **shapval**: Contains the resulting SHAP values.
     - **utils**: Contains complementing code to the four main tasks: fine-tuning, masking, list retrieval and visualization.
     - **ConfMtxes**: Contains the results after masking the word lists and each word individually. 
        - **bbase**: confusion matrices and metrics for BERT-base.
             - **plots**: plots for BERT-base after masking the word lists and each word individually.
        - **distilroberta**: confusion matrices and metrics for DistilRoBERTa.
             - **plots**: plots for DistilRoBERTa after masking the word lists and each word individually.
  - **test/**: contains a test script to calculate the SHAP values when masking the word "muslim".



### **Usage instructions**
This is the pipeline for the technical analysis within the "src" directory.
#### Environment
Environment creation:

`mamba env create -f requirements.yaml`

Environment activation:

`conda activate hs_thesis`

#### Fine-tuning
Tokenization of the HateXplain dataset, and retrieval of novel words:

`./pre_finetuning.sh`

Fine-tuning of the BERT-base model:

`python3 bert_ft_sweep.py`

Fine-tuning of the DistilRoBERTa model:

`python3 distilroberta_ft_sweep.py`

Fine-tuning with a Weights&Biases agent:

`wandb sweep sweep.yaml --project 'PROJECT_NAME'`

#### Lists retrieval
Retrieval of the list of most frequent words and the list of most influential words:

`./lists_calc.sh`

Visualization of results within shapval/SHAP_Results.ipynb

#### Masking
Masking of word lists and each word individually:

`./masking.sh`


#### Visualization
Visualization of the masking results:

`.\visualization.sh`


### **Others**
#### Bias score

The dataset has to be processed first with:

'python3 dataset_preprocess.py expanded'

Then: 

`./utils/bias_score.py [word] [model] [class]`

[word]: The word for which you want to calculate the bias score.

[model]: The model. Options are:

    "distilroberta"
    "bbase"

[class]: The class. Options are:

    "hs" (hate speech)
    "off" (offensive)
    "nor" (neutral)

#### Data distributions
Can be inspected within the jupyter notebook:  /utils/dataset_analyzer.ipynb

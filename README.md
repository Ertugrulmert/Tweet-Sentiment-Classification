# CIL 2022 Project 2: Sentiment Classification

This repository contains source code used by the team MAMLS in the 2022 Computational Intelligence Lab Project 2: Twitter Sentiment Classification.

The online version of this repository can be found at https://gitlab.ethz.ch/maml/cil-2022

Please refer to the project report "Project_Report_Sentiment_Classification" provided in the repository for further information about methodology and experiment results.

## Setup
### Downloading necessary data

* Download `checkpoints_contents.zip` from https://drive.google.com/file/d/1PqzOOAIJTyIdFJQd73_8XC5gYpwUj1wX/view?usp=sharing and extract the contents of the archive into the `checkpoints` directory
* Download `twitter-datasets_contents.zip` from https://drive.google.com/file/d/15MMDViAHEB_W8PycIoamKqvBq3YEUQdE/view?usp=sharing and extract the contents of the archive into the `twitter-datasets` directory

### Installing necessary dependencies
For this step we recommend being in a virtual environment running python `>=3.8.5`

```
pip install -r requirements.txt
```

## Quick Start

To train the model our group submitted and have it output the submission file, simply run the following command:

```
python3 scripts/run.py stats_bert
```

Since training the model is time-consuming (13 hours on the Euler cluster), to just run the model checkpoint that we have provided, change the parameter `LoadCheckpoint` to `True` in the `models/BERT/config.ini` file and run the following command:
```
python3 scripts/run.py stats_bert --only_predict
```

## Running the code on Euler

* Set up the environment

```
module load gcc/8.2.0 python_gpu/3.8.5 eth_proxy
pip install virtualenvwrapper
source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv cil_mamls
```
* Transfer the repository to the euler cluster and set up all of the required files as outlined above

* Install requirements

```
pip install -r requirements.txt
```

* Run the code

To be able to run the code, the command needs to be submitted as a batch job on Euler. We recommend requesting gpu access when running the code.

Example usage:

```
bsub -n 8 -W 16:00 -R "rusage[mem=8192, ngpus_excl_p=1]" python scripts/run.py stats_bert
```

## Reproducing all experiments

To run any experiment, it is needed to run the `scripts/run.py` script. The script takes two input arguments. 
1. The first is a positional argument which determines the model type. The possible values of the argument are:
    * `mlp` - to run a purely MLP-based classification model - config file is located in the `models/MLP` directory
    * `svm` - to run an SVM-based classification model - config file is located in the `models/SVM` directory
    * `bilstm` - to run a BiLSTM-based classification model - config file is located in the `models/BiLSTM` directory
    * `bert` - to run a BERT-based classification model using an MLP as its classification head - config file is located in the `models/BERT` directory
    * `bilstm_bert` - to run a BERT-based classification model using a BiLSTM as its classification head - config file is located in the `models/BERT` directory
    * `stats_bert` - to run a BERT-based classification model using a BiLSTM as its classification head that additionally takes statistical, lexical and topic features - config file is located in the `models/BERT` directory
2. The second argument is an optional argument given as `--only_predict`. If it is given, the model will not be trained before generating predictions for the test data. If this argument is given without there being a model checkpoint to initialize the model from, the predictions will be generated using a model with randomly initialized parameters.

### Example usage

If we want to run a BiLSTM-based model without training, we can execute the following command:

```
python3 scripts/run.py bilstm --only_predict
```

### Changing hyperparameters and training and testing environment

To change any hyperparameters or configuration of the training and testing environment (e.g. learning rate, batch size, number of convolutional layers, location of the checkpoint file...), `config.ini` files corresponding to the specific model type need to be changed. They are located in the `models/<model_type>` directory.
For example, if we want to change the learning rate used while training a BiLSTM-based model, in the `models/BiLSTM/config.ini` file, we change the value of the `LR` parameter. All of the `config.ini` files are provided with the configuration of the best-performing model in the category.

## Repository Structure
* `scripts` - Scripts to run the code as an end user
* `twitter-datasets/data` - Datasets on which the code can be run
    * `final_test_data.txt`, `final_train_pos_full.txt` and `final_train_neg_full.txt` are the version of the dataset with all of the preprocessing methods applied
    * `final_test_data.txt`, `final_train_pos_full.txt` and `final_train_neg_full.txt` are the unprocessed versions of the datasets
    * `test_stats.csv`, `pos_stats.csv` and `neg_stats.csv` are precomputed statistical features corresponding to each of the datasets
* `twitter-datasets/glove_twitter` - Where Glove embedding files are download and Fasttext embedding files saved after training
* `models` - Source code of the models and utility files used for experiments and final solution. Every subdirectory contains a `config.ini` file. When running the model, they define model hyperparameters and the configuration of the training environment. If a different configuration is needed, the parameters in the `config.ini` files need to be changed
    * `SVM` - Source code for the SVM-based baseline
    * `MLP` - Source code for the MLP-based baselines
    * `BiLSTM` - Source code for the BiLSTM models used in our experimentation
    * `BERT` - Source code for models that use BERT transformers
* `preprocessing` - Source code for preprocessing the datasets
* `submissions` - Submission files generaed by the user will be saved here
* `checkpoints` - Model checkpoints generated during training will be saved here
    * `lda` - Latent Dirichlet Allocation topic model files will be saved here
    * `biLSTM` - biLSTM based model files will be saved here
    * `tokenizer` - Tokenizer files will be saved here
    * `param_search_logs` - Parameter search results for BiLSTM based models will be saved here
    * `RoBERTa_stats_final.pth` - Checkpoint with model weights used to generate our team's final submission to the Kaggle leaderboard

## Preprocessing datasets

We provide the fully preprocessed and unpreprocessed versions of our dataset, but further variations of the dataset can be generated with any combination of the preprocessing steps.

Generating the preprocessed datasets has to be done manually within code, but the notebook `Preprocessing.ipynb` outlines the example usage.

Sequence-to-sequence text normalization is implemented separately from other preprocessing steps because it relies on third-party code. This preprocessing step can be done through code, but it relies on **having docker installed and running on the device running the code**. Additionally, the internal workings of the function create a docker container running the sequence-to-sequence model and make API calls to it. This can be very slow so we recommend following the steps in `preprocessing/seq2seq/README.md` to apply sequence-to-sequence normalization to another dataset.

## Authors

Team name: MAMLS

Anton Alexandrov, Moritz Dück, Mert Ertugrul, Lovro Rabuzin

D-INFK, ETH Zurich

# Modeling and Analyzing the Influence of Non-Item Pages on Sequential Next-Item Prediction
This repository contains the code for our paper: 

Modeling and Analyzing the Influence of Non-Item Pages on Sequential Next-Item Prediction (available on [arXiv](https://arxiv.org/abs/2408.15953))

The code is based on the RecBole framework (https://recbole.io/).

## Running the Code and the Experiments
Here, you will find instructions on how to run the code and the experiments from the paper.

## 1. Installation
Ensure you have **Python >=3.9 and <3.11** installed on your system.

1. Clone the repository `git clone https://github.com/LSX-UniWue/non-items-recbole.git`
2. Navigate into the project directory
   `cd non-items-recbole`
3. Create a virtual environment, e.g. with virtualenv:
   `virtualenv venv --python=python3.9`
4. Activate the virtual environment
  - On macOS/Linux:`source venv/bin/activate`
  - On Windows: `venv\Scripts\activate`
5. Install dependencies
  - Using pip:`pip install -r requirements.txt`
  - Using Poetry: `poetry install`

### Set up wandb
We rely on wandb for logging and tracking experiments and suggest using it.
You can create an account at [wandb.ai](https://wandb.ai/).
Also see the [RecBole](https://recbole.io/docs/user_guide/usage/use_weights_and_biases.html) documentation.
To login, type `export WANDB_API_KEY=<your_api_key>` and `wandb login` in your terminal.

## 2. Data Preparation
First, you will need to acquire and prepare the datasets, but we try to keep this as simple as possible.

### 2.1. Download
* MovieLens-1M and MovieLens-20M are automatically downloaded by the data_preparation script.
* The [COVEO](https://github.com/coveooss/SIGIR-ecom-data-challenge) dataset has to be downloaded manually from [here](https://www.coveo.com/en/ailabs/sigir-ecom-data-challenge), as it requires registering.
    * Don't use Safari for downloading, as you will end up missing a file.
    * Unzip with e.g. `unzip path/to/SIGIR-ecom-data-challenge.zip -d ./datasets/`
    * There should be three files in  `SIGIR-ecom-data-challenge\train`: `browsing_train.csv`, `search_train.csv`, `sku_to_content.csv`
    * Provide your dataset path in the next step
  
### 2.2. Data Preprocessing
The following script is used to preprocess a dataset (ml-1m by default):
`python data_preparation/main.py`

##### Arguments
* `--dataset`: The dataset to preprocess. Options are `ml-1m`, `ml-20m`, `coveo`
* `--dataset_path`: The path to the dataset. Default is `.datasets/raw/`
* `--random`: Generate randomized non-item dataset for MovieLens. Default is False.
* `--use_original_split`: Use the exact same split as in the paper for the ML-20m dataset, new random split otherwise. Default is True. Coveo uses a fixed split anyway.

Note: Data was originally preprocessed in [ASME](https://github.com/elisabethfischer/non-item-transformers/tree/icdm23-non-items) framework.

#### Commands to recreate the datasets used in the paper 
This might take some time - especially Coveo-Pageview.
1. `python data_preparation/main.py --dataset ml-20m --random True` - SynData, GroupedSynData and Randomized SynData
2. `python data_preparation/main.py --dataset coveo-search --dataset_path .datasets/coveo/` - Coveo Search Dataset, provide your path to the raw data
3. `python data_preparation/main.py --dataset coveo-pageview --dataset_path .datasets/coveo/` - Coveo Pageview Dataset, provide your path to the raw data

Note: The datasets are saved in the `final` folder in the dataset_path.

### 3. Model Configurations
Generate and adjust the model configurations to your needs with the [generate_configs.py](configs/paper/create_configs.py) script.
Usage (with default arguments set for a local, non-gpu run):

`python configs/paper/create_configs.py`

This will automatically create the model configurations for all experiments in the [paper config folder](configs/paper/).

#### Arguments
* `--log_wandb`: Log to wandb. Default is True.
* `--wandb_entity`: Your wandb entity. Default is None and therefore your default entity.
* `--data_path`: The path to save the datasets. Default is `./datasets/`
* `--checkpoint_dir`: The checkpoint directory. Default is `./checkpoint/`
* `--use_gpu`: Use GPU. Default is False.
* `--gpu_id`: The GPU ID to use. Default is None.
* `--train_batch_size`: The batch size for training. Default is 64.

### 4. Running Experiments
Finally, to train the models, you can use the `run_recbole.py` script. Type

`python run_recbole.py --config_files 'path_to_config'`

to train a model with the specified configuration file. RecBole will evaluate the model on the test set after training.

## Evaluation (per user)
If you want to evaluate a model again, you can use the `run_eval.py` script.
`run_eval.py --model_file /path/to/your/saved/modelfile.pth --device your_device`
#### Arguments
* `--model_file`: Path to the saved model
* `--device`: The device you're using, e.g. `cuda` or `cpu`
* `--eval_per_user`: Calculate the metrics for each user on the test set separately, e.g. for user-level significance testing. Default is False.

## Significance Testing
We evaluate the significance of the results using the paired Student's t-test on user level.

You will need to evaluate the models on user level as described above with 
`run_eval.py --model_file /path/to/your/saved/modelfile.pth --device your_device --eval_per_user True`.
 This will log the per-user metrics as an artifact to wandb.
Use [this notebook](notebooks/significance_testing.ipynb) to calculate the significance of the results.

## HypTrails 
A notebook to conduct the HypTrails analysis can be found in the [notebooks](notebooks) folder.


## Additional Visualizations for the Paper
We provide additional plots for models on both Coveo datasets and the SynData dataset in the [visualizations](visualizations) folder. 
These include t-SNE plots of the item embedding spaces and the differences in item-to-item cosine similarity between 
different embeddings/models. See the paper for more explanations. 
Generally, all plots are based on the models trained with seed 212, unless marked otherwise.
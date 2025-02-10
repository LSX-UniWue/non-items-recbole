
This repository contains the code for our paper: 
* Modeling and Analyzing the Influence of Non-Item Pages on Sequential Next-Item Prediction (available on [arXiv](https://arxiv.org/abs/2408.15953))

It is based on the RecBole framework (https://recbole.io/).

## Installation
* Note: requires Python >3.9 and <3.11
* we suggest you create an environment for this project using virtualenv (or another tool like conda)

First checkout this repository, then enter in the repository folder and run this commands to create and activate a new environment:
* Install the requirements by running `pip install -r requirements.txt`


## Data Preparation

All files necessary for the data preparation are in the data_preparation folder.

### Download
* MovieLens-1M and MovieLens-20M are automatically downloaded
* The [COVEO](https://github.com/coveooss/SIGIR-ecom-data-challenge) dataset has to be downloaded manually from [here](https://www.coveo.com/en/ailabs/sigir-ecom-data-challenge)
  * Move the downloaded dataset to `datasets/raw/`
  
### Preprocessing

`python data_preparation/data_preparation.py`

  * We provide a notebook for each dataset in data_preparation (Data originally preprocessed with https://github.com/elisabethfischer/non-item-transformers/tree/icdm23-non-items)
  * data_preparation/convert_for_recbole.py converts the data to the RecBole format
* Configs are provided in configs/paper/ for the experiments in the paper. 
* You can run them with `python run_recbole.py --model='model_name' --dataset='dataset_name' --config_files 'path_to_config'`
* a notebook for the hyptrails experiments is provided in notebooks/

### Non-Item Models
Our non-item models can be found in recbole/model/sequential_attribute_recommender.
- BERT4RecAttr
- CaserAttr
- GRU4RecAttr
- COREAttr
- SASRecAttr
- LightSANsAttr
- NextItNetAttr
- NARMAttr
This repository contains the code for our paper "Modeling and Analyzing the Influence of Non-Item Pages on Sequential Next-Item Prediction".
It is based on the RecBole framework (https://recbole.io/).


## Usage
* Install the requirements by running `pip install -r requirements.txt`
* To preprare the data, you will need to download them from 
  * https://grouplens.org/datasets/movielens/20m/ for the MovieLens-20M dataset and
  * https://github.com/coveooss/SIGIR-ecom-data-challenge for the Coveo dataset.
  * We provide a notebook for each dataset in data_preparation (Data originally preprocessed with https://github.com/elisabethfischer/non-item-transformers/tree/icdm23-non-items)
  * data_preparation/convert_for_recbole.py converts the data to the RecBole format
* Configs are provided in configs/paper/ for the experiments in the paper. 
* You can run them with `python run_recbole.py --model='model_name' --dataset='dataset_name' --config_files 'path_to_config'`

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
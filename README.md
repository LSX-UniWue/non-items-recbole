This repository contains the code for our paper "Adapting Sequential Recommender Models to Content Recommendation in Chat Data using Non-Item Page-Models"
It is based on the RecBole framework (https://recbole.io/) and our previous paper "Modeling and Analyzing the Influence of Non-Item Pages on Sequential Next-Item Prediction" (see main branch).


## Usage
* Install the requirements by running `pip install -r requirements.txt`
* configs/local and dataset/ml-extended allow sample runs of the models on the ml-extended dataset.
* You can run them with `python run_recbole.py --model='model_name' --dataset='dataset_name' --config_files 'path_to_config'`
* Paper configs can be generated with the notebook in configs/paper 

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

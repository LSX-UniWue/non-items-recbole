This repository contains the code for our paper ["Adapting Sequential Recommender Models to Content Recommendation in Chat Data using Non-Item Page-Models"](https://github.com/LSX-UniWue/non-items-recbole/blob/kars-workshop-24/KARs_Workshop_RecSys_2024.pdf). 
It is based on the [RecBole](https://recbole.io/) framework and our previous work ["Modeling and Analyzing the Influence of Non-Item Pages on Sequential Next-Item Prediction"](https://arxiv.org/abs/2408.15953).

## Data
We provide three of the datasets from our paper in the final and preprocessed versions: 
- Redial-Mention/Redial-Mention-Noise
- Handball-Synth
- Twitch
  
The default variant contains messages as non-items, "tokens" contains messages split into seperate tokens and "items_only" contains only item interactions for baseline models.
* [Download](https://professor-x.de/data/KARS-2024/)


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

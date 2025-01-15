This repository contains the code for our paper "Modeling and Analyzing the Influence of Non-Item Pages on Sequential Next-Item Prediction".
It is based on the RecBole framework (https://recbole.io/).


## Usage
* Install the requirements by running `pip install -r requirements.txt`
* configs/local and dataset/ml-extended allow sample runs of the models on the ml-extended dataset.
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

### Data Augmentation Option
* train_subsequences: Build subsequences from the training data
* test_subsequences: Build subsequences from the test data.
* subsequences_end_with_items: In case of non-item models, the subsequences end with real items.
* test_empty_sequences: sequence information is removed from the test data, predictions are made on th user only
* only_train_tokens: Only use the tokens from the training data for building vocabularies and mappings! NOT FINISHED
* test_only_users_with_infos: Removes all users without information in .user from the test data

### Evaluation Options 
* eval_args:
* eval_per_item: calc metrics per item and count the occurences of items
* item_id: item_id_type : To filter non-item pages from the evaluation
* eval_sequence_len: calc metrics per sequence length up to max_sequence_len
* test_all_epochs: calc metrics on test set after each epoch

import argparse
import os

import yaml


def create_configs(working_directory, data_name, mode, ITEM_ID_FIELD, wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                   data_path, checkpoint_dir, epochs,wandb_entity, use_gpu=False, item_id_type=None, gpu_id=None, log_wandb=True,
                   train_batch_size=64, random=""):

    config_base_path = working_directory+"{}/{}/base.yaml".format(data_name,mode)
    subdir_path = working_directory+"models/"
    data_path = data_path+"final/"

    for model_file in os.listdir(subdir_path):
        if model_file.endswith(".yaml"):
            model_file_path = os.path.join(subdir_path, model_file)
            model_name= model_file.split(".")[0]
            m_name = model_name.lower()
            checkpoint_path = checkpoint_dir+"{}/{}/{}/{}/".format(data_name+random, m_name, mode, seed)
            wandb_name= "{}-{}-{}".format(data_name+random,mode,m_name)
            config_name = wandb_name+"-"+str(seed)
            final_config_path = "{}/{}/{}/{}/{}.yaml".format(working_directory,data_name+random,mode,seed,config_name)
            with open(model_file_path, 'r') as model_stream:
                model_yaml = yaml.safe_load(model_stream)

                # Open the base config file
                with open(config_base_path, 'r') as base_config_stream:
                    base_yaml = yaml.safe_load(base_config_stream)
                    base_yaml["checkpoint_dir"] = checkpoint_path
                    base_yaml["seed"] = seed
                    base_yaml["data_path"] = data_path
                    if item_id_type is not None:
                        base_yaml["eval_args"] = {'group_by': 'user', 'item_id': item_id_type, 'mode': {'test': 'full', 'valid': 'full'}, 'order': 'TO', 'split': {'RS': [0.8, 0.1, 0.1]}}
                    base_yaml["use_gpu"]= use_gpu
                    base_yaml["gpu_id"]= gpu_id
                    base_yaml["log_wandb"]= log_wandb
                    base_yaml["epochs"]= epochs
                    base_yaml["train_batch_size"]= train_batch_size
                    base_yaml["ITEM_ID_FIELD"] = ITEM_ID_FIELD
                    base_yaml["MAX_ITEM_LIST_LENGTH"] = MAX_ITEM_LIST_LENGTH
                    base_yaml["wandb_name"] = wandb_name
                    base_yaml["wandb_entity"] = wandb_entity
                    base_yaml["wandb_project"] = wandb_project
                    base_yaml["dataset"] = data_name+random
                    base_yaml = {**model_yaml, **base_yaml}
                    # Add the additional parameters for BERT4RecAttr, seq length is to compensate for RecBoles input creation scheme
                    if base_yaml["model"] == "BERT4RecAttr":
                        base_yaml["masked_training"] = True
                        base_yaml["num_of_masked_items_choice"] = "variable_min"
                        base_yaml["MAX_ITEM_LIST_LENGTH"] = base_yaml["MAX_ITEM_LIST_LENGTH"] + 1

                # make sure path exists
                os.makedirs(os.path.dirname(final_config_path), exist_ok=True)
                with open(final_config_path, 'w') as output_file:
                    yaml.dump(base_yaml, output_file, sort_keys=False)

#main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_wandb", "-lw", type=bool, default=True, help="log to wandb")
    parser.add_argument("--wandb_entity", "-we", type=str, default=None, help="wandb entity")
    parser.add_argument("--data_path", "-dp", type=str, default="./datasets/", help="path to save datasets")
    parser.add_argument("--checkpoint_dir", "-cd", type=str, default="./checkpoint/", help="checkpoint directory")
    parser.add_argument("--use_gpu", "-ug", type=bool, default=False, help="use gpu")
    parser.add_argument("--gpu_id", "-gid", type=int, default=None, help="gpu id")
    parser.add_argument("--train_batch_size", "-tbs", type=int, default=64, help="train batch size")

    args, _ = parser.parse_known_args()

    working_directory = "./configs/paper/"
    log_wandb = args.log_wandb
    wandb_entity = args.wandb_entity
    data_path = args.data_path
    checkpoint_dir = args.checkpoint_dir
    use_gpu = args.use_gpu
    gpu_id = args.gpu_id
    train_batch_size = args.train_batch_size
    seeds = [212, 6, 10, 42, 404]

    exp_name = "coveo-pageview"
    wandb_project = "plp-"+exp_name
    MAX_ITEM_LIST_LENGTH = 30
    epochs = 50
    for seed in seeds:
        for mode in ["title_id"]:
            create_configs(working_directory, "coveo", mode, "product_sku_hash", wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                   data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, None, gpu_id, log_wandb,
                   train_batch_size)
        for mode in ["title_id"]:
            create_configs(working_directory, "coveo-pageview", mode, "category_product_id", wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                           data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, "item_id_type", gpu_id, log_wandb,
                           train_batch_size)

    exp_name = "coveo-search"
    wandb_project = "plp-"+exp_name
    MAX_ITEM_LIST_LENGTH = 30
    epochs = 50
    for seed in seeds:
        for mode in ["title_id"]:
            create_configs(working_directory, "coveo-sl", mode, "product_sku_hash", wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                           data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, None, gpu_id, log_wandb,
                           train_batch_size)
        for mode in ["category_id", "firstcategory_id", "firstitem_id"]:
            id_map = {"category_id": "category_product_id", "firstcategory_id": "first_result_cat", "firstitem_id": "first_result_product"}
            ITEM_ID_FIELD = id_map[mode]
            create_configs(working_directory, "coveo-sl-search", mode, ITEM_ID_FIELD, wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                           data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, "item_id_type", gpu_id, log_wandb,
                           train_batch_size)
        for mode in ["category_pe", "firstcategory_pe", "query_pe"]:
            create_configs(working_directory, "coveo-sl-search", mode, "product_sku_hash", wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                           data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, "item_id_type", gpu_id, log_wandb,
                           train_batch_size)


    exp_name = "ml-20m"
    wandb_project = "plp-"+exp_name
    MAX_ITEM_LIST_LENGTH = 200
    epochs = 100
    for movielens_name in ["ml-20m", "ml-1m"]:
        for seed in seeds:
            for mode in ["title_id"]:
                create_configs(working_directory, movielens_name, mode, "title", wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                               data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, None, gpu_id, log_wandb,
                               train_batch_size)

            postfixes = ["-extended", "-first"]
            for data_name in [movielens_name+postfix for postfix in postfixes]:
                for mode in ["genre_id"]:
                    create_configs(working_directory, data_name, mode, "title_genres", wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                                   data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, "item_id_type", gpu_id, log_wandb,
                                   train_batch_size)
                for mode in ["genre_pe"]:
                    create_configs(working_directory, data_name, mode, "title", wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                                   data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, "item_id_type", gpu_id, log_wandb,
                                   train_batch_size)
            postfixes = ["-"+str(fraction) for fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
            for random in [postfix for postfix in postfixes]:
                data_name = movielens_name+"-random"
                for mode in ["genre_id"]:
                    create_configs(working_directory, data_name, mode, "title_genres", wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                                   data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, "item_id_type", gpu_id, log_wandb,
                                   train_batch_size, random )
                for mode in ["genre_pe"]:
                    create_configs(working_directory, data_name, mode, "title", wandb_project, seed, MAX_ITEM_LIST_LENGTH,
                                   data_path, checkpoint_dir, epochs, wandb_entity, use_gpu, "item_id_type", gpu_id, log_wandb,
                                   train_batch_size, random)
import os
import zipfile
from pathlib import Path

import numpy as np
import requests
import argparse

from asme_converters import Movielens1MConverter, Movielens20MConverter, CoveoConverter
from data_preparation.convert_for_recbole import convert_to_recbole
from data_preparation.create_ml_datasets import create_extended_movielens_data


def download_movielens(dataset_dir, dataset_name="ml-20m"):
    """
    Downloads and extracts the specified MovieLens dataset from GroupLens.

    Args:
        dataset_name (str): The name of the dataset to download ("ml-20m" or "ml-1m").
        dataset_dir (str): The directory to save the downloaded dataset.
    """

    if dataset_name not in ("ml-20m", "ml-1m"):
        raise ValueError("Invalid dataset name. Choose 'ml-20m' or 'ml-1m'.")

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    base_url = "http://files.grouplens.org/datasets/movielens/"
    zip_file = dataset_name+".zip"
    zip_path = os.path.join(dataset_dir, zip_file)

    if not os.path.exists(zip_path):
        print(f"Downloading {dataset_name}...")
        url = base_url + zip_file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {dataset_name} to {zip_path}")

    extracted_dir = os.path.join(dataset_dir, dataset_name)
    if not os.path.exists(extracted_dir):
        print(f"Extracting {dataset_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print(f"Extracted {dataset_name} to {extracted_dir}")
    else:
        print(f"{dataset_name} already extracted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="ml-1m", help="name of dataset:"
                                                                           "\nml-1m: MovieLens 1M dataset"
                                                                           "\nml-20m: MovieLens 20M dataset"
                                                                           "\ncoveo: Coveo dataset")
    parser.add_argument("--dataset_path", type=str, default="./datasets/", help="path to save datasets")
    parser.add_argument("--random", type=bool, default=False, help="create randomized non-item pages for movielens data")
    parser.add_argument("--use_original_split", type=bool, default=True, help="use original split for movielens 20m")
    args, _ = parser.parse_known_args()

    print(args)
    np.random.seed(42)
    dataset = args.dataset
    if args.dataset in ["ml-1m", "ml-20m"]:
        if not os.path.exists(args.dataset_path+"raw/"+args.dataset):
            os.makedirs(args.dataset_path, exist_ok=True)
            download_movielens(args.dataset_path+"raw/", args.dataset)
        if args.dataset == "ml-1m":
            converter = Movielens1MConverter()
            converter.apply(input_dir=args.dataset_path+"raw/",output_file=Path(args.dataset_path+"temp/"+dataset))
            convert_to_recbole(dataset, args.dataset_path + "temp/" + dataset, args.dataset_path + "final/" + dataset)
        if args.dataset == "ml-20m":
            converter = Movielens20MConverter(use_original_split=args.use_original_split)
            converter.apply(input_dir=args.dataset_path+"raw/",output_file=Path(args.dataset_path+"temp/"+dataset))
            convert_to_recbole(dataset, args.dataset_path + "temp/" + dataset, args.dataset_path + "final/" + dataset)
        else:
            print(f"{args.dataset} folder already exists.")

        #extended version
        for stage in ["test","train","validation"]:
            print("Create extended data for", stage, args.dataset)
            create_extended_movielens_data(input_dir=args.dataset_path+"temp/"+dataset,
                                           output_dir=args.dataset_path+"temp/"+dataset+"-extended",
                                           name=dataset,final_name=dataset+"-extended",
                                           stage=stage, modified_pages="genres", fraction=1.0)
        convert_to_recbole(dataset+"-extended", args.dataset_path+"temp/"+dataset+"-extended", args.dataset_path+"final/"+dataset+"-extended")
        for stage in ["test","train","validation"]:
            print("Create extended data for", stage, args.dataset)
            create_extended_movielens_data(input_dir=args.dataset_path+"temp/"+dataset,
                                           output_dir=args.dataset_path+"temp/"+dataset+"-first",
                                           name=dataset,final_name=dataset+"-first",
                                           stage=stage, modified_pages="first", fraction=1.0)
        convert_to_recbole(dataset+"-first", args.dataset_path+"temp/"+dataset+"-first", args.dataset_path+"final/"+dataset+"-first")
        if args.random:
            for fraction in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
                for stage in ["test","train","validation"]:
                    print("Create extended data for", stage, args.dataset, "random", fraction)
                    create_extended_movielens_data(input_dir=args.dataset_path+"temp/"+dataset,
                                                   output_dir=args.dataset_path+"temp/"+dataset+"-random-"+str(fraction),
                                                   name=dataset, final_name=dataset+"-random-"+str(fraction),
                                                   stage=stage, modified_pages="random", fraction=fraction)
                print("Convert to Recbole")
                convert_to_recbole(dataset+"-extended", args.dataset_path+"temp/"+dataset+"-random-"+str(fraction), args.dataset_path+"final/"+dataset+"-random-"+str(fraction))
        print("Delete temp files")
        os.system("rm -r "+args.dataset_path+"temp/")

    elif args.dataset == "coveo-search":
        filter_immediate_duplicates = True
        end_of_train: int = 1552138259347  # timestamp for ~ 70/15/15 split, based on item interactions
        end_of_validation: int = 1553704815974
        min_sequence_length: int = 2
        min_item_feedback: int = 1

        dataset_name= "coveo-sl"
        print("Prepare", dataset_name)
        converter = CoveoConverter(end_of_train, end_of_validation, min_item_feedback, min_sequence_length,
                                   False, dataset_name, True, False,
                                   filter_immediate_duplicates, delimiter="\t")
        converter.apply(input_dir=args.dataset_path+"/train", output_dir=Path(args.dataset_path+"/temp/"+dataset_name))
        print("Convert", dataset_name)
        convert_to_recbole(dataset_name, args.dataset_path + "/temp/" + dataset_name, args.dataset_path + "/final/" + dataset_name)

        dataset_name = "coveo-sl-search"
        print("Prepare", dataset_name)
        converter = CoveoConverter(end_of_train, end_of_validation, min_item_feedback, min_sequence_length,
                                   False,dataset_name, True, True,
                                   filter_immediate_duplicates, delimiter="\t")
        converter.apply(input_dir=args.dataset_path+"/train", output_dir=Path(args.dataset_path+"/temp/"+dataset_name))
        print("convert", dataset_name)
        convert_to_recbole(dataset_name, args.dataset_path + "/temp/" + dataset_name, args.dataset_path + "/final/" + dataset_name)
        print("Delete temp files")
        os.system("rm -r "+args.dataset_path+"/temp/")

    elif args.dataset == "coveo-pageview":
        filter_immediate_duplicates = True
        end_of_train: int = 1552138259347  # timestamp for ~ 70/15/15 split, based on item interactions
        end_of_validation: int = 1553704815974
        min_sequence_length: int = 2
        min_item_feedback: int = 1

        dataset_name = "coveo"
        print("Prepare", dataset_name)
        converter = CoveoConverter(end_of_train, end_of_validation, min_item_feedback, min_sequence_length,
                                   False, dataset_name, False, False,
                                   filter_immediate_duplicates, delimiter="\t")
        converter.apply(input_dir=args.dataset_path+"/train", output_dir=Path(args.dataset_path+"/temp/"+dataset_name))
        print("Convert", dataset_name)
        convert_to_recbole(dataset_name, args.dataset_path + "/temp/" + dataset_name, args.dataset_path + "/final/" + dataset_name)

        dataset_name = "coveo-pageview"
        print("Prepare", dataset_name)
        converter = CoveoConverter(end_of_train, end_of_validation, min_item_feedback, min_sequence_length,
                                   True, dataset_name, False, False,
                                   filter_immediate_duplicates, delimiter="\t")
        converter.apply(input_dir=args.dataset_path+"/train", output_dir=Path(args.dataset_path+"/temp/"+dataset_name))
        print("Convert", dataset_name)
        convert_to_recbole(dataset_name, args.dataset_path + "/temp/" + dataset_name, args.dataset_path + "/final/" + dataset_name)
        print("Delete temp files")
        os.system("rm -r "+args.dataset_path+"/temp/")





import os
import zipfile
from pathlib import Path

import numpy as np
import requests
import argparse

from asme_converters import Movielens1MConverter, Movielens20MConverter
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
    #parser.add_argument("--download", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default="ml-1m", help="name of dataset:"
                                                                           "\nml-1m: MovieLens 1M dataset"
                                                                           "\nml-20m: MovieLens 20M dataset"
                                                                           "\ncoveo: Coveo dataset")
    parser.add_argument("--dataset_path", type=str, default="./datasets/", help="path to save datasets")
    parser.add_argument("--random", type=bool, default=False, help="create randomized non-item pages for movielens data")
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
            converter = Movielens20MConverter()
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
        #delete temp files
        print("Delete temp files")
        os.system("rm -r "+args.dataset_path+"temp/")



from pathlib import Path
import random

import pandas as pd

import typer
import torch

from data_preparation.data_prep import read_csv


import os

def random_value_from_column(column):
    # return a random value from the pandas column

    return random.choice(column)

def remove_consecutive_same_genre(group_df):
    return group_df.loc[group_df['genres'] != group_df['genres'].shift()]
def create_extended_movielens_data(input_dir, output_dir, name, stage,modified_pages = "genres", fraction = 1.0):
    file_type = ".csv"
    encoding = "latin-1"
    delimiter = "\t"
    item_df = read_csv(input_dir, f'{name}.{stage}', file_type, "\t", header=0, encoding=encoding)
    item_df["title_genres"] = item_df["title"]
    item_df["title_uid"] = item_df["title"]
    item_df["old_title"] = item_df["title"]
    item_df["item_id_type"] = 1
    if modified_pages == "genres":
        page_df_mod = item_df.copy()
        page_df_mod["old_title"] = page_df_mod["title"]
        page_df_mod["title"] = "OVERVIEW-PAGE"
        page_df_mod["title_genres"] = page_df_mod["genres"]
        page_df_mod["title_uid"] = page_df_mod["userId"]
        page_df_mod["item_id_type"] = 0
        page_df_mod["rating"] = -1
        page_df_mod["year"] = 0
    if modified_pages == "random":
        page_df_mod = item_df.copy()

        random_part = page_df_mod.sample(frac=fraction)
        random_part["genres"] = item_df["genres"].sample(frac=fraction).values
        useful_part = page_df_mod.drop(random_part.index)
        page_df_mod = pd.concat([random_part,useful_part], ignore_index=True)
        page_df_mod["old_title"] = page_df_mod["title"]
        page_df_mod["title"] = "OVERVIEW-PAGE"
        page_df_mod["title_genres"] = page_df_mod["genres"] #random.choices(page_df_mod["genres"], k=len(page_df_mod))#page_df_mod['genres'].apply(random_value_from_column)
        page_df_mod["title_uid"] = page_df_mod["userId"]
        page_df_mod["item_id_type"] = 0
        page_df_mod["rating"] = -1
        page_df_mod["year"] = 0
    if modified_pages == "first":
        page_df_mod = item_df.copy()
        page_df_mod["old_title"] = page_df_mod["title"]
        page_df_mod["title"] = "OVERVIEW-PAGE"
        page_df_mod["title_genres"] = page_df_mod["genres"]
        page_df_mod["title_uid"] = page_df_mod["userId"]
        page_df_mod["item_id_type"] = 0
        page_df_mod["rating"] = -1
        page_df_mod["year"] = 0
        page_df_mod = page_df_mod.groupby('userId').apply(remove_consecutive_same_genre).reset_index(drop=True)
    item_df['original_order'] = item_df.groupby(['userId', 'timestamp']).cumcount() + 1
    page_df_mod['original_order'] = page_df_mod.groupby(['userId', 'timestamp']).cumcount() + 1
    item_df = pd.concat([item_df,page_df_mod], ignore_index=True)
    item_df = item_df.sort_values(["userId","timestamp","original_order","item_id_type"])
    if name == "ml-1m":
        item_df = item_df[['userId', 'rating', 'timestamp', 'gender', 'age',
                       'occupation', 'title', 'genres', 'year', 'user_all', 'title_genres', 'title_uid', 'item_id_type','zip']]
    else:
        item_df = item_df[['userId', 'rating', 'timestamp', 'title', 'genres', 'year', 'title_genres', 'title_uid', 'item_id_type']]
    os.makedirs(output_dir, exist_ok=True)
    item_df.to_csv(f'{output_dir}/{name+"-extended"}.{stage}{file_type}', sep=delimiter, index=False)


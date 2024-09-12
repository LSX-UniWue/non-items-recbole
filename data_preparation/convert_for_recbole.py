import os
from pathlib import Path

import numpy as np
import pandas as pd
import typer
app = typer.Typer()



@app.command()

def create(name: str = typer.Argument(..., help='dataset name, e.g. coveo, ml-20m, ml-20m-extended'),
           input_dir: Path = typer.Argument(..., help='the path to the config file', exists=True),
           output_dir: Path = typer.Argument(..., help='the path to the config file', exists=False)):

    converters = {}
    float_cols = []
    float_seq_cols = []
    if name == "coveo":
        float_cols = ["server_timestamp_epoch_ms"]
        token_cols = ["session_id_hash", "product_sku_hash", "hashed_url"]
        token_seq_cols = ["category_hash"]
        dtype = {"server_timestamp_epoch_ms": int, "session_id_hash": str, "product_sku_hash": str,
               "category_hash":str}

    if name == "coveo-sl":
        float_cols = ["server_timestamp_epoch_ms"]
        token_cols = ["session_id_hash", "product_sku_hash", "hashed_url"]
        token_seq_cols = ["category_hash"]
        dtype = {"server_timestamp_epoch_ms": int, "session_id_hash": str, "product_sku_hash": str,
                 "category_hash":str}

    if name == "coveo-pageview":
        #session_id_hash product_sku_hash    server_timestamp_epoch_ms   category_hash   item_id_type	category_product_id
        float_cols = ["server_timestamp_epoch_ms", "item_id_type"]
        token_cols = ["session_id_hash", "product_sku_hash", "category_product_id"]
        token_seq_cols = ["category_hash"]
        vocab_cols = ['product_sku_hash:token', "category_product_id:token", "item_id_type:float"]
        dtype = {"server_timestamp_epoch_ms": int, "session_id_hash": str, "product_sku_hash": str,
                 "category_hash": str, "category_product_id":str, "item_id_type":int}

    if name == "coveo-sl-search":
        float_cols = ["server_timestamp_epoch_ms", "item_id_type"]
        token_cols = ["session_id_hash", "product_sku_hash", "category_product_id", "first_result_product", "first_result_cat"]
        float_seq_cols = ["query_vector"]
        token_seq_cols = ["category_hash"]
        vocab_cols = ['product_sku_hash:token', "category_product_id:token", "item_id_type:float", "first_result_product:token", "first_result_cat:token"]
        dtype = {"server_timestamp_epoch_ms": int, "session_id_hash": str, "product_sku_hash": str,
                 "category_hash": str, "category_product_id":str, "item_id_type":int, "first_result_product":str, "first_result_cat":str, "query_vector":str}

    if name == "ml-20m-extended":
        float_cols = ["timestamp","item_id_type"]
        token_cols = ["userId", "title", "title_genres"]
        vocab_cols = ['title:token', "title_genres:token", "item_id_type:float"]
        token_seq_cols = ["genres"]
        dtype = {"timestamp": int, "userId": str, "title_genres": str, "title": str, "genres":str, "item_id_type":int}

    if name == "ml-20m":
        float_cols = ["timestamp"]
        token_cols = ["userId", "movieId","title"]
        #vocab_cols = ['title:token', "title_genres:token", "item_id_type:float"]
        token_seq_cols = ["genres"]
        dtype= {"timestamp": int, "userId": str, "movieId": str, "title": str, "genres":str}

    output_name = name + "-recbole"
    output_dir = output_dir / output_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        if file.endswith("train.csv") or file.endswith("test.csv") or file.endswith("validation.csv"):
            print(input_dir/file)
            df = pd.read_csv(input_dir / file, header=0, sep="\t",engine='pyarrow',dtype=dtype)
            #rename columns
            float_cols_dict = {c: c + ":float" for c in float_cols}
            df = df.rename(columns=float_cols_dict)
            token_cols_dict = {c: c + ":token" for c in token_cols}
            df = df.rename(columns=token_cols_dict)
            token_seq_cols_dict = {c: c + ":token_seq" for c in token_seq_cols}
            df = df.rename(columns=token_seq_cols_dict)
            float_seq_cols_dict = {c: c + ":float_seq" for c in float_seq_cols}
            df = df.rename(columns=float_seq_cols_dict)
            df = df[list(float_cols_dict.values()) +list(token_cols_dict.values()) +list(token_seq_cols_dict.values())+list(float_seq_cols_dict.values())]

            if file.endswith("test.csv") or file.endswith("validation.csv"):
                if name in ["coveo-pageview","coveo-sl-search"]: #ensure last entry of a session has item_id_type as 1 for test and validation
                    df = filter_for_last_item(df, file, name)
            if name in ["coveo-sl-search"]:
                df["query_vector:float_seq"] = df["query_vector:float_seq"].str.replace('[', '',regex=True)
                df["query_vector:float_seq"] = df["query_vector:float_seq"].str.replace(']', '',regex=True)
                df["query_vector:float_seq"] = df["query_vector:float_seq"].str.replace(',', ' ',regex=True)
                #.apply(convert_to_list_float)
                df["first_result_cat_attr:token_seq"] = df["first_result_cat:token"].str.replace('\\', ' ',regex=True)
                df["category_hash:token_seq"] = df["category_hash:token_seq"].str.replace('\\', ' ',regex=True)
            df_name = output_name+"."+file.split(".")[1]+".inter"
            df.to_csv(output_dir / df_name, sep="\t", index=False, header=True)

            if file.endswith("train.csv") and name in ["ml-20m-extended", "coveo-pageview", "coveo-sl-search"]:
               #create vocabulary from df
                df_name = output_name+".item"
                df = df[vocab_cols].drop_duplicates()
                df.to_csv(output_dir/df_name, sep="\t", index=False, header=True)


def filter_for_last_item(df, file, name):
    if file.endswith("test.csv") or file.endswith("validation.csv"):
        df = df.sort_values(by=['session_id_hash:token', 'server_timestamp_epoch_ms:float'])
        result_df = pd.DataFrame(columns=df.columns)
        grouped_df = df.groupby('session_id_hash:token')
        for name, group in grouped_df:
            last_nonzero_index = group[group['item_id_type:float'].ne(0)].drop_duplicates(
                subset='session_id_hash:token', keep='last').index
            if not last_nonzero_index.empty:
                last_nonzero_index = last_nonzero_index[0]
                result_df = pd.concat([result_df, group.loc[:last_nonzero_index]])
        df = result_df
    return df

def convert_to_list_float(value):
    if value == '[]':
        raise ValueError("Empty list")
    else:
        return np.array([float(x) for x in value.strip('[]').split(', ')])

if __name__ == "__main__":
    app()
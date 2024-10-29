from __future__ import annotations

import os
import random

import pandas as pd
from pathlib import Path
import json
import re

from loguru import logger
from tqdm import tqdm
from random import choice

from recbole.data.chat_convert.model import Dataset

train_file = Path("Datasets/ReDial/train_data.jsonl")
test_file = Path("Datasets/ReDial/test_data.jsonl")


def seed_everything(seed: int) -> int:
    """Stolen from https://pytorch-lightning.readthedocs.io/en/1.7.7/_modules/pytorch_lightning/utilities/seed.html#seed_everything

    Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
    """
    if seed is None:
        raise ValueError(
            "seed_everything() called without a seed. A seed must be provided or set as an environment variable PL_GLOBAL_SEED")
    elif not isinstance(seed, int):
        seed = int(seed)

    logger.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        logger.info(f"numpy seed set to {seed}")
    except ImportError:
        logger.warning("Could not set numpy seed.")
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"PyTorch seed set to {seed}")
    except ImportError:
        logger.warning("Could not set torch seed.")
        pass

    return seed


seed_everything(42)

mention_regex = re.compile(r"@[0-9]+")
model = 'all-MiniLM-L6-v2'

abstracts = json.loads(Path("Datasets/ReDial/abstracts.json").read_text())
noisy_titles = json.loads(Path("Datasets/ReDial/filtered_llama_titles.json").read_text())


def get_noisy_title(title):
    if title not in noisy_titles:
        logger.warning(f"Could not find noisy title for {title}")
        return title
    return choice(noisy_titles[title])


def convert_file(file, debug: bool | int = False, split_tokens: bool = False, noise: bool = False,
                 mentions_first: bool = False):
    if isinstance(debug, bool):
        debug = 10 if debug else 0
    dataset = Dataset(model, name="redial")
    session_id = -1
    max_session_length = 0

    lines = file.read_text().splitlines()
    for line in tqdm(lines, total=debug or len(lines)):
        if debug and session_id > debug:
            break
        session_id += 1
        data = json.loads(line)
        session_lenth = 0
        movie_mentions = data["movieMentions"]
        user = data["initiatorWorkerId"]
        recommender = data["respondentWorkerId"]
        for message in data["messages"]:
            text: str = message["text"]
            mentions = mention_regex.findall(text)
            for mention in mentions:
                try:
                    movie_title = movie_mentions[mention[1:]]
                    if noise:
                        movie_title = get_noisy_title(movie_title).strip('"').strip(".").strip()
                    else:
                        movie_title = re.sub(r"\([0-9]{4}\)", "", movie_title).strip()
                except KeyError as e:
                    logger.warning(f"Could not find movie title for mention {mention}: {e}")
                    movie_title = mention
                text = text.replace(mention, movie_title)

            if mentions_first:
                session_lenth = add_mentions(dataset, mentions, message, movie_mentions, recommender, session_id,
                                             session_lenth)

            dataset.add_turn(
                timestamp=message["timeOffset"],
                item_id_type=0,
                userId=session_id,
                title="USER-MESSAGE",
                message=text,
                split_tokens=split_tokens,
            )
            session_lenth += 1
            if not mentions_first:
                session_lenth = add_mentions(dataset, mentions, message, movie_mentions, recommender, session_id,
                                             session_lenth)

            if session_lenth > max_session_length:
                max_session_length = session_lenth

    return dataset


def add_mentions(dataset, mentions, message, movie_mentions, recommender, session_id, session_lenth):
    for mention in mentions:
        try:
            movie_title = movie_mentions[mention[1:]]
        except KeyError as e:
            logger.warning(f"Could not find movie title for mention {mention}: {e}")
            movie_title = mention
        try:
            abstract = f"{movie_title}: {abstracts[movie_title]}"
        except:
            logger.warning(f"Could not find abstract for movie {movie_title}")
            abstract = movie_title
        if pd.isna(abstract):
            abstract = movie_title
        dataset.add_turn(
            timestamp=message["timeOffset"],
            item_id_type=1 if message["senderWorkerId"] == recommender else 2,
            userId=session_id,
            title=mention,
            message=abstract,
        )
        session_lenth += 1
    return session_lenth


debug = False
split_tokens = True
noise = True
mentions_first = False

train_val = convert_file(train_file, debug=debug, split_tokens=split_tokens, noise=noise, mentions_first=mentions_first)
train = Dataset(model, {k: v for k, v in train_val._sessions.items() if int(k) < int(len(train_val.sessions) * 0.8)},
                name="redial")
val = Dataset(model, {k: v for k, v in train_val._sessions.items() if int(len(train_val.sessions) * 0.8) <= int(k)},
              name="redial")
test = convert_file(test_file, debug=debug, split_tokens=split_tokens, noise=noise, mentions_first=mentions_first)

dataset: Dataset
for split, dataset in zip(("train", "val", "test"), (train, val, test)):
    dataset.cut_sessions()
    dataset_id = f"redial{'_tokens' if split_tokens else ''}{'_noise' if noise else ''}{'_debug' if debug else ''}{'' if not mentions_first else '_mentions_first'}"

    dataset_file = Path(f"Datasets/ReDial/{dataset_id}.{split}.json")
    dataset_file.write_text(dataset.to_json())
    dataset.to_inter(f"Datasets/ReDial/{dataset_id}.{split}.inter", save_sample=True)
    dataset.to_inter(f"Datasets/ReDial/{dataset_id}_items_only.{split}.inter", dataset.only_item_turns,
                     save_sample=True)

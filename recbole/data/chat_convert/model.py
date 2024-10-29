from __future__ import annotations

import pickle
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Deque, Type, Optional

import pandas as pd
from dataclasses_json import dataclass_json, DataClassJsonMixin
from loguru import logger
from sentence_transformers import SentenceTransformer

OFFSET = 0.000001

logger.remove()
logger.add(sys.stderr, level="INFO")


@dataclass_json
@dataclass
class Turn:
    timestamp: float
    item_id_type: int
    userId: str
    title: str
    embedding: str
    message: str


@dataclass_json
@dataclass
class Session:
    userId: str
    turns: Deque[Turn]

    def is_valid(self, items_only: bool = False):
        if len(self.turns) < 2:
            return False
        if items_only and len([turn for turn in self.turns if turn.item_id_type == 1]) < 2:
            return False
        if self.turns[-1].item_id_type != 1:
            return False
        return True

    def cut_nonitems_at_end(self):
        while len(self.turns) > 0 and self.turns[-1].item_id_type == 0:
            self.turns.pop()

    def has_timestamp(self, timestamp: float):
        return any(-OFFSET < turn.timestamp - timestamp < OFFSET for turn in self.turns)


class DummyEmbeddingModel:
    def encode(self, text):
        return [0.0] * 768


class DiskCachedEmbeddingModel:
    def __init__(self, model: str, cache_path: Path = None, dataset: str = ""):
        self.model = SentenceTransformer(model)
        if cache_path is None:
            cache_path = Path(f"{model}{('_' + dataset) if dataset else ''}.pickle")
        self.cache_path = cache_path
        if cache_path.exists():
            self.cache = pickle.load(cache_path.open("rb"))
        else:
            self.cache = {}

    def encode(self, text):
        if text not in self.cache:
            self.cache[text] = self.model.encode(text)
            if random.randint(0, 100) == 0:
                logger.debug(f"Cache size: {len(self.cache)}")
                with self.cache_path.open("wb") as f:
                    pickle.dump(self.cache, f)
            logger.debug(f"Cache miss: {text}")
        else:
            logger.debug(f"Cache hit: {text}")

        return self.cache[text]


@dataclass_json
@dataclass
class Dataset:
    _sessions: Dict[str, Session]
    embedding_model: str = None

    def __init__(self, embedding_model: Optional[str], _sessions: Dict[str, Session] = None, name: str = ""):
        self._sessions = _sessions or {}
        self.embedding_model = embedding_model
        self._embedding_model = DiskCachedEmbeddingModel(embedding_model,
                                                         dataset=name) if embedding_model else DummyEmbeddingModel()

    def add_turn(self, timestamp: float, item_id_type: int, userId: str, title: str, message: str,
                 split_tokens: bool = False):
        if not split_tokens:
            self._add_turn_full(item_id_type, message, timestamp, title, userId)
        else:
            self._add_turn_split(item_id_type, message, timestamp, title, userId)

    def _add_turn_full(self, item_id_type, message, timestamp, title, userId, check_timestamp: bool = True):
        embedding = ",".join(map(str, self._embedding_model.encode(message)))
        if userId not in self._sessions:
            self._sessions[userId] = Session(userId, deque())
        if check_timestamp:
            while self._sessions[userId].has_timestamp(timestamp):
                logger.debug(f"Duplicate timestamp: {timestamp}. Adding {OFFSET} as offset.")
                timestamp += OFFSET
        self._sessions[userId].turns.append(Turn(timestamp, item_id_type, userId, title, embedding, message))

    def _add_turn_split(self, item_id_type, message, timestamp, title, userId):
        tokens = message.split()
        for i, token in enumerate(tokens):
            self._add_turn_full(item_id_type, token, timestamp + i * OFFSET, title, userId, check_timestamp=False)

    def cut_sessions(self):
        for session in self._sessions.values():
            session.cut_nonitems_at_end()

    @property
    def sessions(self):
        return [session for session in self._sessions.values()]

    @property
    def filtered_sessions(self):
        return [session for session in self._sessions.values() if session.is_valid()]

    @property
    def filtered_sessions_items_only(self):
        return [session for session in self._sessions.values() if session.is_valid(items_only=True)]

    @property
    def turns(self):
        return [turn for session in self._sessions.values() for turn in session.turns]

    @property
    def filtered_turns(self):
        return [turn for session in self.filtered_sessions for turn in session.turns]

    @property
    def only_item_turns(self):
        return [turn for session in self.filtered_sessions_items_only for turn in session.turns if
                turn.item_id_type != 0]

    def to_df(self, turns=None):
        if turns is None:
            turns = self.filtered_turns
        df = pd.DataFrame(turns)
        df.columns = ["timestamp:float", "item_id_type:float", "userId:token", "title:token", "embedding:float_seq",
                      "message:token"]
        df["detailed_item_id_type:float"] = df["item_id_type:float"]
        df["item_id_type:float"] = df["item_id_type:float"].astype(bool).astype(int)
        return df

    def to_inter(self, inter_file: Path | str, turns=None, save_sample: bool = False):
        inter_file = Path(inter_file)
        if turns is None:
            turns = self.filtered_turns
        df = self.to_df(turns)
        inter_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(inter_file, sep="\t", index=False)
        if "train" in inter_file.name:
            df[["item_id_type:float", "title:token", "message:token"]].drop_duplicates().to_csv(
                inter_file.with_suffix(".item"), sep="\t", index=False)

            aggregated = df.groupby(["userId:token"]).size()
            suffix = inter_file.suffix
            new_name = inter_file.name.replace(suffix, "_stats.csv")
            aggregated.to_csv(inter_file.parent / new_name, sep="\t", header=False)
            new_name = inter_file.name.replace(suffix, "_90.txt")
            (inter_file.parent / new_name).write_text(aggregated.quantile(0.9).astype(str))

        if save_sample:
            df[:100].to_csv(Path(inter_file).with_suffix(".head.tsv"), sep="\t", index=False)

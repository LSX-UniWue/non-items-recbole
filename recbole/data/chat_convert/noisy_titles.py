import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from prompting import LLM, llama31_70b, gpt4mini, llama31_8b, YesNoLLM

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")

abstracts_file = Path("../Datasets/ReDial/abstracts.json")
llama_titles_file = Path("../Datasets/ReDial/llama_titles.json")
filtered_llama_titles_file = Path("../Datasets/ReDial/filtered_llama_titles.json")
gpt_titles_file = Path("../Datasets/ReDial/gpt_titles.json")

yes_no_system_prompt = """You will be passed a pair of original movie title and a noisy version of it. Your task is to determine if the noisy version is a valid reference that could be used in a conversation. This can be either the original title, a slightly modified version or a description. If you think the noisy version is a valid alternative, return "yes". If you think the noisy version is not a valid alternative, return "no". Do not return anything else."""
yes_no_llama = YesNoLLM(llama31_8b, yes_no_system_prompt, cache_maxsize=5000, max_tries=10)


def generate_noisy_titles():
    system_prompt = """Return a name for this movie as it would likely be used in a conversation. For example, if the movie is called "The Matrix", you might return "Matrix". For 'The Lord of the Rings: The Fellowship of the Ring', you might return "The first Lord of the Rings movie". Return only one possible name, nothing else"""
    gpt = LLM(gpt4mini, system_prompt)
    llama = LLM(llama31_8b, system_prompt)

    llama_titles = defaultdict(list)
    gpt_titles = {}

    abstracts = json.loads(abstracts_file.read_text())
    titles = list(abstracts.keys())
    for title in tqdm(titles):
        print("Movie ID:", title)
        for i in range(20):
            # gpt_title = gpt(title)
            llama_title = llama(title)
            # print("GPT:", gpt_title)
            print("LLAMA:", llama_title)
            # if quoted title: extract title
            # if len(llama_title) > 50:
            #    logger.warning(f"Title {llama_title} too long, skipping")
            #    continue
            if "\n" in llama_title:
                logger.warning(f"Title {llama_title} contains newline, skipping")
                continue
            search = re.search(r'"[^"]*"', llama_title)
            if search:
                llama_title = search.group(0)
                logger.info(f"Extracted title: {llama_title}")
            llama_title = llama_title.strip(".")
            llama_titles[title].append(llama_title)
            # gpt_titles[title] = gpt_title

    llama_titles_file.write_text(json.dumps(llama_titles))
    gpt_titles_file.write_text(json.dumps(gpt_titles))


def filter_noisy_titles():
    noisy_titles = json.loads(llama_titles_file.read_text())

    filtered_noisy_titles = defaultdict(list)
    for movie_id, titles in tqdm(noisy_titles.items()):
        logger.info(f"Movie Title: {movie_id}")
        for title in titles:
            try:
                response, reason = yes_no_llama(f"The original title is: {movie_id}. The noisy title is: {title}")
            except ValueError:
                logger.warning(f"Could not get a response for title {title}")
                response = False
                reason = "No response"
            reason_oneline = reason.replace('\n', ' ')
            logger.info(f"Revised title: {title}, Response: {response}, Reason: {reason_oneline}")
            if response:
                filtered_noisy_titles[movie_id].append(title)
                logger.info(f"Accepted title: {title}")
            else:
                logger.info(f"Rejected title: {title}")

    filtered_llama_titles_file.write_text(json.dumps(filtered_noisy_titles))


if __name__ == '__main__':
    """
    for movie_id, title in (
            ("The Lord of the Rings: The Fellowship of the Ring", "The first Lord of the Rings movie"),
            ("Spider-Man: No Way Home", "No Way Home"),
    ):
        print(yes_no_llama(f"The original title is: {movie_id}. The noisy title is: {title}"))
    """

    filter_noisy_titles()

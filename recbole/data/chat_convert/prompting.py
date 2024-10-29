from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import T

import openai
from loguru import logger
from openai import NOT_GIVEN
from pydantic import BaseModel

OLLAMA_URL = "<OLLAMA_URL>"
OPENAI_API_KEY = "<YOUR KEY HERE>"

openai.api_key = OPENAI_API_KEY


@dataclass
class LLMArchitecture:
    name: str
    model: str
    max_tokens: int
    openai: bool

    @property
    def estimated_num_characters(self):
        return int(self.max_tokens * 3.5)


llama31_70b = LLMArchitecture("llama3.1:70b", "llama3.1:70b", -1, False)
llama31_8b = LLMArchitecture("llama3.1:8b", "llama3.1:8b", -1, False)
# gpt4o = LLMArchitecture("gpt-4o", "gpt-4o", 128000, True)
gpt4mini = LLMArchitecture("gpt-4o-mini", "gpt-4o-mini", 128000, True)


class Language(BaseModel):
    name: str


class LLM:
    def __init__(self, model: LLMArchitecture, system_prompt: str, post_prompt: str = None, cache_maxsize: int = 0,
                 seed=NOT_GIVEN, json: bool = False, output_format: T[BaseModel] = None):
        if model.openai:
            self.base_url = None
        else:
            self.base_url = OLLAMA_URL
            if output_format is not None:
                example_for_schema = get_example_for_schema(output_format, system_prompt)
                logger.warning(
                    f"output_format is only supported for OpenAI models. Adding JSON description to post_prompt: {example_for_schema}")
                if post_prompt is None:
                    post_prompt = ""
                post_prompt += example_for_schema
                json = True
                # output_format = None
        self.model = model
        self.system_prompt = system_prompt
        self.post_prompt = ("...\n\n" + post_prompt) if post_prompt else ""
        self.cache_maxsize = cache_maxsize
        self.seed = seed
        self.json = json
        self.output_format = output_format

        self.__call__ = lru_cache(maxsize=self.cache_maxsize)(self.__call__)

    def __call__(self, prompt) -> str | T:
        if self.base_url:
            openai.base_url = self.base_url
        try:
            model_max_chars = self.model.estimated_num_characters
            if model_max_chars > -1 and len(prompt) > model_max_chars:
                prompt = prompt[:model_max_chars]
                logger.warning(f"Truncated prompt to {model_max_chars} characters.")
            prompt += self.post_prompt
            logger.debug(f"System prompt is: {self.system_prompt}")
            shortened_prompt = prompt if len(prompt) < 1000 else prompt[:1000] + '\n...\n' + prompt[-1000:]
            logger.debug(f"Querying {self.model}, seed {self.seed} with prompt: {shortened_prompt}")
            # response = openai.chat.completions.create(model=self.model.model,
            response = openai.beta.chat.completions.parse(model=self.model.model,
                                                          messages=[{"role": "system", "content": self.system_prompt},
                                                                    {"role": "user",
                                                                     "content": prompt}], seed=self.seed,
                                                          response_format=self.output_format if self.output_format is not None else {
                                                              "type": "json_object"} if self.json else {
                                                              "type": "text"})
            if self.output_format is not None:
                content = response.choices[0].message.parsed
            else:
                content = response.choices[0].message.content
            logger.debug(f"Model {self.model} says: {content}")
            return content
        finally:
            openai.base_url = None


class YesNoReason(BaseModel):
    answer: bool
    reason: str


class YesNo(BaseModel):
    answer: bool


class YesNoLLM(LLM):
    def __init__(self, model: LLMArchitecture, system_prompt: str, post_prompt: str = None, cache_maxsize: int = 0,
                 seed=NOT_GIVEN, max_tries=3, reason: bool = False):
        if model.openai:
            super().__init__(model, system_prompt, post_prompt, cache_maxsize, seed=seed,
                             output_format=YesNoReason if reason else YesNo)
        else:
            if not post_prompt or "yes" not in post_prompt:
                logger.warning("Post prompt does not contain 'yes'. Adding instruction.")
                post_prompt = post_prompt + "Answer only with yes or no, nothing else." if not reason else "Answer with yes or no, followed by a reason"
            super(YesNoLLM, self).__init__(model, system_prompt,
                                           post_prompt,
                                           cache_maxsize,
                                           seed=seed, json=False)
        self.max_tries = max_tries
        self.reason = reason

    def __call__(self, prompt) -> (bool, str):
        if self.model.openai:
            response = super().__call__(prompt)
            return response
        tries = 0
        while tries < self.max_tries:
            response = super().__call__(prompt)
            response_lower = response.lower()
            if response_lower.startswith("yes") or response_lower.startswith("no"):
                answer = response_lower.startswith("yes")
                if self.reason:
                    return YesNoReason(answer=answer, reason=response)
                else:
                    return YesNo(answer=answer)
            tries += 1
        raise ValueError(f"Could not get a yes/no answer after {self.max_tries} tries.")


# @cache_to_disk(maxsize=-1)
def get_example_for_schema(schema: BaseModel, query: str):
    language_llm = LLM(gpt4mini, system_prompt="Return the English name of the language of the provided text.", seed=42,
                       output_format=Language)

    language = language_llm(query).name

    system_prompt = f"Your job is to guide a language model in generating a response in the provided output format. Your instructions should be in {language}. Address the language model informally."
    instruction_gpt = LLM(gpt4mini, system_prompt, seed=42)
    example_gpt = LLM(gpt4mini, system_prompt, seed=42, output_format=schema)

    example: schema = example_gpt("Give an example of the output format you want the model to generate.")
    return instruction_gpt(
        "Start by telling the model 'Your output should match this format:'.") + "\n" + example.json()


if __name__ == '__main__':
    gpt = YesNoLLM(llama31_70b, "This is a test prompt.", seed=42, reason=True)
    response = gpt("What is the meaning of life?")
    print(response)

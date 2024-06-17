import json
import os
from typing import Any
from dotenv import load_dotenv

import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

from modules.definitions import ROOT
from modules.rag.loading import chroma_loading


class UrbAssistant:
    """
    LLM assistant for answering questions about city objects.
    """

    def __init__(self, model_name: str, device_map: str = 'auto', max_new_tokens: int = 4096, **kwargs) -> None:
        """Initialize LLM and tokenizer for assistant.

        Args:
            model_name (str): Model name for initialization.
            device_map (str, optional): Device for computation. Defaults to 'auto'.
            max_new_tokens (int, optional): The maximum numbers of tokens to generate, 
            ignoring the number of tokens in the prompt. Defaults to 4096.
        """
        load_dotenv(ROOT / 'config.env')
        self._max_new_tokens = max_new_tokens
        self._system_prompt = None
        self._database = None  # Database for RAG ## TODO

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=device_map, token=os.environ.get('HF_TOKEN'), **kwargs)
        self._model.eval()

    def set_sys_prompt(self, new_prompt: str) -> None:
        self._system_prompt = new_prompt

    def init_retirever(self, pth: str, collection_name: str):
        chroma_loading(pth=pth, collection_name=collection_name)

    def __call__(self,
                 user_question: str,
                 temperature: float = .8,
                 top_p: float = .8,
                 *args: Any,
                 **kwargs: Any) -> str:
        """Generate response from LLM using retrieved context.

        Args:
            user_question (str): A user question that requires an answer.
            temperature (float, optional): Generation temperature. 
            The higher ,the less stable answers will be. Defaults to .8.
            top_p (float, optional): Nuclear sampling. Selects the most likely tokens from a probability distribution, 
            considering the cumulative probability until it reaches a predefined threshold “top_p”. Defaults to .8.

        Returns:
            str: Model's answer to user question. 
            Model responses in following format: "EXPLANATION": explanation to given answer. 
            "ANSWER": Model's answer to asked question.
        """
        # formatted_prompt = self._database.retrieve_context(user_question)
        messages = [{'role': 'system', 'content': self._system_prompt},
                    {'role': 'user', 'content': user_question}]
        terminators = [self._tokenizer.eos_token_id,
                       self._tokenizer.convert_tokens_to_ids('<|eot_id|>')]
        input_ids = self._tokenizer.apply_chat_template(messages,
                                                        add_generation_prompt=True,
                                                        return_tensors='pt').to(self._model.device)
        output = self._model.generate(input_ids,
                                      max_new_tokens=self._max_new_tokens,
                                      eos_token_id=terminators,
                                      pad_token_id=self._tokenizer.eos_token_id,
                                      do_sample=True,
                                      temperature=temperature)
        response = output[0][input_ids.shape[-1]:]
        return self._tokenizer.decode(response, skip_special_tokens=True)


class WebAssistant:
    """
    Web implementation of LLM assistant for answering urbanistic questions.
    """

    def __init__(self) -> None:
        """
        Initialize an instanse of LLM assistant.
        """
        load_dotenv(ROOT / 'config.env')
        self._system_prompt = None
        self._context = None
        self._url = os.environ.get('SAIGA_URL')

    def set_sys_prompt(self, new_prompt: str) -> None:
        """Set model's role and generation instructions.

        Args:
            new_prompt (str): New instructions.
        """
        self._system_prompt = new_prompt

    def add_context(self, context: str) -> None:
        """Add a context to model's prompt

        Args:
            context (str): context related to question.
        """
        self._context = context

    def __call__(self, user_question: str,
                 temperature: float = .015,
                 top_p: float = .5,
                 *args: Any,
                 **kwargs: Any) -> str:
        """Get a response from model for given question.

        Args:
            user_question (str): A user's prompt. Question that requires an answer.
            temperature (float, optional): Generation temperature. 
            The higher ,the less stable answers will be. Defaults to 0.015.
            top_p (float, optional): Nuclear sampling. Selects the most likely tokens from a probability distribution, 
            considering the cumulative probability until it reaches a predefined threshold “top_p”. Defaults to 0.5.

        Returns:
            str: Model's answer to user's question. 
        """
        formatted_prompt = {'system': self._system_prompt,
                            'user': user_question,
                            'context': self._context,
                            'temperature': temperature,
                            'top_p': top_p}
        response = requests.post(url=self._url, json=formatted_prompt)
        if kwargs.get('as_json'):
            return json.loads(response.text)['response']
        else:
            return response.text
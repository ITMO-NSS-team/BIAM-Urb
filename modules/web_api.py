import json
import os
from typing import Any
from dotenv import load_dotenv
import requests

from modules.definitions import ROOT

class WebAssistant:
    """
    Web implementation of LLM assistant for answering urbanistic questions.
    """
    def __init__(self) -> None:
        load_dotenv(ROOT / 'config.env')
        self._system_prompt = None
        self._context = None
        self._url = os.environ.get('SAIGA_URL')

    def set_sys_prompt(self, new_prompt: str) -> None:
        self._system_prompt = new_prompt

    def add_context(self, context: str) -> None:
        self._context = context

    def __call__(self, user_question:str, 
                 temperature: float = .015,
                 top_p: float  = .5,
                 *args: Any,
                 **kwargs: Any) -> str:
        formatted_prompt = {'system': self._system_prompt,
                            'user': user_question,
                            'context': self._context,
                            'temperature': temperature, 
                            'top_p': top_p}
        response = requests.post(url=self._url, json=formatted_prompt)
        # if kwargs.get('as_json'):
        #     return json.loads(response.text)['response']
        # else:
        #     return response.text
        return json.loads(response.text)['response']
        
import os
from typing import Any
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv
from openai import OpenAI

from modules.definitions import ROOT, SYS_PROMPT


class CustomLLMEval(DeepEvalBaseLLM):
    """
    Implementation of Evaluation agent based on large language model for Assistant's answers evaluation.
    """

    def __init__(self,
                 model: str,
                 sys_prompt: str = SYS_PROMPT,
                 *args,
                 **kwargs):
        """Initialize instance with evaluation LLM.

        Args:
            model (str, optional): Evaluation model's name.
        """
        load_dotenv(ROOT / 'config.env')
        self._sys_prompt = sys_prompt
        self._model_name = model
        self.model = self.load_model(*args, **kwargs)

    def load_model(self, *args, **kwargs):
        """
        Load model's instance.
        """
        # TODO extend pull of possible LLMs (Not only just OpenAI's models)
        return OpenAI(api_key=os.environ.get('VSE_GPT_KEY'), *args, **kwargs)

    def generate(self, prompt: str, context: str = None, temperature: float = .015, *args, **kwargs) -> str:
        """Get a response form LLM to given question.

        Args:
            prompt (str): User's question, the model must answer.
            context (str, optional): Supplementary information, may be used for answer.
            temperature (float, optional): Determines randomnes and diversity of generated answers. 
            The higher temperature, the mode diverse answers is. Defaults to .015.

        Returns:
            str: Model's response for user's question.
        """
        formatted_message = [{'role': 'system', 'content': self._sys_prompt},
                             {'role': 'user', 'content': f'Вопрос:{prompt} Контекст:{context}'}]
        response = self.model.chat.completions.create(model=self._model_name,
                                                      messages=formatted_message,
                                                      temperature=temperature,
                                                      n=1,
                                                      max_tokens=8182)
        return response.choices[0].message.content

    async def a_generate(self, prompt: str, context: str = None, temperature: float = .015, *args, **kwargs) -> str:
        return self.generate(prompt, context, temperature, *args, **kwargs)

    def get_model_name(self, *args, **kwargs) -> str:
        return "Implementation of cusom LLM for evaluation."

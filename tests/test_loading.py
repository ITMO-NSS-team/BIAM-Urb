from glob import glob
import json
import random
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (AnswerRelevancyMetric,
                              ContextualPrecisionMetric,
                              ContextualRecallMetric,
                              ContextualRelevancyMetric,
                              FaithfulnessMetric)

from experiments.evaluation import CustomLLMEval
from modules.definitions import ROOT
from modules.utils.dataset import load_data


def test_ans_relevancy():
    """
    Test if a LLM's answers are actually correct by comparing how llm's answer relevant to provided expected answer.
    """
    dataset_pth = ROOT / 'data' / 'answers' / 'LLAMA_DATASET_FOR_EVAL'
    ans_relevancy_metric = AnswerRelevancyMetric(model=CustomLLMEval(model='openai/gpt-4-turbo',
                                                                     base_url='https://api.vsegpt.ru/v1',
                                                                     sys_prompt=""),
                                                 verbose_mode=False)
    # Read all files with llm answers and build evaluation dataset.
    eval_dataset = EvaluationDataset()
    for json_file in dataset_pth.glob('*.json'):
        with open(json_file) as pth:
            data = json.load(pth)
        test_case = LLMTestCase(input=data['query'],
                                actual_output=data['llama_answer'],
                                context=[str(data['context'])])
        eval_dataset.add_test_case(test_case)
    eval_dataset.evaluate([ans_relevancy_metric])
    return


def test_answers_w_context():
    dataset_pth = ROOT / 'data' / 'answers' / 'LLAMA_DATASET_FOR_EVAL'
    model = CustomLLMEval(model='openai/gpt-4-turbo',
                          base_url='https://api.vsegpt.ru/v1',
                          sys_prompt="")
    context_precision = ContextualPrecisionMetric(model=model)
    context_recall = ContextualRecallMetric(model=model)
    context_relevancy = ContextualRelevancyMetric(model=model)
    faithfulness = FaithfulnessMetric(model=model)
    metrics = [context_precision, context_recall, context_relevancy, faithfulness]
    # Read all files with llm answers and build evaluation dataset.
    eval_dataset = EvaluationDataset()
    for json_file in dataset_pth.glob('*.json'):
        with open(json_file) as pth:
            data = json.load(pth)
        test_case = LLMTestCase(input=data['query'],
                                actual_output=data['llama_answer'],
                                context=[str(data['context'])])
        eval_dataset.add_test_case(test_case)
    eval_dataset.evaluate(metrics)


if __name__ == "__main__":
    ans = test_ans_relevancy()
    ans_new = test_answers_w_context()
    print('Hold')

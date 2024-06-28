from glob import glob
import json
import random
import re
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (AnswerRelevancyMetric,
                              ContextualPrecisionMetric,
                              ContextualRecallMetric,
                              ContextualRelevancyMetric,
                              FaithfulnessMetric)
import pandas as pd

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
                                                 verbose_mode=True)

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
    model = CustomLLMEval(model='anthropic/claude-3-haiku', #'openai/gpt-4-turbo',
                          base_url='https://api.vsegpt.ru/v1',
                          sys_prompt="")
    context_precision = ContextualPrecisionMetric(
        model=model, verbose_mode=True, async_mode=False)
    context_recall = ContextualRecallMetric(model=model)
    context_relevancy = ContextualRelevancyMetric(
        model=model, async_mode=False, verbose_mode=True, include_reason=True)
    faithfulness = FaithfulnessMetric(model=model, async_mode=False, verbose_mode=True, include_reason=True)
    metrics = [context_relevancy]
    # Read all files with llm answers and build evaluation dataset.
    dataset = pd.read_csv(ROOT / 'data' / 'full_strategy_answers.csv')
    res = {'query': [], 'llm_ans': [], 'context': [], 'score': []}
    for _, row in dataset.iterrows():
        query = row['query']
        actual_out = row['llm_ans']
        context = row['context']
        test_case = LLMTestCase(input=query, actual_output=actual_out, retrieval_context=[context])
        metric = faithfulness.measure(test_case)
        res['query'].append(query)
        res['llm_ans'].append(actual_out)
        res['context'].append(context)
        res['score'].append(metric)
    # eval_dataset = EvaluationDataset()
    # eval_dataset.add_test_cases_from_csv_file(ROOT / 'data' / 'full_strategy_answers.csv',
    #                                           input_col_name='query',
    #                                           actual_output_col_name='llm_ans',
    #                                           retrieval_context_col_name='context', )
    # for json_file in dataset_pth.glob('*.json'):
    #     with open(json_file) as pth:
    #         data = json.load(pth)
    #     test_case = LLMTestCase(input=data['query'],
    #                             actual_output=data['llama_answer'],
    #                             context=[str(data['context'])])
    #     eval_dataset.add_test_case(test_case)
    # eval_dataset.evaluate(metrics)
    res_df = pd.DataFrame.from_dict(res)
    print('Done')


if __name__ == "__main__":
    # ans = test_ans_relevancy()
    ans_new = test_answers_w_context()
    print('Hold')

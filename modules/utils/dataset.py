import json
import geojson

from pathlib import Path
from typing import Dict, Optional


from modules.definitions import ROOT


def load_data(idx: int, top_n: Optional[int] = None) -> Dict:
    """Load list of possible questions and related context.
    Takes number of chunk to load as index and loads qiestions-context pair.

    Args:
        id (int): number of chunk.
        top_n (int, optional): Amount of questions to be taken. By default takes all questions.

    Returns:
        Dict: Dictionary with list of questions related to certain context, expected answer,  
        and context related to these questions.
    """
    question_context = {'questions': [],
                        'targets': [],
                        'context': None}
    with open(Path(ROOT, 'data', 'datasets', f'data_{idx}.json'), 'r') as json_data:
        questions = json.load(json_data)
        questions = [questions[q_id] for q_id in list(questions.keys())[:top_n]]
    with open(Path(ROOT, 'data', 'buildings', f'buildings_part_{idx}.geojson'), 'r') as buildings_data:
        question_context['context'] = geojson.load(buildings_data)

    for pair in questions:
        question_context['questions'].append(pair['query'])
        question_context['targets'].append(pair['response'])

    return question_context


if __name__ == "__main__":
    # TODO move to unit tests
    tst = load_data(1)
    print('test')
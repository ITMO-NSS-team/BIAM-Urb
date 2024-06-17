import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import geojson

from modules.definitions import ROOT


def load_geojson_data(id: int, top_n: Optional[int] = None) -> Dict:
    """Load list of possible questions and related context.
    Takes number of chunk to load as index and loads qiestions-context pair.

    Args:
        id (int): number of chunk.

    Returns:
        Dict: Dictionary with list of questions related to certain context, 
        and context related to these questions.
    """
    question_context = {'questions': [],
                        'targets': [],
                        'context': None}
    with open(Path(ROOT, 'data', 'datasets', f'data_{id}.json')) as json_data:
        questions = json.load(json_data)
    with open(Path(ROOT, 'data', 'buildings', f'buildings_part_{id}.geojson')) as buildings_data:
        question_context['context'] = geojson.load(buildings_data)

    for question_id in list(questions.keys()):
        question_context['questions'].append(questions[question_id]['query'])
        question_context['targets'].append(questions[question_id]['response'])

    return question_context


if __name__ == "__main__":
    tst = load_geojson_data(1)
    print('test')
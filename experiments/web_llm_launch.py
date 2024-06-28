import pathlib
from modules.definitions import ROOT
from modules.models import WebHostedLlm
from modules.utils.dataset import load_data


if __name__ == '__main__': 
    amount = 2
    top_n = 10
    model = WebHostedLlm(model_url='http://10.32.2.2:8672/v1/chat/completions')
    questions_target = load_data(1, top_n=top_n)
    context = str(questions_target['context'])
    question = questions_target['questions'][1]
    expected_answer = questions_target['targets'][1]
    response = model.generate(question, context, top_p=.05, temperature=0.01).split('EXPLANATION:')[0].replace('ANSWER:', '')
    print(response)
    print('Done.')
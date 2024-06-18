import pathlib
from modules.definitions import ROOT
from modules.models import WebAssistant
from modules.utils.dataset import load_data


if __name__ == '__main__': 
    amount = 2
    top_n = 10
    model = WebAssistant()

    for i in range(amount):
        questions_target = load_data(i, top_n=top_n)
        context = str(questions_target['context'])
        for pair_i in range(top_n):
            question = questions_target['questions'][pair_i]
            expected_answer = questions_target['targets'][pair_i]
            response = model(question, context, top_p=.05, temperature=0.01).split('EXPLANATION:')[0].replace('ANSWER:', '')
            print('Hold')

    print('Done.')
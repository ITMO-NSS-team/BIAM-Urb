from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BitsAndBytesConfig

from chroma_rag.loading import ChromaConnector
from modules.model import UrbAssistant
from modules.web_api import WebAssistant
from modules.standard_prompt import standard_sys_prompt


class Question(BaseModel):
    question_body: str


app = FastAPI()


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}
#
#
# @app.get("/question/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


@app.post('/question')
def read_item(question: Question):
    collection_name = 'strategy'
    chroma_inst = ChromaConnector()
    res = chroma_inst.chroma_view(question.question_body, collection_name)
    context = res[0][0].page_content

    model = WebAssistant()
    model.set_sys_prompt(standard_sys_prompt)
    model.add_context(context)
    res = model(question.question_body, as_json=True)

    # bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    # assistant = UrbAssistant('meta-llama/Meta-Llama-3-8B-Instruct', quantization_config=bnb_config)
    # assistant.set_sys_prompt(standard_sys_prompt)
    # user_message = f'Question:{question.question_body}\nContext:{context}'
    # ans = assistant(user_message, temperature=0.015, top_p=.05)
    # return {"answer": ans.split('ANSWER: ')[-1]}
    return {'res': res}

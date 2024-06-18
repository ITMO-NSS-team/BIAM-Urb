from fastapi import FastAPI
from pydantic import BaseModel

import chroma_rag.loading as chroma_connector
from models.web_api import WebAssistant
from models.standard_prompt import standard_sys_prompt


class Question(BaseModel):
    question_body: str


app = FastAPI()

@app.post('/question')
async def read_item(question: Question):
    """Get a response for a given question using a RAG pipeline with a vector DB and LLM.

    Args:
        question (Question): A question from the user (natural language, no additional prompts).

    Returns:
        dict: pipeline's answer to the user's question.
    """
    collection_name = 'strategy-spb'
    res = chroma_connector.chroma_view(question.question_body, collection_name)
    context = res[0][0].page_content

    model = WebAssistant()
    model.set_sys_prompt(standard_sys_prompt)
    model.add_context(context)
    res = model(question.question_body, as_json=True)
    return {'res': res}

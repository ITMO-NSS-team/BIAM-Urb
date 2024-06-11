from typing import Union
from fastapi import FastAPI

import chromadb
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer

chroma_client = chromadb.HttpClient(host='10.32.1.34', port=9941)
ef = HuggingFaceEmbeddingServer(url="http://10.32.1.34:9942/embed")

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/question/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/question")
def read_item(question: str):
    collection = chroma_client.get_collection("my_collection2", embedding_function=ef)
    results = collection.query(
        query_texts=[question],  # Chroma will embed this for you
        n_results=2  # how many results to return
    )
    return {'res': results}

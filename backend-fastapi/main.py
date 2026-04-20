from fastapi import FastAPI
from pydantic import BaseModel
from scripts.search_chroma import search

app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    k: int = 5


@app.get("/")
def root():
    return {"message": "Chroma search API is running"}


@app.post("/search")
def search_endpoint(request: SearchRequest):
    results = search(request.query, request.k)
    return {
        "query": request.query,
        "k": request.k,
        "results": results
    }